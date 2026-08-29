"""分别评价固定模型在 validation 与 test 上的生存表现。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import hydra
import matplotlib
import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter
from omegaconf import DictConfig
from sksurv.compare import compare_survival
from sksurv.metrics import (
    brier_score,
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv

from trails.artifacts import save_json
from trails.data import ClinicalTimeSeriesDataset
from trails.prediction import TrailsPrediction
from trails_simulate.config import resolved_payload

from .config import MimicEvaluationConfig
from .data import BASELINE_COVARIATE_COLUMNS
from .paths import resolve_input_path, resolve_output_path

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

RACE_GROUPS = ("WHITE", "BLACK", "ASIAN", "HISPANIC_OR_LATINO", "OTHER_OR_UNKNOWN")


def _group_race(values: pd.Series) -> pd.Series:
    race = values.fillna("UNKNOWN").astype(str).str.upper()
    grouped = np.select(
        [
            race.str.contains("WHITE"),
            race.str.contains("BLACK"),
            race.str.contains("ASIAN"),
            race.str.contains("HISPANIC|LATINO", regex=True),
        ],
        list(RACE_GROUPS[:-1]),
        default=RACE_GROUPS[-1],
    )
    return pd.Series(grouped, index=values.index, dtype="string")


def _load_predictions(
    split_dir: Path,
) -> tuple[pd.DataFrame, TrailsPrediction, ClinicalTimeSeriesDataset]:
    dataset = ClinicalTimeSeriesDataset.load(split_dir / "dataset.pt")
    prediction = TrailsPrediction.load(split_dir / "model_prediction.pt")
    raw_patient_ids = dataset.metadata.get("patient_ids")
    if not isinstance(raw_patient_ids, list):
        raise ValueError(f"数据集缺少 patient_ids：{split_dir}")
    patient_ids = [str(value) for value in raw_patient_ids]
    pred_cluster = prediction.predict().numpy().astype(np.int64, copy=False)
    risk_score = prediction.risk_score().numpy().astype(np.float64, copy=False)
    survival_time = torch.stack(
        [dataset[index].survival_time for index in range(len(dataset))]
    ).numpy()
    event = torch.stack([dataset[index].event for index in range(len(dataset))]).numpy()
    if (
        len({len(patient_ids), len(pred_cluster), len(risk_score), len(survival_time), len(event)})
        > 1
    ):
        raise ValueError(f"数据集与预测长度不一致：{split_dir}")
    if len(set(patient_ids)) != len(patient_ids):
        raise ValueError(f"数据集存在重复 patient_id：{split_dir}")
    if not np.isfinite(np.column_stack((risk_score, survival_time, event))).all():
        raise ValueError(f"数据集或预测存在非有限数值：{split_dir}")
    if np.any(survival_time <= 0) or not np.isin(event, [0.0, 1.0]).all():
        raise ValueError(f"数据集的时间或事件编码无效：{split_dir}")
    frame = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "pred_cluster": pred_cluster,
            "risk_score": risk_score,
            "survival_time": survival_time,
            "event": event,
        }
    )
    raw_covariates = dataset.metadata.get("baseline_covariates")
    if not isinstance(raw_covariates, list):
        raise ValueError(f"数据集缺少 baseline_covariates：{split_dir}")
    covariates = pd.DataFrame(raw_covariates)
    required_covariates = {"patient_id", *BASELINE_COVARIATE_COLUMNS}
    if missing := sorted(required_covariates - set(covariates.columns)):
        raise ValueError(f"基线协变量缺少字段：{missing}")
    covariates["patient_id"] = covariates["patient_id"].astype(str)
    if set(covariates["patient_id"]) != set(patient_ids):
        raise ValueError(f"基线协变量与数据集 patient_id 不一致：{split_dir}")
    frame = frame.merge(
        covariates.loc[:, ["patient_id", *BASELINE_COVARIATE_COLUMNS]],
        on="patient_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    return frame, prediction, dataset


class SurvivalCalibration:
    """计算并绘制分位数组 Kaplan–Meier 生存校准结果。"""

    def __init__(
        self,
        event: np.ndarray,
        followup: np.ndarray,
        predicted_survival: np.ndarray,
        times: np.ndarray,
        n_bins: int,
    ) -> None:
        self.event = event
        self.followup = followup
        self.predicted_survival = predicted_survival
        self.times = times
        self.n_bins = n_bins
        self._table: pd.DataFrame | None = None
        self._weighted_absolute_errors: dict[str, float] | None = None

    def calculate(self) -> tuple[pd.DataFrame, dict[str, float]]:
        """返回逐组校准表和各时间点的加权绝对误差。"""
        if self._table is not None and self._weighted_absolute_errors is not None:
            return self._table.copy(), dict(self._weighted_absolute_errors)

        rows: list[dict[str, float | int]] = []
        weighted_errors: dict[str, float] = {}
        for time_index, calibration_time in enumerate(self.times):
            predicted = self.predicted_survival[:, time_index]
            groups = np.asarray(
                pd.qcut(
                    predicted,
                    q=min(self.n_bins, len(self.event)),
                    labels=False,
                    duplicates="drop",
                ),
                dtype=np.float64,
            )
            if np.isnan(groups).all():
                groups = np.zeros(len(self.event), dtype=np.float64)

            time_rows: list[dict[str, float | int]] = []
            for bin_index in sorted(int(value) for value in np.unique(groups[~np.isnan(groups)])):
                selected = groups == bin_index
                km_time, km_survival = kaplan_meier_estimator(
                    self.event[selected],
                    self.followup[selected],
                )[:2]
                km_index = int(np.searchsorted(km_time, calibration_time, side="right") - 1)
                observed = 1.0 if km_index < 0 else float(km_survival[km_index])
                mean_predicted = float(predicted[selected].mean())
                time_rows.append(
                    {
                        "time": float(calibration_time),
                        "bin": bin_index,
                        "n_patients": int(selected.sum()),
                        "mean_predicted_survival": mean_predicted,
                        "observed_survival_km": observed,
                        "absolute_error": abs(mean_predicted - observed),
                    }
                )
            rows.extend(time_rows)
            weighted_errors[str(calibration_time)] = float(
                np.average(
                    [row["absolute_error"] for row in time_rows],
                    weights=[row["n_patients"] for row in time_rows],
                )
            )

        self._table = pd.DataFrame(rows)
        self._weighted_absolute_errors = weighted_errors
        return self._table.copy(), dict(weighted_errors)

    def plot(self, output_stem: Path, display_times: Sequence[float]) -> dict[str, str]:
        """绘制指定时间点的预测生存率与 KM 观察生存率。"""
        if not display_times:
            raise ValueError("display_times must not be empty")
        table, weighted_errors = self.calculate()
        figure, raw_axes = plt.subplots(
            1,
            len(display_times),
            figsize=(4.2 * len(display_times), 4.0),
            squeeze=False,
            layout="constrained",
        )
        axes = raw_axes.ravel()
        for axis, display_time in zip(axes, display_times, strict=True):
            selected = np.isclose(table["time"].to_numpy(dtype=np.float64), display_time)
            if not selected.any():
                plt.close(figure)
                raise ValueError(f"校准结果不包含绘图时间点：{display_time}")
            panel = table.loc[selected].sort_values("mean_predicted_survival")
            axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="0.5", linewidth=1.0)
            axis.plot(
                panel["mean_predicted_survival"],
                panel["observed_survival_km"],
                marker="o",
                color="#0072B2",
                linewidth=1.5,
            )
            axis.set(
                title=(
                    f"Day {display_time:g}\n"
                    f"Weighted absolute error = {weighted_errors[str(display_time)]:.3f}"
                ),
                xlabel="Mean predicted survival",
                ylabel="Kaplan–Meier observed survival",
                xlim=(0.0, 1.0),
                ylim=(0.0, 1.0),
            )
            axis.set_aspect("equal", adjustable="box")

        output_stem.parent.mkdir(parents=True, exist_ok=True)
        png_path = output_stem.with_suffix(".png")
        pdf_path = output_stem.with_suffix(".pdf")
        figure.savefig(png_path, dpi=220)
        figure.savefig(pdf_path)
        plt.close(figure)
        return {"calibration_plot_png": str(png_path), "calibration_plot_pdf": str(pdf_path)}


class AdjustedCoxAnalysis:
    """拟合簇效应的调整后 Cox 模型并绘制森林图。"""

    def __init__(self, frame: pd.DataFrame, reference_cluster: int) -> None:
        self.frame = frame
        self.reference_cluster = reference_cluster
        self._table: pd.DataFrame | None = None
        self._summary: dict[str, Any] | None = None

    def fit(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """返回相对统一参考簇的调整后 HR、95% CI 和 p 值。"""
        if self._table is not None and self._summary is not None:
            return self._table.copy(), dict(self._summary)

        analysis = self.frame.loc[
            :, ["pred_cluster", "survival_time", "event", *BASELINE_COVARIATE_COLUMNS]
        ].copy()
        age = cast(pd.Series, pd.to_numeric(analysis["age"], errors="coerce")).astype(float)
        sofa = cast(pd.Series, pd.to_numeric(analysis["sofa_score"], errors="coerce")).astype(float)
        analysis["age_per_10_years"] = age / 10.0
        analysis["sofa_score"] = sofa
        analysis["gender"] = analysis["gender"].fillna("UNKNOWN").astype(str).str.upper()
        analysis["race_group"] = _group_race(cast(pd.Series, analysis["race"]))
        analysis = analysis.drop(columns=["age", "race"]).dropna(
            subset=["age_per_10_years", "sofa_score"]
        )
        occupied = sorted(int(value) for value in analysis["pred_cluster"].unique())
        if self.reference_cluster not in occupied:
            raise ValueError(f"Cox参考簇{self.reference_cluster}在当前评价集中没有患者")
        if len(occupied) < 2:
            raise ValueError("调整后Cox至少需要两个有患者的簇")

        formula = (
            f"C(pred_cluster, Treatment(reference={self.reference_cluster}))"
            " + age_per_10_years + C(gender) + C(race_group) + sofa_score"
        )
        fitter = CoxPHFitter()
        fitter.fit(
            analysis,
            duration_col="survival_time",
            event_col="event",
            formula=formula,
        )
        model_summary = cast(pd.DataFrame, fitter.summary)
        cluster_terms = model_summary.loc[
            model_summary.index.astype(str).str.startswith("C(pred_cluster"),
            ["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"],
        ].copy()
        cluster_terms["pred_cluster"] = [
            int(str(term).rsplit("[T.", maxsplit=1)[1].removesuffix("]"))
            for term in cluster_terms.index
        ]
        cluster_terms = cluster_terms.rename(
            columns={
                "exp(coef)": "hazard_ratio",
                "exp(coef) lower 95%": "ci_lower_95",
                "exp(coef) upper 95%": "ci_upper_95",
                "p": "p_value",
            }
        )
        counts = analysis["pred_cluster"].value_counts()
        reference = pd.DataFrame(
            {
                "pred_cluster": [self.reference_cluster],
                "hazard_ratio": [1.0],
                "ci_lower_95": [1.0],
                "ci_upper_95": [1.0],
                "p_value": [np.nan],
            }
        )
        table = pd.concat([reference, cluster_terms.reset_index(drop=True)], ignore_index=True)
        table["n_patients"] = table["pred_cluster"].map(counts).astype(int)
        table["is_reference"] = table["pred_cluster"] == self.reference_cluster
        table = table.sort_values("pred_cluster").reset_index(drop=True)
        self._table = table
        self._summary = {
            "reference_cluster": self.reference_cluster,
            "reference_rule": "lowest mean risk_score in train",
            "adjustment_covariates": ["age_per_10_years", "gender", "race_group", "sofa_score"],
            "race_groups": list(RACE_GROUPS),
            "n_input": len(self.frame),
            "n_complete_case": len(analysis),
            "n_excluded_missing_numeric": len(self.frame) - len(analysis),
            "concordance_index": float(fitter.concordance_index_),
        }
        return table.copy(), dict(self._summary)

    def plot(self, output_stem: Path) -> dict[str, str]:
        """绘制仅包含簇效应的调整后 HR 森林图。"""
        table, _ = self.fit()
        y_positions = np.arange(len(table))
        hazard_ratio = table["hazard_ratio"].to_numpy(dtype=np.float64)
        ci_lower = table["ci_lower_95"].to_numpy(dtype=np.float64)
        ci_upper = table["ci_upper_95"].to_numpy(dtype=np.float64)
        figure, axis = plt.subplots(figsize=(6.4, 0.65 * len(table) + 2.2), layout="constrained")
        axis.errorbar(
            hazard_ratio,
            y_positions,
            xerr=np.vstack((hazard_ratio - ci_lower, ci_upper - hazard_ratio)),
            fmt="o",
            color="#0072B2",
            capsize=3,
        )
        axis.axvline(1.0, linestyle="--", color="0.5", linewidth=1.0)
        axis.set(
            title="Adjusted mortality hazard ratios",
            xlabel="Hazard ratio (log scale)",
            yticks=y_positions,
            yticklabels=[
                f"Cluster {cluster} (reference)" if is_reference else f"Cluster {cluster}"
                for cluster, is_reference in zip(
                    table["pred_cluster"].to_numpy(dtype=np.int64),
                    table["is_reference"].to_numpy(dtype=bool),
                    strict=True,
                )
            ],
        )
        axis.set_xscale("log")
        lower_limit = max(0.05, float(ci_lower.min()) * 0.8)
        upper_limit = float(ci_upper.max()) * 1.2
        tick_values = np.power(
            2.0,
            np.arange(np.floor(np.log2(lower_limit)), np.ceil(np.log2(upper_limit)) + 1),
        )
        tick_values = tick_values[(tick_values >= lower_limit) & (tick_values <= upper_limit)]
        axis.set_xlim(lower_limit, upper_limit)
        axis.set_xticks(tick_values, labels=[f"{value:g}" for value in tick_values])
        axis.minorticks_off()
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.2)
        png_path = output_stem.with_suffix(".png")
        pdf_path = output_stem.with_suffix(".pdf")
        figure.savefig(png_path, dpi=220)
        figure.savefig(pdf_path)
        plt.close(figure)
        return {"adjusted_cox_plot_png": str(png_path), "adjusted_cox_plot_pdf": str(pdf_path)}


class ClusterClinicalCharacteristics:
    """生成 Overall 与各预测簇的临床特征描述表。"""

    def __init__(self, frame: pd.DataFrame, n_clusters: int) -> None:
        self.frame = frame
        self.n_clusters = n_clusters

    @staticmethod
    def _mean_sd(values: pd.Series) -> str:
        observed = cast(pd.Series, pd.to_numeric(values, errors="coerce")).dropna()
        if observed.empty:
            return "NA"
        observed_values = observed.to_numpy(dtype=np.float64)
        standard_deviation = (
            0.0 if len(observed_values) == 1 else float(observed_values.std(ddof=1))
        )
        return f"{observed_values.mean():.1f} ({standard_deviation:.1f})"

    @staticmethod
    def _median_iqr(values: pd.Series) -> str:
        observed = cast(pd.Series, pd.to_numeric(values, errors="coerce")).dropna()
        if observed.empty:
            return "NA"
        quantiles = observed.quantile([0.25, 0.5, 0.75]).to_numpy(dtype=np.float64)
        return f"{quantiles[1]:.1f} [{quantiles[0]:.1f}, {quantiles[2]:.1f}]"

    @staticmethod
    def _count_percent(selected: pd.Series, denominator: int) -> str:
        count = int(selected.sum())
        return "0 (NA)" if denominator == 0 else f"{count} ({100.0 * count / denominator:.1f}%)"

    def calculate(self) -> pd.DataFrame:
        """返回适合直接审阅的临床特征宽表。"""
        data = self.frame.loc[:, ["pred_cluster", *BASELINE_COVARIATE_COLUMNS]].copy()
        data["gender"] = data["gender"].fillna("UNKNOWN").astype(str).str.upper()
        data["race_group"] = _group_race(cast(pd.Series, data["race"]))
        groups = [("overall", data)] + [
            (f"cluster_{cluster}", data.loc[data["pred_cluster"] == cluster])
            for cluster in range(self.n_clusters)
        ]

        rows: list[dict[str, str]] = []

        def append_row(characteristic: str, level: str, values: dict[str, str]) -> None:
            rows.append({"characteristic": characteristic, "level": level, **values})

        append_row("Patients", "n", {name: str(len(group)) for name, group in groups})
        append_row(
            "Age, years",
            "mean (SD)",
            {name: self._mean_sd(cast(pd.Series, group["age"])) for name, group in groups},
        )
        append_row(
            "Age missing",
            "n (%)",
            {name: self._count_percent(group["age"].isna(), len(group)) for name, group in groups},
        )
        for level in ("F", "M", "UNKNOWN"):
            append_row(
                "Gender",
                level,
                {
                    name: self._count_percent(group["gender"] == level, len(group))
                    for name, group in groups
                },
            )
        for level in RACE_GROUPS:
            append_row(
                "Race",
                level,
                {
                    name: self._count_percent(group["race_group"] == level, len(group))
                    for name, group in groups
                },
            )
        append_row(
            "SOFA at sepsis onset",
            "median [IQR]",
            {
                name: self._median_iqr(cast(pd.Series, group["sofa_score"]))
                for name, group in groups
            },
        )
        append_row(
            "SOFA missing",
            "n (%)",
            {
                name: self._count_percent(group["sofa_score"].isna(), len(group))
                for name, group in groups
            },
        )
        return pd.DataFrame(rows)


class ClusterTrajectoryAnalysis:
    """在临床单位下汇总并绘制各预测簇的纵向轨迹。"""

    def __init__(
        self,
        dataset: ClinicalTimeSeriesDataset,
        clusters: np.ndarray,
        preprocessing: pd.DataFrame,
        n_clusters: int,
        bin_hours: float,
    ) -> None:
        self.dataset = dataset.with_return_kind("aligned")
        self.clusters = clusters
        self.preprocessing = preprocessing
        self.n_clusters = n_clusters
        self.bin_hours = bin_hours
        self._table: pd.DataFrame | None = None

    def calculate(self) -> pd.DataFrame:
        """按患者分箱后返回簇级 median、IQR 和样本量。"""
        if self._table is not None:
            return self._table.copy()
        if not np.isclose(48.0 / self.bin_hours, round(48.0 / self.bin_hours)):
            raise ValueError("trajectory_bin_hours 必须整除48小时")
        if len(self.dataset) != len(self.clusters):
            raise ValueError("轨迹数据集与预测簇长度不一致")
        required = {"feature", "center", "scale"}
        if missing := sorted(required - set(self.preprocessing.columns)):
            raise ValueError(f"预处理参数缺少字段：{missing}")
        parameters = self.preprocessing.loc[:, ["feature", "center", "scale"]].copy()
        if set(parameters["feature"]) != set(self.dataset.feature_names):
            raise ValueError("预处理参数与轨迹变量不一致")

        patient_ids = [str(value) for value in self.dataset.metadata["patient_ids"]]
        feature_names = np.asarray(self.dataset.feature_names, dtype=object)
        patient_parts: list[np.ndarray] = []
        cluster_parts: list[np.ndarray] = []
        time_parts: list[np.ndarray] = []
        feature_parts: list[np.ndarray] = []
        value_parts: list[np.ndarray] = []
        for patient_index, sample in enumerate(self.dataset.samples):
            aligned = sample.to_aligned()
            visit_index, feature_index = torch.nonzero(aligned.mask > 0, as_tuple=True)
            n_observed = int(visit_index.numel())
            patient_parts.append(np.full(n_observed, patient_ids[patient_index], dtype=object))
            cluster_parts.append(np.full(n_observed, self.clusters[patient_index], dtype=np.int64))
            time_parts.append(aligned.times[visit_index].numpy())
            feature_indices = feature_index.numpy()
            feature_parts.append(feature_names[feature_indices])
            value_parts.append(aligned.x[visit_index, feature_index].numpy())
        observations = pd.DataFrame(
            {
                "patient_id": np.concatenate(patient_parts),
                "pred_cluster": np.concatenate(cluster_parts),
                "time": np.concatenate(time_parts),
                "feature": np.concatenate(feature_parts),
                "standardized_value": np.concatenate(value_parts),
            }
        ).merge(parameters, on="feature", how="left", validate="many_to_one")
        observations["value"] = (
            observations["standardized_value"] * observations["scale"] + observations["center"]
        )
        observations["time_bin_start"] = np.minimum(
            np.floor(observations["time"] / self.bin_hours) * self.bin_hours,
            48.0 - self.bin_hours,
        )

        # 先压缩到患者层面，避免观测频率较高的患者主导簇轨迹。
        patient_level = observations.groupby(
            ["patient_id", "pred_cluster", "feature", "time_bin_start"],
            as_index=False,
            observed=True,
        ).agg(patient_value=("value", "median"), n_observations=("value", "size"))
        grouped = patient_level.groupby(
            ["feature", "time_bin_start", "pred_cluster"], observed=True
        )
        summary = grouped.agg(
            n_patients=("patient_id", "nunique"),
            n_observations=("n_observations", "sum"),
            median=("patient_value", "median"),
        )
        quantiles = grouped["patient_value"].quantile(np.array([0.25, 0.75])).unstack()
        quantiles.columns = ["q25", "q75"]
        summary = summary.join(quantiles)
        full_index = pd.MultiIndex.from_product(
            [
                self.dataset.feature_names,
                np.arange(0.0, 48.0, self.bin_hours),
                range(self.n_clusters),
            ],
            names=["feature", "time_bin_start", "pred_cluster"],
        )
        summary = summary.reindex(full_index).reset_index()
        summary["time_bin_end"] = summary["time_bin_start"] + self.bin_hours
        summary["time_bin_midpoint"] = summary["time_bin_start"] + self.bin_hours / 2.0
        summary[["n_patients", "n_observations"]] = (
            summary[["n_patients", "n_observations"]].fillna(0).astype(int)
        )
        self._table = summary
        return summary.copy()

    def plot(self, output_stem: Path) -> dict[str, str]:
        """为所有变量绘制簇中位数轨迹及患者间IQR。"""
        table = self.calculate()
        n_columns = min(4, len(self.dataset.feature_names))
        n_rows = int(np.ceil(len(self.dataset.feature_names) / n_columns))
        figure, raw_axes = plt.subplots(
            n_rows,
            n_columns,
            figsize=(14.0, 2.8 * n_rows),
            squeeze=False,
            layout="constrained",
        )
        axes = raw_axes.ravel()
        color_map = plt.get_cmap("tab10")
        for axis, feature in zip(axes, self.dataset.feature_names, strict=False):
            feature_table = table.loc[table["feature"] == feature]
            for cluster in range(self.n_clusters):
                cluster_table = feature_table.loc[feature_table["pred_cluster"] == cluster].dropna(
                    subset=["median"]
                )
                time = cluster_table["time_bin_midpoint"].to_numpy(dtype=np.float64)
                median = cluster_table["median"].to_numpy(dtype=np.float64)
                q25 = cluster_table["q25"].to_numpy(dtype=np.float64)
                q75 = cluster_table["q75"].to_numpy(dtype=np.float64)
                color = color_map(cluster % 10)
                axis.plot(time, median, color=color, label=f"Cluster {cluster}")
                axis.fill_between(time, q25, q75, color=color, alpha=0.15)
            axis.set(title=feature, xlabel="Hours after ICU admission", xlim=(0.0, 48.0))
            axis.grid(alpha=0.2)
        for axis in axes[len(self.dataset.feature_names) :]:
            axis.set_visible(False)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="outside upper center", ncols=self.n_clusters)
        figure.supylabel("Median observed value (IQR)")
        png_path = output_stem.with_suffix(".png")
        pdf_path = output_stem.with_suffix(".pdf")
        figure.savefig(png_path, dpi=220)
        figure.savefig(pdf_path)
        plt.close(figure)
        return {"trajectory_plot_png": str(png_path), "trajectory_plot_pdf": str(pdf_path)}


def _evaluate_split(
    train: pd.DataFrame,
    target: pd.DataFrame,
    prediction: TrailsPrediction,
    dataset: ClinicalTimeSeriesDataset,
    preprocessing: pd.DataFrame,
    config: MimicEvaluationConfig,
    output_dir: Path,
    evaluation_set: str,
    cox_reference_cluster: int,
) -> dict[str, Any]:
    """以 train 删失分布为参考评价一个固定目标划分。"""
    train_event = train["event"].to_numpy(dtype=bool)
    train_time = train["survival_time"].to_numpy(dtype=np.float64)
    target_event = target["event"].to_numpy(dtype=bool)
    target_followup = target["survival_time"].to_numpy(dtype=np.float64)
    risk = target["risk_score"].to_numpy(dtype=np.float64)
    target_cluster = target["pred_cluster"].to_numpy(dtype=np.int64)
    train_survival = Surv.from_arrays(train_event, train_time)
    target_survival = Surv.from_arrays(target_event, target_followup)
    auc_times = np.asarray(config.auc_times, dtype=np.float64)
    if len(auc_times) == 0 or np.any(np.diff(auc_times) <= 0):
        raise ValueError("auc_times 必须为非空严格递增序列")
    if auc_times[0] <= float(target_followup.min()) or auc_times[-1] >= float(
        target_followup.max()
    ):
        raise ValueError(f"auc_times 必须位于 {evaluation_set} 随访时间的开区间内")
    probability_times = np.asarray(config.probability_times, dtype=np.float64)
    if len(probability_times) < 2 or np.any(np.diff(probability_times) <= 0):
        raise ValueError("probability_times 必须包含至少两个严格递增时间点")
    if probability_times[0] <= float(target_followup.min()) or probability_times[-1] >= float(
        target_followup.max()
    ):
        raise ValueError(f"probability_times 必须位于 {evaluation_set} 随访时间的开区间内")

    harrell = concordance_index_censored(target_event, target_followup, risk)
    ipcw = concordance_index_ipcw(train_survival, target_survival, risk, tau=config.tau)
    dynamic_auc, mean_auc = cumulative_dynamic_auc(train_survival, target_survival, risk, auc_times)
    survival_probabilities = prediction.survival(config.probability_times).numpy()
    _, brier_values = brier_score(
        train_survival,
        target_survival,
        survival_probabilities,
        probability_times,
    )
    integrated_brier = integrated_brier_score(
        train_survival,
        target_survival,
        survival_probabilities,
        probability_times,
    )

    calibration = SurvivalCalibration(
        target_event,
        target_followup,
        survival_probabilities,
        probability_times,
        config.calibration_bins,
    )
    calibration_table, calibration_errors = calibration.calculate()

    occupied = np.sort(np.unique(target_cluster))
    logrank_chi2: float | None = None
    logrank_p: float | None = None
    logrank_stats = pd.DataFrame(index=occupied)
    if len(occupied) > 1:
        statistic, p_value, logrank_stats, _ = cast(
            tuple[float, float, pd.DataFrame, np.ndarray],
            compare_survival(target_survival, target_cluster, return_stats=True),
        )
        logrank_chi2, logrank_p = float(statistic), float(p_value)

    cluster_summary = cast(
        pd.DataFrame,
        target.groupby("pred_cluster", observed=True)
        .agg(
            n_patients=("patient_id", "size"),
            event_count=("event", "sum"),
            event_rate=("event", "mean"),
            median_followup=("survival_time", "median"),
            mean_risk_score=("risk_score", "mean"),
        )
        .reindex(range(prediction.predict_proba().shape[1])),
    )
    counts = cast(pd.Series, cluster_summary["n_patients"]).fillna(0).astype(int)
    cluster_summary = cluster_summary.assign(n_patients=counts, fraction=counts / len(target))
    cluster_summary = cluster_summary.join(logrank_stats, how="left").reset_index()
    output_dir.mkdir(parents=True, exist_ok=True)
    cluster_summary.to_csv(output_dir / "cluster_survival_summary.csv", index=False)
    pd.DataFrame({"time": probability_times, "brier_score": brier_values}).to_csv(
        output_dir / "brier_scores.csv", index=False
    )
    calibration_table.to_csv(output_dir / "calibration.csv", index=False)
    calibration_plots = calibration.plot(output_dir / "calibration", config.auc_times)

    cluster_figure, cluster_axis = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
    color_map = plt.get_cmap("tab10")
    for cluster_label in occupied:
        selected = target_cluster == cluster_label
        km_time, km_survival = kaplan_meier_estimator(
            target_event[selected],
            target_followup[selected],
        )[:2]
        cluster_axis.step(
            np.concatenate(([0.0], km_time)),
            np.concatenate(([1.0], km_survival)),
            where="post",
            color=color_map(int(cluster_label) % 10),
            label=f"Cluster {cluster_label} (n={int(selected.sum())})",
        )
    logrank_label = "not available" if logrank_p is None else f"p={logrank_p:.3g}"
    cluster_axis.set(
        title=f"Predicted-cluster survival (log-rank {logrank_label})",
        xlabel="Days after landmark",
        ylabel="Kaplan–Meier survival probability",
        xlim=(0.0, config.tau),
        ylim=(0.0, 1.02),
    )
    cluster_axis.legend(frameon=False)
    cluster_axis.grid(alpha=0.2)
    cluster_png = output_dir / "cluster_survival.png"
    cluster_pdf = output_dir / "cluster_survival.pdf"
    cluster_figure.savefig(cluster_png, dpi=220)
    cluster_figure.savefig(cluster_pdf)
    plt.close(cluster_figure)

    metrics_figure, metrics_axes = plt.subplots(
        1,
        3,
        figsize=(12.6, 3.8),
        layout="constrained",
    )
    metrics_axes[0].plot(auc_times, dynamic_auc, marker="o", color="#0072B2")
    metrics_axes[0].axhline(0.5, linestyle="--", color="0.5", linewidth=1.0)
    metrics_axes[0].set(
        title="Cumulative/dynamic AUC",
        xlabel="Days after landmark",
        ylabel="AUC",
        ylim=(0.0, 1.0),
        xticks=auc_times,
    )
    metrics_axes[1].plot(probability_times, brier_values, color="#D55E00")
    metrics_axes[1].set(
        title=f"Brier score (IBS={integrated_brier:.3f})",
        xlabel="Days after landmark",
        ylabel="Brier score",
    )
    metrics_axes[1].set_ylim(bottom=0.0)
    calibration_values = np.asarray(
        [calibration_errors[str(time)] for time in probability_times],
        dtype=np.float64,
    )
    metrics_axes[2].plot(probability_times, calibration_values, color="#009E73")
    metrics_axes[2].set(
        title="Grouped calibration error",
        xlabel="Days after landmark",
        ylabel="Weighted absolute error",
    )
    metrics_axes[2].set_ylim(bottom=0.0)
    for axis in metrics_axes:
        axis.grid(alpha=0.2)
    metrics_png = output_dir / "time_metrics.png"
    metrics_pdf = output_dir / "time_metrics.pdf"
    metrics_figure.savefig(metrics_png, dpi=220)
    metrics_figure.savefig(metrics_pdf)
    plt.close(metrics_figure)

    cox = AdjustedCoxAnalysis(target, cox_reference_cluster)
    cox_table, cox_summary = cox.fit()
    cox_table.to_csv(output_dir / "adjusted_cox.csv", index=False)
    cox_plots = cox.plot(output_dir / "adjusted_cox")
    cox_summary["outputs"] = {
        "table": str(output_dir / "adjusted_cox.csv"),
        **cox_plots,
    }
    save_json(output_dir / "adjusted_cox.json", cox_summary)
    clinical_characteristics = ClusterClinicalCharacteristics(
        target,
        int(prediction.predict_proba().shape[1]),
    ).calculate()
    clinical_characteristics.to_csv(output_dir / "clinical_characteristics.csv", index=False)
    trajectories = ClusterTrajectoryAnalysis(
        dataset,
        target_cluster,
        preprocessing,
        int(prediction.predict_proba().shape[1]),
        config.trajectory_bin_hours,
    )
    trajectories.calculate().to_csv(output_dir / "cluster_trajectories.csv", index=False)
    trajectory_plots = trajectories.plot(output_dir / "cluster_trajectories")

    summary: dict[str, Any] = {
        "evaluation_set": evaluation_set,
        "censoring_reference_set": "train",
        "n_train": len(train),
        "n_evaluated": len(target),
        "n_clusters": int(prediction.predict_proba().shape[1]),
        "harrell_cindex": float(harrell[0]),
        "ipcw_cindex": float(ipcw[0]),
        "ipcw_tau": config.tau,
        "dynamic_auc": {
            str(time): float(value) for time, value in zip(auc_times, dynamic_auc, strict=True)
        },
        "mean_dynamic_auc": float(mean_auc),
        "integrated_brier_score": float(integrated_brier),
        "brier_score": {
            str(time): float(value)
            for time, value in zip(probability_times, brier_values, strict=True)
        },
        "calibration_weighted_absolute_error": calibration_errors,
        "calibration_bins": config.calibration_bins,
        "global_logrank": {
            "chi_square": logrank_chi2,
            "p_value": logrank_p,
            "occupied_clusters": len(occupied),
        },
        "adjusted_cox": cox_summary,
        "clinical_characteristics": {
            "variables": list(BASELINE_COVARIATE_COLUMNS),
            "hypothesis_tests": False,
        },
        "cluster_trajectories": {
            "bin_hours": config.trajectory_bin_hours,
            "value_scale": "clinical units after train-fitted winsorization",
            "aggregation": "patient-bin median, then cluster median and IQR",
        },
        "outputs": {
            "cluster_survival_summary": str(output_dir / "cluster_survival_summary.csv"),
            "brier_scores": str(output_dir / "brier_scores.csv"),
            "calibration": str(output_dir / "calibration.csv"),
            **calibration_plots,
            "cluster_survival_plot_png": str(cluster_png),
            "cluster_survival_plot_pdf": str(cluster_pdf),
            "time_metrics_plot_png": str(metrics_png),
            "time_metrics_plot_pdf": str(metrics_pdf),
            "adjusted_cox": str(output_dir / "adjusted_cox.json"),
            "clinical_characteristics": str(output_dir / "clinical_characteristics.csv"),
            "cluster_trajectories": str(output_dir / "cluster_trajectories.csv"),
            **trajectory_plots,
        },
    }
    save_json(output_dir / "survival_evaluation.json", summary)
    return summary


def run(config: MimicEvaluationConfig) -> dict[str, Any]:
    input_dir = resolve_input_path(config.input_dir)
    output_dir = resolve_output_path(config.paths.dir, Path.cwd())
    split_dirs = {name: input_dir / name for name in ("train", "validation", "test")}
    required = [
        split_dir / filename
        for split_dir in split_dirs.values()
        for filename in ("dataset.pt", "model_prediction.pt")
    ] + [input_dir / "preprocessing_parameters.csv"]
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(f"缺少冻结预测：{missing}")
    loaded = {name: _load_predictions(path) for name, path in split_dirs.items()}
    preprocessing = pd.read_csv(input_dir / "preprocessing_parameters.csv")
    cluster_counts = {
        name: int(prediction.predict_proba().shape[1])
        for name, (_frame, prediction, _dataset) in loaded.items()
    }
    if len(set(cluster_counts.values())) != 1:
        raise ValueError(f"train/validation/test 簇数不一致：{cluster_counts}")
    patient_sets = {
        name: set(frame["patient_id"]) for name, (frame, _prediction, _dataset) in loaded.items()
    }
    if (
        patient_sets["train"].intersection(patient_sets["validation"])
        or patient_sets["train"].intersection(patient_sets["test"])
        or patient_sets["validation"].intersection(patient_sets["test"])
    ):
        raise ValueError("train/validation/test patient_id 存在重叠")
    train = loaded["train"][0]
    train_cluster_risk = cast(
        pd.Series,
        train.groupby("pred_cluster", observed=True)["risk_score"].mean(),
    )
    reference_position = int(np.argmin(train_cluster_risk.to_numpy(dtype=np.float64)))
    cox_reference_cluster = int(
        train_cluster_risk.index.to_numpy(dtype=np.int64)[reference_position]
    )
    validation_summary = _evaluate_split(
        train,
        *loaded["validation"],
        preprocessing,
        config,
        output_dir / "validation",
        "model_selection_validation",
        cox_reference_cluster,
    )
    test_summary = _evaluate_split(
        train,
        *loaded["test"],
        preprocessing,
        config,
        output_dir / "test",
        "internal_held_out_test",
        cox_reference_cluster,
    )
    summary = {
        "input_dir": str(input_dir),
        "censoring_reference_set": "train",
        "cox_reference_cluster": cox_reference_cluster,
        "validation": validation_summary,
        "test": test_summary,
    }
    save_json(output_dir / "evaluation_summary.json", summary)
    return summary


@hydra.main(config_path="../../configs", config_name="mimic/evaluate", version_base="1.3")
def main(raw_config: DictConfig) -> None:
    run(MimicEvaluationConfig.model_validate(resolved_payload(raw_config)))


if __name__ == "__main__":
    main()

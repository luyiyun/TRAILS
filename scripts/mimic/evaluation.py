"""双09评价共享的输入读取、校准、调整Cox、临床描述与轨迹图表。"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter
from sksurv.nonparametric import kaplan_meier_estimator

from trails import TrailsPrediction
from trails.data import ClinicalTimeSeriesDataset

from ..utils.baseline_features import dataset_patient_ids, dataset_survival_arrays
from ..utils.baselines import BaselinePrediction
from .config import MimicEvaluationConfig
from .data import BASELINE_COVARIATE_COLUMNS
from .frozen import sha256_file
from .paths import resolve_input_path

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

RACE_GROUPS = ("WHITE", "BLACK", "ASIAN", "HISPANIC_OR_LATINO", "OTHER_OR_UNKNOWN")


def evaluation_methods(config: MimicEvaluationConfig) -> list[dict[str, Any]]:
    """读取完成manifest，并验证08确实使用了当前07的冻结输入。"""
    source = resolve_input_path(config.input_dir)
    original = json.loads((source / "run_manifest.json").read_text())
    methods: list[dict[str, Any]] = [
        {
            "name": "trails",
            "seed": original["training"]["seed"],
            "directory": source,
            "key": "trails",
            "capabilities": ["cluster", "survival"],
            "prediction_format": "trails",
        }
    ]
    for root in config.baseline_dirs:
        root = resolve_input_path(root)
        manifest = json.loads((root / "baselines_manifest.json").read_text())
        if manifest["status"] != "completed":
            raise ValueError(f"拒绝把未完整成功的08运行纳入评价：{root}")
        if manifest["risk_horizon"] != config.tau:
            raise ValueError("08和09风险时间窗不一致")
        for relative, expected in manifest["source_sha256"].items():
            if sha256_file(source / relative) != expected:
                raise ValueError(f"08来源与当前07不一致：{relative}")
        for record in manifest["methods"]:
            method = dict(record)
            if method["status"] != "completed":
                raise ValueError("08方法尚未完成")
            method["directory"] = root / method["directory"]
            for relative, expected in method["artifacts"].items():
                if sha256_file(method["directory"] / relative) != expected:
                    raise ValueError(f"基线冻结产物指纹不符：{method['name']}/{relative}")
            method["key"] = f"{method['name']}/seed-{method['seed']}"
            methods.append(method)
    keys = [method["key"] for method in methods]
    if len(set(keys)) != len(keys):
        raise ValueError("评价方法×seed重复")
    return methods


def load_prediction(
    method: dict[str, Any],
    split: str,
    dataset: ClinicalTimeSeriesDataset,
    config: MimicEvaluationConfig,
) -> BaselinePrediction:
    """按产物类型分支读取，再提取评价真正需要的标签和曲线。"""
    directory = Path(method["directory"]) / split
    if method["prediction_format"] == "trails":
        saved = TrailsPrediction.load(directory / "model_prediction.pt")
        survival = "survival" in method["capabilities"]
        prediction = BaselinePrediction(
            method_name=method["name"],
            patient_ids=dataset_patient_ids(dataset),
            cluster_labels=saved.predict().numpy().astype(np.int64),
            n_clusters=saved.predict_proba().shape[1],
            risk_score=saved.risk_score(config.tau).numpy().astype(np.float64)
            if survival
            else None,
            risk_horizon=config.tau if survival else None,
            survival_times=np.asarray(config.probability_times) if survival else None,
            survival_probabilities=saved.survival(config.probability_times)
            .numpy()
            .astype(np.float64)
            if survival
            else None,
        )
    else:
        prediction = BaselinePrediction.load(directory / "baseline_prediction.npz")
    if prediction.patient_ids != dataset_patient_ids(dataset):
        raise ValueError("预测患者顺序与冻结dataset不一致")
    if prediction.capabilities != frozenset(method["capabilities"]):
        raise ValueError("预测能力与manifest不一致")
    return prediction


def prediction_frame(
    dataset: ClinicalTimeSeriesDataset, prediction: BaselinePrediction
) -> pd.DataFrame:
    """按原始患者顺序构造评价表；患者行只保留在内存中。"""
    event, time = dataset_survival_arrays(dataset)
    frame = pd.DataFrame(
        {"patient_id": prediction.patient_ids, "event": event, "survival_time": time}
    )
    if prediction.cluster_labels is not None:
        frame["pred_cluster"] = prediction.cluster_labels
        covariates = pd.DataFrame(dataset.metadata["baseline_covariates"])
        covariates["patient_id"] = covariates["patient_id"].astype(str)
        if set(covariates["patient_id"]) != set(prediction.patient_ids):
            raise ValueError("协变量患者集合与预测不一致")
        frame = frame.merge(covariates, on="patient_id", validate="one_to_one", sort=False)
    if prediction.risk_score is not None:
        frame["risk_score"] = prediction.risk_score
    return frame


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

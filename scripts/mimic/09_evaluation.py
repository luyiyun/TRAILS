"""TRAILS与基线的统一评价入口。"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Collection
from itertools import combinations
from pathlib import Path
from typing import Any, Literal, cast

import hydra
import numpy as np
import pandas as pd
from lifelines.exceptions import ConvergenceError
from omegaconf import DictConfig
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
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

from trails import ClinicalTimeSeriesDataset
from trails.artifacts import save_json
from trails_simulate.config import resolved_payload

from ..utils.baselines import BaselinePrediction
from .config import MimicEvaluationConfig
from .evaluation import (
    AdjustedCoxAnalysis,
    ClusterClinicalCharacteristics,
    ClusterTrajectoryAnalysis,
    SurvivalCalibration,
    evaluation_methods,
    load_prediction,
    plt,
    prediction_frame,
)
from .frozen import load_frozen_datasets
from .intervention_evaluation import ClusterInterventionAnalysis
from .paths import resolve_input_path

LOGGER = logging.getLogger(__name__)

EvaluationMetricName = Literal[
    "cluster_distribution",
    "cluster_survival",
    "adjusted_cox",
    "clinical_characteristics",
    "cluster_trajectories",
    "organ_support_treatment",
    "harrell_cindex",
    "ipcw_cindex",
    "dynamic_auc",
    "brier_score",
    "integrated_brier_score",
    "survival_calibration",
    "survival_at_tau",
]
EvaluationPlotName = Literal[
    "cluster_survival",
    "adjusted_cox",
    "clinical_characteristics",
    "cluster_trajectories",
    "organ_support_treatment",
    "survival_calibration",
    "dynamic_auc",
    "brier_score",
    "calibration_error",
]
CLUSTER_METRICS = frozenset(
    {
        "cluster_distribution",
        "cluster_survival",
        "adjusted_cox",
        "clinical_characteristics",
        "cluster_trajectories",
        "organ_support_treatment",
    }
)
CLUSTER_PLOTS = frozenset(
    {
        "cluster_survival",
        "adjusted_cox",
        "clinical_characteristics",
        "cluster_trajectories",
        "organ_support_treatment",
    }
)
SURVIVAL_PLOTS = frozenset(
    {"survival_calibration", "dynamic_auc", "brier_score", "calibration_error"}
)
SURVIVAL_METRICS = frozenset(
    {
        "harrell_cindex",
        "ipcw_cindex",
        "dynamic_auc",
        "brier_score",
        "integrated_brier_score",
        "survival_calibration",
        "survival_at_tau",
    }
)
ALL_METRICS = CLUSTER_METRICS | SURVIVAL_METRICS
ALL_PLOTS = CLUSTER_PLOTS | SURVIVAL_PLOTS


def _resolve_selection(
    selection: Collection[str] | Literal["ALL"] | None,
    known: frozenset[str],
    applicable: frozenset[str],
    label: str,
) -> frozenset[str]:
    """把ALL、关闭值和显式名称归一化为当前能力适用的选择。"""
    if selection == "ALL":
        return applicable
    if selection is None:
        return frozenset()
    selected = frozenset(selection)
    if unknown := sorted(selected - known):
        raise ValueError(f"未知{label}：{unknown}")
    return selected & applicable


def evaluate_split(
    target: pd.DataFrame,
    dataset: ClinicalTimeSeriesDataset,
    n_clusters: int | None,
    preprocessing: pd.DataFrame,
    interventions: pd.DataFrame,
    config: MimicEvaluationConfig,
    output: Path,
    reference: int | None,
    *,
    train: pd.DataFrame | None = None,
    prediction: BaselinePrediction | None = None,
    evaluation_set: str | None = None,
    metrics: Collection[EvaluationMetricName] | Literal["ALL"] | None = "ALL",
    plots: Collection[EvaluationPlotName] | Literal["ALL"] | None = "ALL",
) -> dict[str, Any]:
    """按选择描述一个冻结划分；退化簇和不可估计Cox保持显式。"""
    cluster_available = (
        n_clusters is not None and reference is not None and "pred_cluster" in target
    )
    survival_available = (
        train is not None
        and prediction is not None
        and evaluation_set is not None
        and "risk_score" in target
    )
    applicable_metrics = (CLUSTER_METRICS if cluster_available else frozenset()) | (
        SURVIVAL_METRICS if survival_available else frozenset()
    )
    applicable_plots = (CLUSTER_PLOTS if cluster_available else frozenset()) | (
        SURVIVAL_PLOTS if survival_available else frozenset()
    )
    selected_metrics = _resolve_selection(metrics, ALL_METRICS, applicable_metrics, "评价指标")
    selected_plots = _resolve_selection(plots, ALL_PLOTS, applicable_plots, "评价图")
    plot_dependencies = {
        "cluster_survival": {"cluster_survival"},
        "adjusted_cox": {"adjusted_cox"},
        "clinical_characteristics": {"clinical_characteristics"},
        "cluster_trajectories": {"cluster_trajectories"},
        "organ_support_treatment": {"organ_support_treatment"},
        "survival_calibration": {"survival_calibration"},
        "dynamic_auc": {"dynamic_auc"},
        "brier_score": {"brier_score", "integrated_brier_score"},
        "calibration_error": {"survival_calibration"},
    }
    for plot in selected_plots:
        if missing := plot_dependencies[plot] - selected_metrics:
            raise ValueError(f"绘图{plot}要求同时计算指标{sorted(missing)}")
    if not selected_metrics and not selected_plots:
        return {}

    # ==================================================================================
    # 一、准备聚类评价共用的标签、结局和簇占用信息。
    # ==================================================================================
    output.mkdir(parents=True, exist_ok=True)
    event = target["event"].to_numpy(dtype=bool)
    time = target["survival_time"].to_numpy(dtype=np.float64)
    labels = (
        target["pred_cluster"].to_numpy(dtype=np.int64)
        if cluster_available
        else np.empty(0, dtype=np.int64)
    )
    cluster_count = n_clusters or 0
    counts = np.bincount(labels, minlength=cluster_count)
    occupied = np.flatnonzero(counts)
    fractions = counts / len(target) if cluster_available else np.empty(0)
    summary: dict[str, Any] = {"n_evaluated": len(target)}
    if cluster_available:
        summary["n_clusters"] = n_clusters

    # ==================================================================================
    # 二、计算评价指标。
    # ==================================================================================
    # 1. 簇分布：检查空簇、最小簇和整体分配均衡程度。
    if "cluster_distribution" in selected_metrics:
        summary.update(
            occupied_clusters=len(occupied),
            empty_clusters=int((counts == 0).sum()),
            min_cluster_fraction=float(fractions.min()),
            normalized_entropy=float(
                -(fractions[fractions > 0] * np.log(fractions[fractions > 0])).sum()
                / np.log(cluster_count)
            ),
        )

    # 2. 簇间生存差异：保存事件率，并用全局log-rank作描述性比较。
    if "cluster_survival" in selected_metrics:
        logrank: dict[str, Any] = {
            "chi_square": None,
            "p_value": None,
            "occupied_clusters": len(occupied),
        }
        if len(occupied) > 1:
            try:
                statistic, p_value = cast(
                    tuple[float, float],
                    compare_survival(Surv.from_arrays(event, time), labels),
                )
                logrank.update(chi_square=float(statistic), p_value=float(p_value))
            except np.linalg.LinAlgError:
                logrank["unavailable_reason"] = "singular log-rank covariance"
        cluster_summary = cast(
            pd.DataFrame,
            target.groupby("pred_cluster", observed=True)
            .agg(
                n_patients=("patient_id", "size"),
                event_count=("event", "sum"),
                event_rate=("event", "mean"),
                median_followup=("survival_time", "median"),
            )
            .reindex(range(cluster_count)),
        )
        cluster_summary["n_patients"] = counts
        cluster_summary["fraction"] = fractions
        cluster_summary.to_csv(output / "cluster_survival_summary.csv", index_label="pred_cluster")
        summary["global_logrank"] = logrank

    # 3. 调整Cox：固定使用训练集最低KM死亡风险簇作为参考。
    if "adjusted_cox" in selected_metrics:
        assert reference is not None
        cox_summary: dict[str, Any] = {"status": "unavailable", "reference_cluster": reference}
        if len(occupied) < 2 or reference not in occupied:
            cox_summary["reason"] = "fewer than two occupied clusters or missing train reference"
        else:
            try:
                cox = AdjustedCoxAnalysis(target, reference)
                table, cox_summary = cox.fit()
                cox_summary.update(
                    status="completed", reference_rule="lowest train KM mortality at tau"
                )
                table.to_csv(output / "adjusted_cox.csv", index=False)
                if "adjusted_cox" in selected_plots:
                    cox_summary["outputs"] = cox.plot(output / "adjusted_cox")
            except (ConvergenceError, ValueError, np.linalg.LinAlgError) as error:
                cox_summary = {
                    "status": "unavailable",
                    "reference_cluster": reference,
                    "reason": type(error).__name__,
                }
        save_json(output / "adjusted_cox.json", cox_summary)
        summary["adjusted_cox"] = cox_summary

    # 4. 临床特征：输出描述统计、总体组间检验及可选的分布/比例比较图。
    if "clinical_characteristics" in selected_metrics:
        characteristics = ClusterClinicalCharacteristics(target, cluster_count)
        characteristics.calculate().to_csv(output / "clinical_characteristics.csv", index=False)
        if "clinical_characteristics" in selected_plots:
            characteristics.plot(output / "clinical_characteristics")

    # 5. 纵向轨迹：恢复临床单位后保存各时间箱的簇中位数与IQR。
    if "cluster_trajectories" in selected_metrics:
        trajectories = ClusterTrajectoryAnalysis(
            dataset, labels, preprocessing, cluster_count, config.trajectory_bin_hours
        )
        trajectories.calculate().to_csv(output / "cluster_trajectories.csv", index=False)
        if "cluster_trajectories" in selected_plots:
            trajectories.plot(output / "cluster_trajectories")

    # 6. 器官支持与治疗：暴露率用全体患者，连续强度仅比较对应暴露者。
    if "organ_support_treatment" in selected_metrics:
        intervention_analysis = ClusterInterventionAnalysis(target, interventions, cluster_count)
        intervention_outputs = intervention_analysis.save(
            output, include_plots="organ_support_treatment" in selected_plots
        )
        summary["organ_support_treatment"] = {
            "binary_population": "all evaluated patients",
            "continuous_population": "patients with the corresponding exposure",
            "multiple_testing": "Benjamini-Hochberg across 19 endpoints within method/split",
            "outputs": intervention_outputs,
        }

    # *. 患者级生存指标：删失分布始终由训练集估计，冻结曲线不外插。
    calibration_analysis: SurvivalCalibration | None = None
    if selected_metrics & SURVIVAL_METRICS:
        assert train is not None and prediction is not None and evaluation_set is not None
        risk = target["risk_score"].to_numpy(dtype=np.float64)
        train_survival = Surv.from_arrays(
            train["event"].to_numpy(dtype=bool),
            train["survival_time"].to_numpy(dtype=np.float64),
        )
        target_survival = Surv.from_arrays(event, time)
        summary.update(
            evaluation_set=evaluation_set,
            censoring_reference_set="train",
            n_train=len(train),
            risk_definition="1-S(tau); fixed-horizon score also used for dynamic AUC",
        )

        # 7. Harrell与IPCW C-index。
        if "harrell_cindex" in selected_metrics:
            summary["harrell_cindex"] = float(concordance_index_censored(event, time, risk)[0])
        if "ipcw_cindex" in selected_metrics:
            summary["ipcw_cindex"] = float(
                concordance_index_ipcw(train_survival, target_survival, risk, tau=config.tau)[0]
            )
            summary["ipcw_tau"] = config.tau

        # 8. 累积/动态AUC。
        auc_times = np.asarray(config.auc_times, dtype=np.float64)
        if "dynamic_auc" in selected_metrics:
            if len(auc_times) == 0 or np.any(np.diff(auc_times) <= 0):
                raise ValueError("auc_times 必须为非空严格递增序列")
            if auc_times[0] <= time.min() or auc_times[-1] >= time.max():
                raise ValueError(f"auc_times 必须位于 {evaluation_set} 随访时间的开区间内")
            dynamic_auc_values, mean_auc = cumulative_dynamic_auc(
                train_survival, target_survival, risk, auc_times
            )
            summary["dynamic_auc"] = dict(
                zip(map(str, auc_times), map(float, dynamic_auc_values), strict=True)
            )
            summary["mean_dynamic_auc"] = float(mean_auc)

        # *. Brier、IBS和分位数组KM校准共用冻结生存曲线。
        probability_times = np.asarray(config.probability_times, dtype=np.float64)
        curve_metrics = {
            "brier_score",
            "integrated_brier_score",
            "survival_calibration",
        }
        if selected_metrics & curve_metrics:
            if len(probability_times) < 2 or np.any(np.diff(probability_times) <= 0):
                raise ValueError("probability_times 必须包含至少两个严格递增时间点")
            if probability_times[0] <= time.min() or probability_times[-1] >= time.max():
                raise ValueError(f"probability_times 必须位于 {evaluation_set} 随访时间的开区间内")
            assert prediction.survival_times is not None
            assert prediction.survival_probabilities is not None
            indices = np.searchsorted(prediction.survival_times, probability_times)
            if np.any(indices >= len(prediction.survival_times)) or not np.allclose(
                prediction.survival_times[indices], probability_times, rtol=0.0, atol=1e-8
            ):
                raise ValueError("冻结曲线缺少09需要的时间点，不能外插或重拟合")
            probabilities = prediction.survival_probabilities[:, indices]

            # 9. Brier
            if "brier_score" in selected_metrics:
                _, brier_values = brier_score(
                    train_survival, target_survival, probabilities, probability_times
                )
                summary["brier_score"] = dict(
                    zip(map(str, probability_times), map(float, brier_values), strict=True)
                )
                pd.DataFrame({"time": probability_times, "brier_score": brier_values}).to_csv(
                    output / "brier_scores.csv", index=False
                )

            # 10. IBS
            if "integrated_brier_score" in selected_metrics:
                integrated_brier = float(
                    integrated_brier_score(
                        train_survival, target_survival, probabilities, probability_times
                    )
                )
                summary["integrated_brier_score"] = integrated_brier

            # 11. 分位数组校准：按配置时间点分别比较分位数组预测值与KM观察值。
            if "survival_calibration" in selected_metrics:
                calibration_analysis = SurvivalCalibration(
                    event, time, probabilities, probability_times, config.calibration_bins
                )
                calibration_table, calibration_errors = calibration_analysis.calculate()
                calibration_table.to_csv(output / "calibration.csv", index=False)
                summary["calibration_weighted_absolute_error"] = calibration_errors
                summary["calibration_bins"] = config.calibration_bins

        # 12. 固定tau时预测生存率与总体KM生存率。
        if "survival_at_tau" in selected_metrics:
            km_time, km_survival = kaplan_meier_estimator(event, time)[:2]
            index = int(np.searchsorted(km_time, config.tau, side="right") - 1)
            summary["mean_predicted_survival_tau"] = float((1.0 - risk).mean())
            summary["observed_km_survival_tau"] = 1.0 if index < 0 else float(km_survival[index])

    # ==================================================================================
    # 三、绘图；绘图只消费已选择并完成计算的指标。
    # ==================================================================================
    # 1. 簇生存：仅为选中内容创建子图，避免空白面板。
    if "cluster_survival" in selected_plots:
        figure, axis = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
        for label in occupied:
            selected = labels == label
            km_t, km_s = kaplan_meier_estimator(event[selected], time[selected])[:2]
            axis.step(
                np.r_[0.0, km_t],
                np.r_[1.0, km_s],
                where="post",
                label=f"Cluster {label} (n={counts[label]})",
            )
        axis.set(
            title=f"Cluster survival (log-rank p={summary['global_logrank']['p_value']})",
            xlabel="Days after landmark",
            ylabel="Kaplan–Meier survival",
            xlim=(0, config.tau),
            ylim=(0, 1.02),
        )
        axis.legend(frameon=False)
        axis.grid(alpha=0.2)
        for suffix in ("png", "pdf"):
            figure.savefig(output / f"cluster_survival.{suffix}", dpi=220)
        plt.close(figure)

    # 2. 生存校准：按配置时间点分别比较分位数组预测值与KM观察值。
    if "survival_calibration" in selected_plots:
        assert calibration_analysis is not None
        summary.setdefault("outputs", {}).update(
            calibration_analysis.plot(output / "calibration", config.auc_times)
        )

    # 3. 时间变化指标：仅为选中内容创建子图，避免空白面板。
    time_plots = [
        name
        for name in ("dynamic_auc", "brier_score", "calibration_error")
        if name in selected_plots
    ]
    if time_plots:
        figure, raw_axes = plt.subplots(
            1,
            len(time_plots),
            figsize=(4.2 * len(time_plots), 3.8),
            squeeze=False,
            layout="constrained",
        )
        for axis, plot_name in zip(raw_axes.ravel(), time_plots, strict=True):
            if plot_name == "dynamic_auc":
                x_values = np.asarray(config.auc_times)
                y_values = [summary["dynamic_auc"][str(float(day))] for day in x_values]
                axis.plot(x_values, y_values, marker="o", color="#0072B2")
                axis.axhline(0.5, linestyle="--", color="0.5", linewidth=1.0)
                axis.set(title="Cumulative/dynamic AUC", ylabel="AUC", ylim=(0.0, 1.0))
            elif plot_name == "brier_score":
                x_values = np.asarray(config.probability_times)
                y_values = [summary["brier_score"][str(float(day))] for day in x_values]
                axis.plot(x_values, y_values, color="#D55E00")
                axis.set(
                    title=f"Brier score (IBS={summary['integrated_brier_score']:.3f})",
                    ylabel="Brier score",
                )
                axis.set_ylim(bottom=0.0)
            else:
                x_values = np.asarray(config.probability_times)
                errors = summary["calibration_weighted_absolute_error"]
                y_values = [errors[str(float(day))] for day in x_values]
                axis.plot(x_values, y_values, color="#009E73")
                axis.set(title="Grouped calibration error", ylabel="Weighted absolute error")
                axis.set_ylim(bottom=0.0)
            axis.set_xlabel("Days after landmark")
            axis.grid(alpha=0.2)
        for suffix in ("png", "pdf"):
            figure.savefig(output / f"time_metrics.{suffix}", dpi=220)
        plt.close(figure)
        summary.setdefault("outputs", {}).update(
            time_metrics_plot_png=str(output / "time_metrics.png"),
            time_metrics_plot_pdf=str(output / "time_metrics.pdf"),
        )

    if selected_metrics & CLUSTER_METRICS:
        summary["cluster_only_survival_prediction_metrics"] = False
        save_json(output / "cluster_evaluation.json", summary)
    if selected_metrics & SURVIVAL_METRICS:
        scalar_metrics = [
            summary[key]
            for key in (
                "harrell_cindex",
                "ipcw_cindex",
                "mean_dynamic_auc",
                "integrated_brier_score",
            )
            if key in summary
        ]
        if not np.isfinite(scalar_metrics).all():
            raise ValueError("生存评价产生非有限核心指标")
        save_json(output / "survival_evaluation.json", summary)
    return summary


def run(config: MimicEvaluationConfig) -> dict[str, Any]:
    """按方法实际能力统一评价聚类与患者级生存预测。"""
    primary_root = resolve_input_path(config.trails_dirs[0])
    split_root, methods = evaluation_methods(config)
    datasets = load_frozen_datasets(split_root)
    preprocessing = pd.read_csv(split_root / "preprocessing_parameters.csv")
    interventions_path = resolve_input_path(config.interventions_csv)
    if not interventions_path.is_file():
        raise FileNotFoundError(f"缺少治疗评价输入：{interventions_path}")
    interventions = pd.read_csv(interventions_path, dtype={"patient_id": str})
    output = config.paths.dir.resolve()
    if (output / "evaluation_summary.json").exists():
        raise FileExistsError(f"拒绝覆盖既有评价：{output}")
    summaries: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    assignments: dict[str, dict[str, np.ndarray]] = {"validation": {}, "test": {}}
    for method in methods:
        capabilities = frozenset(method["capabilities"])
        for capability in sorted({"cluster", "survival"} - capabilities):
            LOGGER.info(
                "Skipping %s evaluation for %s: method has no %s capability",
                capability,
                method["key"],
                capability,
            )
        LOGGER.info("Evaluating %s with capabilities=%s", method["key"], sorted(capabilities))
        train_prediction = load_prediction(method, "train", datasets["train"], config)
        train = prediction_frame(datasets["train"], train_prediction)
        reference: int | None = None
        if "cluster" in capabilities:
            train_risks: dict[int, float] = {}
            # 只选择描述性Cox参考簇，不从簇KM构造患者级生存预测。
            for label, group in train.groupby("pred_cluster", observed=True):
                km_t, km_s = kaplan_meier_estimator(
                    group["event"].to_numpy(dtype=bool),
                    group["survival_time"].to_numpy(dtype=float),
                )[:2]
                position = int(np.searchsorted(km_t, config.tau, side="right") - 1)
                train_risks[int(cast(int, label))] = (
                    0.0 if position < 0 else 1.0 - float(km_s[position])
                )
            reference = min(train_risks, key=lambda label: (train_risks[label], label))
        per_split = {}
        for split in ("validation", "test"):
            prediction = load_prediction(method, split, datasets[split], config)
            if "cluster" in capabilities and prediction.n_clusters != train_prediction.n_clusters:
                raise ValueError("各划分预测簇数不一致")
            frame = prediction_frame(datasets[split], prediction)
            summary = evaluate_split(
                frame,
                datasets[split],
                prediction.n_clusters,
                preprocessing,
                interventions,
                config,
                output / method["key"] / split,
                reference,
                train=train if "survival" in capabilities else None,
                prediction=prediction if "survival" in capabilities else None,
                evaluation_set=split if "survival" in capabilities else None,
            )
            per_split[split] = summary
            if prediction.cluster_labels is not None:
                assignments[split][method["key"]] = prediction.cluster_labels
            row = {
                "method": method["name"],
                "source_run": method["source_run"],
                "seed": method["seed"],
                "split": split,
            }
            for key in (
                "occupied_clusters",
                "empty_clusters",
                "min_cluster_fraction",
                "normalized_entropy",
                "harrell_cindex",
                "ipcw_cindex",
                "mean_dynamic_auc",
                "integrated_brier_score",
                "mean_predicted_survival_tau",
                "observed_km_survival_tau",
            ):
                row[key] = summary.get(key)
            row["logrank_p"] = summary.get("global_logrank", {}).get("p_value")
            row["adjusted_cox_status"] = summary.get("adjusted_cox", {}).get("status")
            calibration_errors = summary.get("calibration_weighted_absolute_error", {})
            for day in config.auc_times:
                row[f"calibration_error_{day}"] = calibration_errors.get(str(float(day)))
            rows.append(row)
        summaries[method["key"]] = per_split
    pairs = [
        {
            "split": split,
            "left": left,
            "right": right,
            "ari": adjusted_rand_score(labels[left], labels[right]),
            "nmi": normalized_mutual_info_score(labels[left], labels[right]),
        }
        for split, labels in assignments.items()
        for left, right in combinations(labels, 2)
    ]
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    pd.DataFrame(pairs, columns=["split", "left", "right", "ari", "nmi"]).to_csv(
        output / "label_agreement.csv", index=False
    )
    summary = {
        "primary_trails_dir": str(primary_root),
        "split_dir": str(split_root),
        "interventions_csv": str(interventions_path),
        "interventions_sha256": hashlib.sha256(interventions_path.read_bytes()).hexdigest(),
        "trails_dirs": [str(resolve_input_path(path)) for path in config.trails_dirs],
        "baseline_dirs": [str(path) for path in config.baseline_dirs],
        "censoring_reference_set": "train",
        "methods": summaries,
        "label_agreement_is_not_ground_truth_accuracy": True,
    }
    save_json(output / "evaluation_summary.json", summary)
    return summary


@hydra.main(config_path="../../configs", config_name="mimic/evaluate", version_base="1.3")
def main(raw_config: DictConfig) -> None:
    run(MimicEvaluationConfig.model_validate(resolved_payload(raw_config)))


if __name__ == "__main__":
    main()

"""统一评价TRAILS和具备患者级生存能力的基线，不重新拟合任何模型。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
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
from trails_simulate.config import resolved_payload

from ..utils.baselines import BaselinePrediction
from .config import MimicEvaluationConfig
from .evaluation import SurvivalCalibration, plt
from .evaluation_inputs import evaluation_methods, load_prediction, prediction_frame
from .frozen import load_frozen_datasets
from .paths import resolve_input_path

LOGGER = logging.getLogger(__name__)


def evaluate_split(
    train: pd.DataFrame,
    target: pd.DataFrame,
    prediction: BaselinePrediction,
    config: MimicEvaluationConfig,
    output_dir: Path,
    evaluation_set: str,
) -> dict[str, Any]:
    """沿用原08指标定义，删失分布始终由train估计。"""
    train_event = train["event"].to_numpy(dtype=bool)
    train_time = train["survival_time"].to_numpy(dtype=np.float64)
    target_event = target["event"].to_numpy(dtype=bool)
    target_followup = target["survival_time"].to_numpy(dtype=np.float64)
    risk = target["risk_score"].to_numpy(dtype=np.float64)
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
    assert prediction.survival_times is not None and prediction.survival_probabilities is not None
    indices = np.searchsorted(prediction.survival_times, probability_times)
    if np.any(indices >= len(prediction.survival_times)) or not np.allclose(
        prediction.survival_times[indices], probability_times, rtol=0.0, atol=1e-8
    ):
        raise ValueError("冻结曲线缺少09需要的时间点，不能外插或重拟合")
    survival_probabilities = prediction.survival_probabilities[:, indices]
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

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"time": probability_times, "brier_score": brier_values}).to_csv(
        output_dir / "brier_scores.csv", index=False
    )
    calibration_table.to_csv(output_dir / "calibration.csv", index=False)
    calibration_plots = calibration.plot(output_dir / "calibration", config.auc_times)

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

    km_time, km_survival = kaplan_meier_estimator(target_event, target_followup)[:2]
    index = int(np.searchsorted(km_time, config.tau, side="right") - 1)
    summary: dict[str, Any] = {
        "evaluation_set": evaluation_set,
        "censoring_reference_set": "train",
        "n_train": len(train),
        "n_evaluated": len(target),
        "harrell_cindex": float(harrell[0]),
        "ipcw_cindex": float(ipcw[0]),
        "ipcw_tau": config.tau,
        "dynamic_auc": {str(t): float(v) for t, v in zip(auc_times, dynamic_auc, strict=True)},
        "mean_dynamic_auc": float(mean_auc),
        "integrated_brier_score": float(integrated_brier),
        "brier_score": {
            str(t): float(v) for t, v in zip(probability_times, brier_values, strict=True)
        },
        "calibration_weighted_absolute_error": calibration_errors,
        "calibration_bins": config.calibration_bins,
        "mean_predicted_survival_tau": float((1.0 - risk).mean()),
        "observed_km_survival_tau": 1.0 if index < 0 else float(km_survival[index]),
        "risk_definition": "1-S(tau); fixed-horizon score also used for dynamic AUC",
        "outputs": {
            "brier_scores": str(output_dir / "brier_scores.csv"),
            "calibration": str(output_dir / "calibration.csv"),
            **calibration_plots,
            "time_metrics_plot_png": str(metrics_png),
            "time_metrics_plot_pdf": str(metrics_pdf),
        },
    }
    values = [
        summary[key]
        for key in ("harrell_cindex", "ipcw_cindex", "mean_dynamic_auc", "integrated_brier_score")
    ]
    if not np.isfinite(values).all():
        raise ValueError("生存评价产生非有限核心指标")
    save_json(output_dir / "survival_evaluation.json", summary)
    return summary


def run(config: MimicEvaluationConfig) -> dict[str, Any]:
    """对每个已冻结方法逐划分评价，保留逐方法结果和统一比较表。"""
    source = resolve_input_path(config.input_dir)
    datasets = load_frozen_datasets(source)
    output = config.paths.dir.resolve() / "survival"
    if (output / "evaluation_summary.json").exists():
        raise FileExistsError(f"拒绝覆盖既有评价：{output}")
    summaries: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for method in evaluation_methods(config):
        if "survival" not in method["capabilities"]:
            continue
        LOGGER.info("Evaluating survival: %s", method["key"])
        train_prediction = load_prediction(method, "train", datasets["train"], config)
        train = prediction_frame(datasets["train"], train_prediction)
        per_split = {}
        for split in ("validation", "test"):
            prediction = load_prediction(method, split, datasets[split], config)
            target = prediction_frame(datasets[split], prediction)
            summary = evaluate_split(
                train, target, prediction, config, output / method["key"] / split, split
            )
            per_split[split] = summary
            row = {"method": method["name"], "seed": method["seed"], "split": split}
            row.update(
                {
                    key: summary[key]
                    for key in (
                        "harrell_cindex",
                        "ipcw_cindex",
                        "mean_dynamic_auc",
                        "integrated_brier_score",
                        "mean_predicted_survival_tau",
                        "observed_km_survival_tau",
                    )
                }
            )
            row.update(
                {
                    f"calibration_error_{day}": summary["calibration_weighted_absolute_error"][
                        str(float(day))
                    ]
                    for day in config.auc_times
                }
            )
            rows.append(row)
        summaries[method["key"]] = per_split
    summary = {
        "input_dir": str(source),
        "baseline_dirs": [str(p) for p in config.baseline_dirs],
        "censoring_reference_set": "train",
        "methods": summaries,
    }
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    save_json(output / "evaluation_summary.json", summary)
    return summary


@hydra.main(config_path="../../configs", config_name="mimic/evaluate", version_base="1.3")
def main(raw_config: DictConfig) -> None:
    run(MimicEvaluationConfig.model_validate(resolved_payload(raw_config)))


if __name__ == "__main__":
    main()

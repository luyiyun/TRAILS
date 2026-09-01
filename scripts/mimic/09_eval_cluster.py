"""统一评价聚类分层、临床描述和轨迹，不构造簇级预测生存模型。"""

from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import pandas as pd
from lifelines.exceptions import ConvergenceError
from omegaconf import DictConfig
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sksurv.compare import compare_survival
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv

from trails import ClinicalTimeSeriesDataset
from trails.artifacts import save_json
from trails_simulate.config import resolved_payload

from .config import MimicEvaluationConfig
from .evaluation import (
    AdjustedCoxAnalysis,
    ClusterClinicalCharacteristics,
    ClusterTrajectoryAnalysis,
    plt,
)
from .evaluation_inputs import evaluation_methods, load_prediction, prediction_frame
from .frozen import load_frozen_datasets
from .paths import resolve_input_path

LOGGER = logging.getLogger(__name__)


def evaluate_split(
    target: pd.DataFrame,
    dataset: ClinicalTimeSeriesDataset,
    n_clusters: int,
    preprocessing: pd.DataFrame,
    config: MimicEvaluationConfig,
    output: Path,
    reference: int,
) -> dict[str, Any]:
    """描述一个冻结划分；退化簇和不可估计Cox显式报告，不改簇标签。"""
    output.mkdir(parents=True, exist_ok=True)
    labels = target["pred_cluster"].to_numpy(dtype=np.int64)
    event = target["event"].to_numpy(dtype=bool)
    time = target["survival_time"].to_numpy(dtype=np.float64)
    counts = np.bincount(labels, minlength=n_clusters)
    occupied = np.flatnonzero(counts)
    fractions = counts / len(target)
    logrank: dict[str, Any] = {
        "chi_square": None,
        "p_value": None,
        "occupied_clusters": len(occupied),
    }
    if len(occupied) > 1:
        try:
            statistic, p_value = cast(
                tuple[float, float], compare_survival(Surv.from_arrays(event, time), labels)
            )
            logrank.update(chi_square=float(statistic), p_value=float(p_value))
        except np.linalg.LinAlgError:
            logrank["unavailable_reason"] = "singular log-rank covariance"
    cluster_summary = cast(
        pd.DataFrame,
        (
            target.groupby("pred_cluster", observed=True)
            .agg(
                n_patients=("patient_id", "size"),
                event_count=("event", "sum"),
                event_rate=("event", "mean"),
                median_followup=("survival_time", "median"),
            )
            .reindex(range(n_clusters))
        ),
    )
    cluster_summary["n_patients"] = counts
    cluster_summary["fraction"] = fractions
    cluster_summary.to_csv(output / "cluster_survival_summary.csv", index_label="pred_cluster")
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
        title=f"Cluster survival (log-rank p={logrank['p_value']})",
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
            cox_summary["outputs"] = cox.plot(output / "adjusted_cox")
        except (ConvergenceError, ValueError, np.linalg.LinAlgError) as error:
            cox_summary = {
                "status": "unavailable",
                "reference_cluster": reference,
                "reason": type(error).__name__,
            }
    save_json(output / "adjusted_cox.json", cox_summary)
    ClusterClinicalCharacteristics(target, n_clusters).calculate().to_csv(
        output / "clinical_characteristics.csv", index=False
    )
    trajectories = ClusterTrajectoryAnalysis(
        dataset, labels, preprocessing, n_clusters, config.trajectory_bin_hours
    )
    trajectories.calculate().to_csv(output / "cluster_trajectories.csv", index=False)
    trajectories.plot(output / "cluster_trajectories")
    summary = {
        "n_evaluated": len(target),
        "n_clusters": n_clusters,
        "occupied_clusters": len(occupied),
        "empty_clusters": int((counts == 0).sum()),
        "min_cluster_fraction": float(fractions.min()),
        "normalized_entropy": float(
            -(fractions[fractions > 0] * np.log(fractions[fractions > 0])).sum()
            / np.log(n_clusters)
        ),
        "global_logrank": logrank,
        "adjusted_cox": cox_summary,
        "cluster_only_survival_prediction_metrics": False,
    }
    save_json(output / "cluster_evaluation.json", summary)
    return summary


def run(config: MimicEvaluationConfig) -> dict[str, Any]:
    """对所有聚类方法评价，并保存方法间/seed间标签一致性。"""
    source = resolve_input_path(config.input_dir)
    datasets = load_frozen_datasets(source)
    preprocessing = pd.read_csv(source / "preprocessing_parameters.csv")
    output = config.paths.dir.resolve() / "cluster"
    if (output / "evaluation_summary.json").exists():
        raise FileExistsError(f"拒绝覆盖既有评价：{output}")
    summaries: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    assignments: dict[str, dict[str, np.ndarray]] = {"validation": {}, "test": {}}
    for method in evaluation_methods(config):
        if "cluster" not in method["capabilities"]:
            continue
        LOGGER.info("Evaluating cluster: %s", method["key"])
        train_prediction = load_prediction(method, "train", datasets["train"], config)
        train = prediction_frame(datasets["train"], train_prediction)
        train_risks: dict[int, float] = {}
        # 只用于选择描述性Cox参考簇；不生成患者预测，也不计算cluster-only预测指标。
        for label, group in train.groupby("pred_cluster", observed=True):
            km_t, km_s = kaplan_meier_estimator(
                group["event"].to_numpy(dtype=bool), group["survival_time"].to_numpy(dtype=float)
            )[:2]
            position = int(np.searchsorted(km_t, config.tau, side="right") - 1)
            train_risks[int(cast(int, label))] = (
                0.0 if position < 0 else 1.0 - float(km_s[position])
            )
        reference = min(train_risks, key=lambda label: (train_risks[label], label))
        per_split = {}
        for split in ("validation", "test"):
            prediction = load_prediction(method, split, datasets[split], config)
            assert prediction.cluster_labels is not None and prediction.n_clusters is not None
            if prediction.n_clusters != train_prediction.n_clusters:
                raise ValueError("各划分预测簇数不一致")
            frame = prediction_frame(datasets[split], prediction)
            summary = evaluate_split(
                frame,
                datasets[split],
                prediction.n_clusters,
                preprocessing,
                config,
                output / method["key"] / split,
                reference,
            )
            per_split[split] = summary
            assignments[split][method["key"]] = prediction.cluster_labels
            rows.append(
                {
                    "method": method["name"],
                    "seed": method["seed"],
                    "split": split,
                    **{
                        key: summary[key]
                        for key in (
                            "occupied_clusters",
                            "empty_clusters",
                            "min_cluster_fraction",
                            "normalized_entropy",
                        )
                    },
                    "logrank_p": summary["global_logrank"]["p_value"],
                    "adjusted_cox_status": summary["adjusted_cox"]["status"],
                }
            )
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
        "input_dir": str(source),
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

"""在06冻结划分上训练基线，保存模型与三划分预测，不在此选择测试结果。"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

from trails import ClinicalTimeSeriesDataset
from trails.artifacts import plot_history, save_history_csv, save_json
from trails_simulate.config import resolved_payload

from ..utils.baseline_features import dataset_patient_ids
from ..utils.baseline_trails import TrailsNoSurvivalBaseline
from ..utils.baselines import BaselineMethod, KSelectableBaselineMethod
from .baseline_registry import BASELINE_REGISTRY
from .config import (
    MimicBaselineMethodConfig,
    MimicBaselinesConfig,
    MimicClusterMethodBaseConfig,
    MPJLCMMMethodConfig,
)
from .frozen import load_frozen_datasets, sha256_file
from .paths import resolve_input_path

LOGGER = logging.getLogger(__name__)


def select_k_and_fit(
    method: MimicBaselineMethodConfig,
    candidate_clusters: tuple[int, ...],
    train: ClinicalTimeSeriesDataset,
    validation: ClinicalTimeSeriesDataset,
    *,
    prediction_times: NDArray[np.float64],
    risk_horizon: float,
    work_dir: Path,
    min_cluster_fraction: float = 0.05,
    silh_tie_tolerance: float = 0.01,
    bic_tie_tolerance: float = 2.0,
    ibs_tie_tolerance: float = 0.005,
    joint_tie_tolerance: float = 0.01,
) -> tuple[int, dict[int, BaselineMethod], dict[str, Any]]:
    """拟合全部候选K，并按注册规则复用最优K对应的各seed模型。"""
    registration = BASELINE_REGISTRY[method.kind]
    rule = registration.k_selection_rule
    if rule not in {"silhouette", "bic", "ibs", "bic_cindex"}:
        raise ValueError(f"{method.name}尚未实现K选择指标")
    metric_names = {
        "silhouette": ("silhouette",),
        "bic": ("bic",),
        "ibs": ("ibs",),
        "bic_cindex": ("bic", "cindex"),
    }[rule]
    rule_tolerance = {
        "silhouette": silh_tie_tolerance,
        "bic": bic_tie_tolerance,
        "ibs": ibs_tie_tolerance,
        "bic_cindex": joint_tie_tolerance,
    }[rule]

    # ==================================================================================
    # 一、逐候选K与seed拟合；baseline只提供原始指标和聚类预测。
    # ==================================================================================
    candidate_rows: list[dict[str, Any]] = []
    eligible_models: dict[int, dict[int, BaselineMethod]] = {}
    eligible_scores: dict[int, float] = {}
    for n_clusters in sorted(candidate_clusters):
        seed_rows: list[dict[str, Any]] = []
        fitted: dict[int, BaselineMethod] = {}
        for seed in method.seeds:
            candidate_dir = work_dir / "k_selection" / f"k-{n_clusters}" / f"seed-{seed}"
            candidate_dir.mkdir(parents=True, exist_ok=False)
            baseline = registration.factory(method, n_clusters, seed, candidate_dir)
            baseline.fit(train, validation)
            selectable = cast(KSelectableBaselineMethod, baseline)
            metrics = dict(
                selectable.k_selection_metrics(
                    train,
                    validation,
                    prediction_times=prediction_times,
                    risk_horizon=risk_horizon,
                )
            )
            if set(metrics) != set(metric_names) or not np.isfinite(tuple(metrics.values())).all():
                raise ValueError(f"{rule} K选择必须返回有限指标：{metric_names}")
            fractions: dict[str, float] = {}
            for split_name, data in (("train", train), ("validation", validation)):
                prediction = baseline.predict(
                    data,
                    prediction_times=prediction_times,
                    risk_horizon=risk_horizon,
                )
                if prediction.cluster_labels is None or prediction.n_clusters != n_clusters:
                    raise ValueError("K选择候选未返回与配置K一致的聚类标签")
                counts = np.bincount(prediction.cluster_labels, minlength=n_clusters)
                fractions[split_name] = float(counts.min() / len(data))
            row: dict[str, Any] = {
                "seed": seed,
                "status": "accepted"
                if min(fractions.values()) >= min_cluster_fraction
                else "rejected",
                "metrics": metrics,
                "min_cluster_fraction": fractions,
            }
            if row["status"] == "accepted":
                fitted[seed] = baseline
            seed_rows.append(row)

        candidate: dict[str, Any] = {"n_clusters": n_clusters, "seeds": seed_rows}
        if len(fitted) == len(method.seeds):
            candidate["status"] = "eligible"
            eligible_models[n_clusters] = fitted
        else:
            candidate["status"] = "ineligible"
        candidate_rows.append(candidate)

    # ==================================================================================
    # 二、先在每个seed内跨K计算联合指标，再按K对全部seed指标求均值。
    # ==================================================================================
    if rule == "bic_cindex":
        for seed_index in range(len(method.seeds)):
            rows = [candidate["seeds"][seed_index] for candidate in candidate_rows]
            bics = np.asarray([row["metrics"]["bic"] for row in rows], dtype=np.float64)
            bic_range = float(bics.max() - bics.min())
            if bic_range == 0.0:
                normalized = bics.copy()
            else:
                normalized = (bics - bics.min()) / bic_range
            for row, bic_normalized in zip(rows, normalized, strict=True):
                row["metrics"].update(
                    {
                        "bic_normalized": float(bic_normalized),
                        "bic_cindex": float(
                            np.hypot(row["metrics"]["cindex"], 1.0 - bic_normalized)
                        ),
                    }
                )
        metric_names = (*metric_names, "bic_normalized", "bic_cindex")

    selection_metric = "bic_cindex" if rule == "bic_cindex" else metric_names[0]
    for candidate in candidate_rows:
        if candidate["status"] != "eligible":
            continue
        seed_rows = candidate["seeds"]
        mean_metrics = {
            name: float(np.mean([row["metrics"][name] for row in seed_rows]))
            for name in metric_names
        }
        candidate["mean_metrics"] = mean_metrics
        eligible_scores[candidate["n_clusters"]] = mean_metrics[selection_metric]

    # ==================================================================================
    # 三、应用门槛；近似并列时固定选择较小K。
    # ==================================================================================
    selected_k = None
    if eligible_scores:
        best_score = (
            max(eligible_scores.values())
            if rule in {"silhouette", "bic_cindex"}
            else min(eligible_scores.values())
        )
        selected_k = min(
            n_clusters
            for n_clusters, score in eligible_scores.items()
            if abs(best_score - score) <= rule_tolerance
        )
    result = {
        "rule": rule,
        "candidate_clusters": sorted(candidate_clusters),
        "seeds": list(method.seeds),
        "min_cluster_fraction": min_cluster_fraction,
        "tie_tolerance": rule_tolerance,
        "selected_k": selected_k,
        "candidates": candidate_rows,
    }
    save_json(work_dir / "k_selection" / "selection.json", result)
    if selected_k is None:
        raise RuntimeError(f"{method.name}没有候选K通过选择门槛")
    return selected_k, eligible_models[selected_k], result


def run(config: MimicBaselinesConfig) -> dict[str, Any]:
    """逐method×seed拟合；任何错误保留原始traceback并立即停止。"""
    # ==================================================================================
    # 一、加载冻结划分并建立原子更新的运行manifest。
    # ==================================================================================
    source = resolve_input_path(config.split_dir)
    output = config.paths.dir.resolve()
    manifest_path = output / "baselines_manifest.json"
    pending = manifest_path.with_suffix(".tmp")
    if manifest_path.exists():
        raise FileExistsError(f"拒绝覆盖既有基线运行：{output}")
    split_manifest = json.loads((source / "split_manifest.json").read_text())
    datasets = load_frozen_datasets(source)
    output.mkdir(parents=True, exist_ok=True)
    split_sizes = {name: len(data) for name, data in datasets.items()}
    if split_manifest["split_counts"] != split_sizes:
        raise ValueError("06 split manifest的划分大小与dataset不一致")
    records: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "format_version": 1,
        "status": "running",
        "data": {
            "split_dir": str(source),
            "split_seed": split_manifest["split_seed"],
            "split_sizes": split_sizes,
            "n_features": datasets["train"].n_features,
        },
        "risk_horizon": config.trainer.risk_horizon,
        "prediction_times": config.prediction_times,
        "config": config.model_dump(mode="json"),
        "methods": records,
    }
    total = sum(len(method.seeds) for method in config.methods)
    started = time.perf_counter()
    prediction_times = np.asarray(config.prediction_times, dtype=np.float64)
    # ==================================================================================
    # 二、逐方法解析K，必要时完成选择，再保存各seed模型与三划分预测。
    # ==================================================================================
    for method in config.methods:
        registration = BASELINE_REGISTRY[method.kind]

        # 1. 先明确该方法属于无K、固定K还是自动选K；只有最后一种调用selector。
        if isinstance(method, (MimicClusterMethodBaseConfig, MPJLCMMMethodConfig)):
            requested_clusters = method.requested_clusters
            requested_k: int | list[int] | None = (
                method.n_clusters if isinstance(method.n_clusters, int) else list(method.n_clusters)
            )
            k_selection_mode = "automatic" if len(requested_clusters) > 1 else "fixed"
        else:
            # CoxPH、RSF和JMbayes2没有聚类数参数，effective_k保持None并直接拟合。
            requested_clusters = ()
            requested_k = None
            k_selection_mode = "not_applicable"
        effective_k = requested_clusters[0] if requested_clusters else None
        capabilities = sorted(registration.capabilities)
        fitted_by_seed: dict[int, BaselineMethod] = {}
        k_selection: dict[str, Any] | None = None
        if k_selection_mode == "automatic":
            effective_k, fitted_by_seed, k_selection = select_k_and_fit(
                method,
                requested_clusters,
                datasets["train"],
                datasets["validation"],
                prediction_times=prediction_times,
                risk_horizon=config.trainer.risk_horizon,
                work_dir=output / method.name,
            )

        # 2. 无K/固定K在这里首次拟合；自动选K则复用selector返回的已拟合模型。
        # 选择阶段不接触test指标，三种路径均只在模型锁定后生成三划分预测。
        for seed in method.seeds:
            relative_dir = Path(method.name) / f"seed-{seed}"
            method_dir = output / relative_dir
            method_dir.mkdir(parents=True, exist_ok=False)
            record: dict[str, Any] = {
                "name": method.name,
                "kind": method.kind,
                "seed": seed,
                "directory": relative_dir.as_posix(),
                "status": "running",
                "prediction_format": registration.prediction_format,
                "capabilities": capabilities,
                "k_selection_mode": k_selection_mode,
                "config": method.model_dump(mode="json"),
            }
            if effective_k is not None:
                record.update(
                    {
                        "requested_n_clusters": requested_k,
                        "effective_n_clusters": effective_k,
                    }
                )
                if k_selection is not None:
                    record["k_selection"] = k_selection
            records.append(record)
            save_json(pending, manifest)
            pending.replace(manifest_path)
            LOGGER.info(
                "Baseline %s/%s: %s seed=%s fit started", len(records), total, method.name, seed
            )
            method_started = time.perf_counter()
            # 如果不去自动选择K，不运行select_k_and_fit()，则fitted_by_seed为空。
            baseline = fitted_by_seed.pop(seed, None)
            if baseline is None:
                baseline = registration.factory(method, effective_k, seed, method_dir)
                baseline.fit(datasets["train"], datasets["validation"])
            # 注册已统一构造、拟合和K选择；此分支只保留TRAILS专属的完整预测与训练轨迹。
            if registration.prediction_format == "trails":
                trails_baseline = cast(TrailsNoSurvivalBaseline, baseline)
                if trails_baseline.estimator is None:
                    raise RuntimeError("TRAILS-no-survival拟合后缺少estimator")
                estimator = trails_baseline.estimator
                record["training"] = estimator.config.model_dump(mode="json")
                trails_baseline.save_model(method_dir / f"model.{registration.model_suffix}")
                save_history_csv(method_dir / "training_history.csv", estimator.history)
                save_json(method_dir / "training_history.json", estimator.history)
                plot_history(method_dir / "training_history.png", estimator.history)
                for split, data in datasets.items():
                    trails_baseline.predict_trails(data).save(
                        method_dir / split / "model_prediction.pt"
                    )
            else:
                baseline.save_model(method_dir / f"model.{registration.model_suffix}")
                for split, data in datasets.items():
                    LOGGER.info("%s seed=%s predict split=%s", method.name, seed, split)
                    prediction = baseline.predict(
                        data,
                        prediction_times=prediction_times,
                        risk_horizon=config.trainer.risk_horizon,
                    )
                    if prediction.patient_ids != dataset_patient_ids(data):
                        raise ValueError("基线预测患者顺序与冻结数据不一致")
                    if prediction.capabilities != baseline.capabilities:
                        raise ValueError("基线预测能力与注册声明不一致")
                    prediction.save(method_dir / split / "baseline_prediction.npz")
            del baseline
            record["status"] = "completed"
            record["artifacts"] = {
                path.relative_to(method_dir).as_posix(): sha256_file(path)
                for path in method_dir.rglob("*")
                if path.is_file()
                and path.parent.name != "r"
                and "r" not in path.relative_to(method_dir).parts
            }
            record["elapsed_seconds"] = time.perf_counter() - method_started
            save_json(method_dir / "method_manifest.json", record)
            manifest["elapsed_seconds"] = time.perf_counter() - started
            save_json(pending, manifest)
            pending.replace(manifest_path)
            LOGGER.info(
                "Baseline %s/%s: %s status=%s elapsed=%.1fs total=%.1fs",
                len(records),
                total,
                method.name,
                record["status"],
                record["elapsed_seconds"],
                manifest["elapsed_seconds"],
            )
    # ==================================================================================
    # 三、只有全部方法完成时才把整批manifest标记为成功。
    # ==================================================================================
    manifest["status"] = (
        "completed" if all(r["status"] == "completed" for r in records) else "failed"
    )
    save_json(pending, manifest)
    pending.replace(manifest_path)
    if manifest["status"] != "completed":
        raise RuntimeError("部分基线失败；已保存成功方法和失败manifest，不得作为完整比较")
    return manifest


@hydra.main(config_path="../../configs", config_name="mimic/baselines", version_base="1.3")
def main(raw_config: DictConfig) -> None:
    run(MimicBaselinesConfig.model_validate(resolved_payload(raw_config)))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from scripts.crc_yunnan.config import CRCRunConfig
from threadpoolctl import threadpool_limits

from trails import (
    ClinicalTimeSeriesDataset,
    ClusterNumberSelector,
    DataConfig,
    TrailsConfig,
    TrailsEstimator,
    resolve_batch_size,
)
from trails.artifacts import save_history_csv, save_json
from trails.survival import SURVIVAL_FORMAT_VERSION
from trails_simulate.config import resolved_payload

LOGGER = logging.getLogger(__name__)


def _save_split_outputs(
    run_dir: Path, name: str, dataset: ClinicalTimeSeriesDataset, estimator: TrailsEstimator
) -> dict[str, Any]:
    """保存既有患者预测，复用公开方法报告生存、潜空间与占簇指标。"""
    prediction = estimator.predict(dataset)
    trainer = estimator.config.trainer
    labels = prediction.predict()
    risk = prediction.risk_score(trainer.risk_horizon, method=trainer.cindex_risk_score)
    times = torch.stack([sample.survival_time for sample in dataset.samples])
    events = torch.stack([sample.event for sample in dataset.samples])
    split_dir = run_dir / name
    prediction.save(split_dir / "model_prediction.pt")
    frame = pd.DataFrame(
        {
            "patient_id": dataset.metadata["patient_ids"],
            "pred_cluster": labels.numpy(),
            "predicted_median_time_days": prediction.median_survival_time().numpy(),
            "risk_score": risk.numpy(),
            "survival_time": times.numpy(),
            "event": events.numpy(),
        }
    )
    probabilities = pd.DataFrame(
        prediction.predict_proba().numpy(),
        columns=[
            f"cluster_probability_{index}" for index in range(estimator.config.model.n_clusters)
        ],
    )
    pd.concat([frame, probabilities], axis=1).to_csv(split_dir / "patient_outputs.csv", index=False)
    metrics = ClusterNumberSelector.evaluate_fitted(estimator, dataset)
    if not bool(events.any()):
        metrics["cindex"] = float("nan")
    fractions = torch.bincount(labels, minlength=estimator.config.model.n_clusters) / len(dataset)
    return {
        "n_patients": len(dataset),
        "n_events": int(events.sum()),
        "metric_reason": "无事件，C-index不可计算" if not bool(events.any()) else "",
        **metrics,
        **{f"cluster_{k}_fraction": float(value) for k, value in enumerate(fractions)},
    }


def run(config: CRCRunConfig) -> dict[str, Any]:
    ##################################################
    # 一、沿用冻结04产物，只读取训练及可选验证集。
    ##################################################
    split_dir, root = config.split.dir.resolve(), config.paths.dir.resolve()
    manifest = json.loads((split_dir / "preproc_manifest.json").read_text())
    names = manifest["datasets"]
    if root == split_dir or split_dir in root.parents or root in split_dir.parents:
        raise ValueError("建模输出不得写入冻结04目录")
    reserved = {"model.pt", "training_history.csv", "metrics.csv", "resolved_config.yaml"}
    if reserved.intersection(names):
        raise ValueError("数据集名称不能与候选模型产物冲突")
    if config.swanlab.enabled:
        raise ValueError("CRC建模请设置swanlab.enabled=false")
    targets = [
        root / name
        for name in (
            "models",
            "run_manifest.json",
            "run_metrics.csv",
            "k_summary.csv",
            "resolved_config.yaml",
        )
    ]
    if existing := [str(path) for path in targets if path.exists()]:
        raise FileExistsError(f"拒绝覆盖已有建模产物：{existing}")
    datasets = {
        name: ClinicalTimeSeriesDataset.load(split_dir / name / "dataset.pt")
        for name in ("train", "validation")
        if name in names
    }
    train, validation = datasets["train"], datasets.get("validation")
    clusters = config.n_clusters if isinstance(config.n_clusters, list) else [config.n_clusters]
    root.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config.model_dump(mode="json"), root / "resolved_config.yaml")

    ##################################################
    # 二、串行训练全部seed/K；不排序、不选择、不使用test早停。
    ##################################################
    models: list[dict[str, Any]] = []
    for seed in config.seeds:
        for k in clusters:
            LOGGER.info(
                "候选 %d/%d：seed=%d，K=%d",
                len(models) + 1,
                len(config.seeds) * len(clusters),
                seed,
                k,
            )
            model_config = TrailsConfig(
                data=DataConfig(n_features=train.n_features),
                model=config.model.model_copy(update={"n_clusters": k}),
                trainer=config.trainer.model_copy(
                    update={
                        "seed": seed,
                        "valid_size": 0.0,
                        "batch_size": resolve_batch_size(len(train), config.trainer.batch_size),
                    }
                ),
                seed=seed,
            )
            estimator = TrailsEstimator(model_config).fit(train, validation_data=validation)
            directory = Path("models") / f"seed-{seed}" / f"k{k}"
            estimator.save(root / directory / "model.pt")
            save_history_csv(root / directory / "training_history.csv", estimator.history)
            OmegaConf.save(
                model_config.model_dump(mode="json"), root / directory / "resolved_config.yaml"
            )
            models.append(
                {
                    "n_clusters": k,
                    "seed": seed,
                    "model": str(directory / "model.pt"),
                    "prediction_dir": str(directory),
                    "training_history": str(directory / "training_history.csv"),
                    "metrics": str(directory / "metrics.csv"),
                }
            )
            del estimator

    ##################################################
    # 三、所有模型冻结后读取其余集，报告各集指标和占簇诊断。
    ##################################################
    for name in names:
        if name not in datasets:
            datasets[name] = ClinicalTimeSeriesDataset.load(split_dir / name / "dataset.pt")
    runs: list[pd.DataFrame] = []
    for record in models:
        estimator = TrailsEstimator.load(root / record["model"], device=config.trainer.device)
        rows = []
        for name in names:
            metrics = _save_split_outputs(
                root / record["prediction_dir"], name, datasets[name], estimator
            )
            reasons = []
            if metrics["cluster_empty_count"]:
                reasons.append("存在空簇")
            if metrics["cluster_min_fraction"] < config.min_cluster_fraction:
                reasons.append(f"最小簇占比低于{config.min_cluster_fraction:.0%}")
            rows.append(
                {
                    "seed": record["seed"],
                    "n_clusters": record["n_clusters"],
                    "dataset": name,
                    **metrics,
                    "passes_occupancy": not reasons,
                    "occupancy_reason": "; ".join(reasons),
                }
            )
        table = pd.DataFrame(rows)
        table.to_csv(root / record["metrics"], index=False)
        runs.append(table)
        del estimator
    metrics = pd.concat(runs, ignore_index=True)
    metrics.to_csv(root / "run_metrics.csv", index=False)
    numeric = [
        column
        for column in metrics.select_dtypes(include="number")
        if column not in {"seed", "n_clusters"}
    ]
    grouped = metrics.groupby(["n_clusters", "dataset"], sort=True)
    summaries = grouped[numeric].agg(["mean", "std"])
    summaries.columns = [f"{column}_{statistic}" for column, statistic in summaries.columns]
    summaries = summaries.join(
        grouped.agg(
            n_seeds=("seed", "nunique"),
            all_seeds_pass_occupancy=("passes_occupancy", "all"),
            minimum_cluster_fraction=("cluster_min_fraction", "min"),
            maximum_empty_clusters=("cluster_empty_count", "max"),
        )
    )
    summaries.reset_index().to_csv(root / "k_summary.csv", index=False)

    ##################################################
    # 四、沿用运行清单引用全部模型，患者产物留在远端。
    ##################################################
    result = {
        "schema_version": 3,
        "split_dir": str(split_dir),
        "preproc_manifest": str(split_dir / "preproc_manifest.json"),
        "training_scope": manifest["training_scope"],
        "datasets": names,
        "has_independent_validation": validation is not None,
        "evaluation_datasets": [name for name in names if name not in {"train", "validation"}],
        "outcome": manifest["outcome"],
        "landmark_days": manifest["landmark_days"],
        "panel": manifest["panel"],
        "models": models,
        "metrics": "run_metrics.csv",
        "k_summary": "k_summary.csv",
        "min_cluster_fraction": config.min_cluster_fraction,
        "cindex_risk_score": config.trainer.cindex_risk_score,
        "survival_loss": config.model.survival_loss,
        "survival_format_version": SURVIVAL_FORMAT_VERSION,
        "interpretation": (
            "候选报告实际可用集指标；人工参考test指标选择K时属于探索评价，test不参与训练或早停。"
        ),
    }
    save_json(root / "run_manifest.json", result)
    LOGGER.info("CRC候选训练及指标汇总完成，未自动选择模型：%s", root)
    return result


@hydra.main(config_path="../../configs", config_name="crc_yunnan/run", version_base="1.3")
def main(raw_config: DictConfig) -> None:
    config = CRCRunConfig.model_validate(resolved_payload(raw_config))
    runtime = config.runtime
    if config.parallel.torch_threads is not None:
        torch.set_num_threads(config.parallel.torch_threads)
    if (
        runtime.torch_interop_threads is not None
        and torch.get_num_interop_threads() != runtime.torch_interop_threads
    ):
        torch.set_num_interop_threads(runtime.torch_interop_threads)
    if runtime.cudnn_benchmark is not None:
        torch.backends.cudnn.benchmark = runtime.cudnn_benchmark
    if runtime.cudnn_deterministic is not None:
        torch.backends.cudnn.deterministic = runtime.cudnn_deterministic
    if runtime.deterministic_algorithms is not None:
        torch.use_deterministic_algorithms(runtime.deterministic_algorithms)
    if runtime.cuda_matmul_allow_tf32 is not None:
        torch.backends.cuda.matmul.allow_tf32 = runtime.cuda_matmul_allow_tf32
    if runtime.cudnn_allow_tf32 is not None:
        torch.backends.cudnn.allow_tf32 = runtime.cudnn_allow_tf32
    # 与07独立进程保持一致：同时限制Torch、K-means使用的BLAS/OpenMP线程。
    with threadpool_limits(limits=config.parallel.torch_threads):
        run(config)


if __name__ == "__main__":
    main()

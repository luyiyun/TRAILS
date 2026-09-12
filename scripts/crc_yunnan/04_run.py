from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from scripts.configs import TrailsApplicationConfig

from trails import (
    ClinicalTimeSeriesDataset,
    ClusterNumberSelector,
    DataConfig,
    TrailsConfig,
    TrailsEstimator,
    resolve_batch_size,
)
from trails.artifacts import save_history_csv, save_json
from trails.metrics import cluster_assignment_diagnostics, concordance_index
from trails_simulate.config import resolved_payload

LOGGER = logging.getLogger(__name__)
SPLIT_NAMES = ("train", "validation", "test")


def _save_split_outputs(
    run_dir: Path,
    split_name: str,
    dataset: ClinicalTimeSeriesDataset,
    estimator: TrailsEstimator,
) -> dict[str, float | int]:
    """一次推理保存完整预测与简明患者表，并返回同口径的基础指标。"""
    prediction = estimator.predict(dataset)
    labels = prediction.predict()
    trainer = estimator.config.trainer
    risk = prediction.risk_score(trainer.risk_horizon, method=trainer.cindex_risk_score)
    median_time = torch.exp(-prediction.risk_score(method="median_survival"))
    survival_time = torch.stack([sample.survival_time for sample in dataset.samples])
    event = torch.stack([sample.event for sample in dataset.samples])
    split_dir = run_dir / split_name
    prediction.save(split_dir / "model_prediction.pt")
    frame = pd.DataFrame(
        {
            "patient_id": dataset.metadata["patient_ids"],
            "pred_cluster": labels.numpy(),
            "predicted_median_time_days": median_time.numpy(),
            "risk_score": risk.numpy(),
            "survival_time": survival_time.numpy(),
            "event": event.numpy(),
        }
    )
    probabilities = pd.DataFrame(
        prediction.predict_proba().numpy(),
        columns=[
            f"cluster_probability_{index}" for index in range(estimator.config.model.n_clusters)
        ],
    )
    pd.concat([frame, probabilities], axis=1).to_csv(split_dir / "patient_outputs.csv", index=False)
    return {
        "n_patients": len(dataset),
        "n_events": int(event.sum().item()),
        "cindex": concordance_index(risk, survival_time, event),
        **cluster_assignment_diagnostics(labels, n_clusters=estimator.config.model.n_clusters),
    }


def run(config: TrailsApplicationConfig) -> dict[str, object]:
    # ==================================================================================
    # 一、定位冻结数据和新输出目录，只读取 train 与 validation。
    # ==================================================================================
    split_dir = config.split.dir.resolve()
    run_dir = config.paths.dir.resolve()
    selection_dir = (run_dir / config.k_selection.result_dir).resolve()
    manifest_path = run_dir / config.outputs.summary
    if config.swanlab.enabled:
        raise ValueError("CRC建模仅保存训练历史和日志，请设置swanlab.enabled=false")
    if run_dir == split_dir or split_dir in run_dir.parents:
        raise ValueError("建模输出不得写入冻结split目录")
    targets = [
        run_dir / "model.pt",
        run_dir / "training_history.csv",
        run_dir / "metrics.csv",
        manifest_path,
        *(run_dir / name for name in SPLIT_NAMES),
    ]
    if config.n_clusters is None:
        targets.append(selection_dir)
    if existing := [str(path) for path in targets if path.exists()]:
        raise FileExistsError(f"拒绝覆盖已有建模产物：{existing}")
    run_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = json.loads((split_dir / "split_manifest.json").read_text(encoding="utf-8"))
    datasets = {
        name: ClinicalTimeSeriesDataset.load(split_dir / name / "dataset.pt")
        for name in ("train", "validation")
    }
    train, validation = datasets["train"], datasets["validation"]
    LOGGER.info("冻结划分 %s：train=%d，validation=%d", split_dir, len(train), len(validation))

    # ==================================================================================
    # 二、复用训练预设；显式传入冻结验证集，不再内部切分。
    # ==================================================================================
    initial_k = config.n_clusters or config.k_selection.candidate_clusters[0]
    trails_config = TrailsConfig(
        data=DataConfig(n_features=train.n_features),
        model=config.model.model_copy(update={"n_clusters": initial_k}),
        trainer=config.trainer.model_copy(
            update={
                "batch_size": resolve_batch_size(len(train), config.trainer.batch_size),
                "valid_size": 0.0,
            }
        ),
        seed=config.trainer.seed,
    )

    # ==================================================================================
    # 三、只在 train/validation 上选择 K，再在入选 K 内确定最终 seed。
    # ==================================================================================
    if config.n_clusters is None:
        selection = config.k_selection
        selector = ClusterNumberSelector(
            selection.candidate_clusters,
            seeds=selection.seeds,
            selection_rule=selection.selection_rule,
            require_non_empty=selection.require_non_empty,
            min_cluster_fraction=selection.min_cluster_fraction,
            min_mean_pairwise_ari=selection.min_mean_pairwise_ari,
            compute_stability=selection.compute_stability,
            estimator_config=trails_config,
        )
        result = selector.select(train, validation_data=validation)
        result.save(selection_dir)
        if result.selected_k is None:
            raise RuntimeError("没有候选K通过配置门槛；选择结果已保存，test尚未读取")
        candidates = result.run_metrics.loc[result.run_metrics["n_clusters"] == result.selected_k]
        best = candidates.sort_values(
            ["selection_score", "cindex", "latent_mixture_bic", "seed"],
            ascending=[False, False, True, True],
        ).iloc[0]
        selected_seed = int(best["seed"])
        estimator = result.selected_estimators[selected_seed]
    else:
        selected_seed = config.trainer.seed
        estimator = TrailsEstimator(trails_config).fit(train, validation_data=validation)

    # ==================================================================================
    # 四、锁定并保存模型后读取 test；三套基础指标使用相同风险口径。
    # ==================================================================================
    selected_k = estimator.config.model.n_clusters
    LOGGER.info("模型已锁定：K=%d，seed=%d；开始三套预测", selected_k, selected_seed)
    estimator.save(run_dir / "model.pt")
    save_history_csv(run_dir / "training_history.csv", estimator.history)
    datasets["test"] = ClinicalTimeSeriesDataset.load(split_dir / "test" / "dataset.pt")
    metrics = pd.DataFrame(
        [
            {"split": name, **_save_split_outputs(run_dir, name, datasets[name], estimator)}
            for name in SPLIT_NAMES
        ]
    )
    metrics.to_csv(run_dir / "metrics.csv", index=False)

    # ==================================================================================
    # 五、最后保存简短运行引用；原数据、模型与患者结果均保留在远端。
    # ==================================================================================
    manifest: dict[str, object] = {
        "schema_version": 1,
        "split_dir": str(split_dir),
        "outcome": split_manifest["outcome"],
        "landmark_days": split_manifest["landmark_days"],
        "panel": split_manifest["panel"],
        "selected_k": selected_k,
        "selected_seed": selected_seed,
        "cindex_risk_score": estimator.config.trainer.cindex_risk_score,
        "model": "model.pt",
        "training_history": "training_history.csv",
        "metrics": "metrics.csv",
        "k_selection_dir": str(selection_dir) if config.n_clusters is None else None,
    }
    save_json(manifest_path, manifest)
    LOGGER.info("CRC TRAILS建模完成：%s", run_dir)
    return manifest


@hydra.main(config_path="../../configs", config_name="crc_yunnan/run", version_base="1.3")
def main(raw_config: DictConfig) -> None:
    config = TrailsApplicationConfig.model_validate(resolved_payload(raw_config))
    run(config)


if __name__ == "__main__":
    main()

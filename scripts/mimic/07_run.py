"""在MIMIC冻结划分上训练全部候选，保存可重载结果而不选择最佳K。"""

from __future__ import annotations

import json
import logging
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from scripts.utils.candidates import save_candidate_outputs
from trails import ClinicalTimeSeriesDataset, ClusterNumberEvaluator, DataConfig, TrailsConfig
from trails.artifacts import plot_history, save_json
from trails.survival import SURVIVAL_FORMAT_VERSION
from trails_simulate.config import resolved_payload

from .config import BASELINE_COVARIATE_COLUMNS, MimicApplicationConfig

LOGGER = logging.getLogger(__name__)


def run(config: MimicApplicationConfig) -> dict[str, Any]:
    ##################################################
    # 一、只读取train/validation，复用06冻结的预处理和划分。
    ##################################################
    source, root = config.split.dir.resolve(), config.paths.dir.resolve()
    if root == source or source in root.parents or root in source.parents:
        raise ValueError("建模输出不得写入冻结06目录")
    targets = [
        root / name
        for name in ("run_manifest.json", "resolved_config.yaml", "models", "evaluator.json")
    ]
    if existing := [str(path) for path in targets if path.exists()]:
        raise FileExistsError(f"拒绝覆盖已有建模产物：{existing}")
    if config.swanlab.enabled:
        raise ValueError("候选批量报告请设置swanlab.enabled=false")
    split_manifest = json.loads((source / "split_manifest.json").read_text())
    if split_manifest["split_seed"] != config.split.seed:
        raise ValueError("split.seed与冻结划分不一致")
    train = ClinicalTimeSeriesDataset.load(source / "train/dataset.pt")
    validation = ClinicalTimeSeriesDataset.load(source / "validation/dataset.pt")
    root.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config.model_dump(mode="json"), root / "resolved_config.yaml")

    ##################################################
    # 二、调用公共评价器；所有候选冻结后才读取test，不自动决定K。
    ##################################################
    clusters = config.n_clusters if isinstance(config.n_clusters, list) else [config.n_clusters]
    evaluator = ClusterNumberEvaluator(
        clusters,
        seeds=config.seeds,
        min_cluster_fraction=config.min_cluster_fraction,
        estimator_config=TrailsConfig(
            data=DataConfig(n_features=train.n_features),
            model=config.model,
            trainer=config.trainer,
            seed=config.trainer.seed,
        ),
    )
    evaluator.fit(
        train,
        validation_data=validation,
        evaluation_data={"test": source / "test/dataset.pt"},
        result_dir=root,
    )

    ##################################################
    # 三、导出冻结预测与来源清单；各模型保持独立可读取。
    ##################################################
    save_candidate_outputs(
        evaluator,
        {
            "train": train,
            "validation": validation,
            "test": source / "test/dataset.pt",
        },
        baseline_columns=BASELINE_COVARIATE_COLUMNS,
    )
    if "plot" in config.artifacts.names or "all" in config.artifacts.names:
        for candidate in evaluator.candidates:
            model = evaluator.load_estimator(
                seed=candidate["seed"], n_clusters=candidate["n_clusters"]
            )
            plot_history(root / candidate["prediction_dir"] / "training_history.png", model.history)
            del model
    summary = {
        "schema_version": 2,
        "evaluator": "evaluator.json",
        "data": {
            "split_dir": str(source),
            "split_seed": config.split.seed,
            "split_sizes": split_manifest["split_counts"],
            "n_features": train.n_features,
        },
        "datasets": ["train", "validation", "test"],
        "models": evaluator.candidates,
        "metrics": "run_metrics.csv",
        "k_summary": "k_summary.csv",
        "min_cluster_fraction": config.min_cluster_fraction,
        "cindex_risk_score": config.trainer.cindex_risk_score,
        "risk_horizon": config.trainer.risk_horizon,
        "survival_format_version": SURVIVAL_FORMAT_VERSION,
        "interpretation": "全部候选冻结后报告各集；人工参考test选择模型时属于探索分析。",
    }
    save_json(root / "run_manifest.json", summary)
    LOGGER.info("MIMIC全部候选已保存，未自动选择模型：%s", root)
    return summary


@hydra.main(config_path="../../configs", config_name="mimic/run", version_base="1.3")
def main(raw_config: DictConfig) -> None:
    run(MimicApplicationConfig.model_validate(resolved_payload(raw_config)))


if __name__ == "__main__":
    main()

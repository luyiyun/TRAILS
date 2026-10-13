from __future__ import annotations

import json
import logging
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from scripts.crc_yunnan.config import CRCRunConfig
from scripts.utils.candidates import save_candidate_outputs
from threadpoolctl import threadpool_limits

from trails import (
    ClinicalTimeSeriesDataset,
    ClusterNumberEvaluator,
    DataConfig,
    TrailsConfig,
)
from trails.artifacts import save_json
from trails.survival import SURVIVAL_FORMAT_VERSION
from trails_simulate.config import resolved_payload

LOGGER = logging.getLogger(__name__)


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
    evaluator = ClusterNumberEvaluator(
        clusters,
        seeds=config.seeds,
        require_non_empty=True,
        min_cluster_fraction=config.min_cluster_fraction,
        estimator_config=TrailsConfig(
            data=DataConfig(n_features=train.n_features),
            model=config.model,
            trainer=config.trainer,
            seed=config.trainer.seed,
        ),
    )
    extra = {name: split_dir / name / "dataset.pt" for name in names if name not in datasets}
    evaluator.fit(train, validation_data=validation, evaluation_data=extra, result_dir=root)

    ##################################################
    # 三、复用冻结预测导出患者表及候选配置，不重复训练或计算指标。
    ##################################################
    save_candidate_outputs(evaluator, {name: split_dir / name / "dataset.pt" for name in names})
    models = evaluator.candidates

    ##################################################
    # 四、沿用运行清单引用全部模型，患者产物留在远端。
    ##################################################
    result = {
        "schema_version": 3,
        "evaluator": "evaluator.json",
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

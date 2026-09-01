"""在07冻结划分上训练基线，保存模型与三划分预测，不在此选择测试结果。"""

from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from trails import TrailsConfig, TrailsEstimator
from trails.artifacts import plot_history, save_history_csv, save_json
from trails_simulate.config import resolved_payload

from ..utils.baseline_features import dataset_patient_ids
from .baseline_registry import BASELINE_REGISTRY
from .config import MimicBaselinesConfig
from .frozen import load_frozen_datasets, sha256_file
from .paths import resolve_input_path

LOGGER = logging.getLogger(__name__)


def run(config: MimicBaselinesConfig) -> dict[str, Any]:
    """逐method×seed拟合，失败保留状态并继续其他方法，整批不伪报成功。"""
    source = resolve_input_path(config.input_dir)
    output = config.paths.dir.resolve()
    manifest_path = output / "baselines_manifest.json"
    pending = manifest_path.with_suffix(".tmp")
    if manifest_path.exists():
        raise FileExistsError(f"拒绝覆盖既有基线运行：{output}")
    source_manifest = json.loads((source / "run_manifest.json").read_text())
    original = TrailsConfig.model_validate(source_manifest["training"])
    n_clusters = config.n_clusters or original.model.n_clusters
    if n_clusters != original.model.n_clusters:
        raise ValueError("对比基线必须使用与07一致的K")
    if config.risk_horizon != original.trainer.risk_horizon:
        raise ValueError("对比基线必须使用与07一致的risk_horizon")
    datasets = load_frozen_datasets(source)
    output.mkdir(parents=True, exist_ok=True)
    input_hashes = {
        name: sha256_file(source / name)
        for name in ["run_manifest.json", "preprocessing_parameters.csv"]
        + [f"{split}/dataset.pt" for split in datasets]
    }
    records: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "format_version": 1,
        "status": "running",
        "input_dir": str(source),
        "source_sha256": input_hashes,
        "n_clusters": n_clusters,
        "risk_horizon": config.risk_horizon,
        "prediction_times": config.prediction_times,
        "split_sizes": {name: len(data) for name, data in datasets.items()},
        "config": config.model_dump(mode="json"),
        "methods": records,
    }
    total = sum(len(method.seeds) for method in config.methods)
    started = time.perf_counter()
    prediction_times = np.asarray(config.prediction_times, dtype=np.float64)
    for method in config.methods:
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
                "prediction_format": "trails"
                if method.kind == "trails_no_survival"
                else "baseline",
                "capabilities": ["cluster"]
                if method.kind == "trails_no_survival"
                else sorted(BASELINE_REGISTRY[method.kind].capabilities),
                "config": method.model_dump(mode="json"),
            }
            records.append(record)
            save_json(pending, manifest)
            pending.replace(manifest_path)
            LOGGER.info(
                "Baseline %s/%s: %s seed=%s fit started", len(records), total, method.name, seed
            )
            method_started = time.perf_counter()
            try:
                if method.kind == "trails_no_survival":
                    payload = original.model_dump(mode="json")
                    payload["model"]["loss"]["survival_weight"] = 0.0
                    payload["trainer"]["seed"] = seed
                    payload["seed"] = seed
                    training = TrailsConfig.model_validate(payload)
                    record["training"] = training.model_dump(mode="json")
                    estimator = TrailsEstimator(training).fit(
                        datasets["train"], validation_data=datasets["validation"]
                    )
                    estimator.save(method_dir / "model.pt")
                    save_history_csv(method_dir / "training_history.csv", estimator.history)
                    save_json(method_dir / "training_history.json", estimator.history)
                    plot_history(method_dir / "training_history.png", estimator.history)
                    for split, data in datasets.items():
                        estimator.predict(data).save(method_dir / split / "model_prediction.pt")
                    del estimator
                else:
                    baseline = BASELINE_REGISTRY[method.kind].factory(
                        method, n_clusters, seed, method_dir
                    )
                    baseline.fit(datasets["train"], datasets["validation"])
                    suffix = "rds" if method.kind in {"mpjlcmm", "jmbayes2"} else "joblib"
                    baseline.save_model(method_dir / f"model.{suffix}")
                    for split, data in datasets.items():
                        LOGGER.info("%s seed=%s predict split=%s", method.name, seed, split)
                        prediction = baseline.predict(
                            data,
                            prediction_times=prediction_times,
                            risk_horizon=config.risk_horizon,
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
            except Exception as error:
                record["status"] = "failed"
                record["error"] = {"type": type(error).__name__, "message": str(error)}
                LOGGER.error(
                    "%s seed=%s failed (%s); see method manifest",
                    method.name,
                    seed,
                    type(error).__name__,
                )
            finally:
                record["elapsed_seconds"] = time.perf_counter() - method_started
                save_json(method_dir / "method_manifest.json", record)
                manifest["elapsed_seconds"] = time.perf_counter() - started
                save_json(pending, manifest)
                pending.replace(manifest_path)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            LOGGER.info(
                "Baseline %s/%s: %s status=%s elapsed=%.1fs total=%.1fs",
                len(records),
                total,
                method.name,
                record["status"],
                record["elapsed_seconds"],
                manifest["elapsed_seconds"],
            )
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

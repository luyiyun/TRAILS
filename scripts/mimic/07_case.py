from __future__ import annotations

import time

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from trails import (
    ClinicalTimeSeriesDataset,
    DataConfig,
    TrailsConfig,
    TrailsEstimator,
    resolve_batch_size,
)
from trails.artifacts import save_history_csv, save_json
from trails_case.config import CaseApplicationConfig
from trails_case.evaluation import (
    CaseResultTables,
    evaluate_case_predictions,
    prediction_payload_from_case_dataset,
)
from trails_case.mimic import SPLIT_SEED, prepare_mimic_datasets
from trails_case.outputs import resolve_input_path, resolve_output_path
from trails_simulate.config import resolved_payload


def _cluster_feature_summary(
    dataset: ClinicalTimeSeriesDataset,
    pred_cluster: torch.Tensor,
    n_clusters: int,
    normalization: pd.DataFrame,
) -> pd.DataFrame:
    sums = torch.zeros(n_clusters, dataset.n_features)
    counts = torch.zeros(n_clusters, dataset.n_features)
    for sample, cluster in zip(dataset.samples, pred_cluster.tolist(), strict=True):
        aligned = sample.to_aligned()
        sums[cluster] += (aligned.x * aligned.mask).sum(dim=0)
        counts[cluster] += aligned.mask.sum(dim=0)

    means = torch.where(counts > 0, sums / counts, torch.nan).numpy()
    summary = pd.DataFrame(
        {
            "pred_cluster": np.repeat(np.arange(n_clusters), dataset.n_features),
            "feature": np.tile(dataset.feature_names, n_clusters),
            "n_observations": counts.numpy().reshape(-1).astype(int),
            "mean_value_standardized": means.reshape(-1),
        }
    )
    summary = summary.merge(
        normalization[["feature", "center", "scale"]], on="feature", validate="many_to_one"
    )
    summary["mean_value_raw"] = (
        summary["mean_value_standardized"] * summary["scale"] + summary["center"]
    )
    return summary


def run(config: CaseApplicationConfig) -> dict[str, object]:
    if config.k_selection.enabled:
        raise ValueError("K 选择实验已移至 scripts/mimic/06_select_k.py")
    run_dir = config.paths.dir
    run_dir.mkdir(parents=True, exist_ok=True)
    patients_csv = resolve_input_path(config.patients_csv)
    observations_csv = resolve_input_path(config.observations_csv)
    required = [patients_csv, observations_csv]
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(f"缺少 MIMIC 建模输入：{missing}")
    device = config.trainer.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("MIMIC TRAILS 分析要求可用 CUDA GPU")

    load_started = time.perf_counter()
    datasets, transformer = prepare_mimic_datasets(
        patients_csv, observations_csv, config.feature_order, config.description
    )
    train = datasets["train"]
    validation = datasets["validation"]
    test = datasets["test"]
    load_seconds = time.perf_counter() - load_started
    trails_config = TrailsConfig(
        data=DataConfig(n_features=train.n_features),
        model=config.model,
        trainer=config.trainer.model_copy(
            update={
                "batch_size": resolve_batch_size(len(train), config.trainer.batch_size),
                "valid_size": 0.0,
            }
        ),
        seed=config.trainer.seed,
    )
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    training_started = time.perf_counter()
    estimator = TrailsEstimator(trails_config).fit(train, validation_data=validation)
    training_seconds = time.perf_counter() - training_started

    prediction_started = time.perf_counter()
    pred_cluster = estimator.predict(test)
    risk_score = estimator.predict_risk(test)
    prediction = prediction_payload_from_case_dataset(
        test,
        patient_ids=list(test.metadata["patient_ids"]),
        pred_cluster=pred_cluster,
        risk_score=risk_score,
    )
    metrics = evaluate_case_predictions(prediction, n_clusters=trails_config.model.n_clusters)
    prediction_seconds = time.perf_counter() - prediction_started

    tables = CaseResultTables(prediction)
    cluster_summary = tables.cluster_summary(n_clusters=trails_config.model.n_clusters)
    assert transformer.parameters_ is not None
    cluster_features = _cluster_feature_summary(
        test, pred_cluster, trails_config.model.n_clusters, transformer.parameters_
    )
    cluster_summary.to_csv(
        resolve_output_path(config.outputs.cluster_summary, run_dir), index=False
    )
    cluster_features.to_csv(
        resolve_output_path(config.outputs.cluster_feature_summary, run_dir), index=False
    )
    save_history_csv(run_dir / "training_history.csv", estimator.history)
    save_json(run_dir / "training_history.json", estimator.history)
    peak_memory = (
        torch.cuda.max_memory_allocated(device) / 1024**3 if device.startswith("cuda") else 0.0
    )
    summary = {
        "data": {
            "split_seed": SPLIT_SEED,
            "split_sizes": {name: len(data) for name, data in datasets.items()},
            "n_features": train.n_features,
            "test_observations": int(test.metadata["n_observations"]),
        },
        "metrics": metrics,
        "resources": {
            "device": device,
            "batch_size": trails_config.trainer.batch_size,
            "load_seconds": load_seconds,
            "training_seconds": training_seconds,
            "prediction_seconds": prediction_seconds,
            "peak_gpu_memory_gib": peak_memory,
        },
        "training": trails_config.model_dump(mode="json"),
        "k_selection": None,
    }
    save_json(resolve_output_path(config.outputs.summary, run_dir), summary)
    return summary


@hydra.main(config_path="../../configs", config_name="mimic_case", version_base="1.3")
def main(raw_config: DictConfig) -> None:
    config = CaseApplicationConfig.model_validate(resolved_payload(raw_config))
    run(config)


if __name__ == "__main__":
    main()

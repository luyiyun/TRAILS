from __future__ import annotations

from pathlib import Path
from typing import Any

from trails.data import ClinicalTimeSeriesDataset

from .config import SimulateApplicationConfig
from .generators import ClinicalTimeSeriesDatasetGeneratorConfig


def generator_config_for_cluster(
    generator_config: ClinicalTimeSeriesDatasetGeneratorConfig,
    *,
    n_clusters: int,
) -> ClinicalTimeSeriesDatasetGeneratorConfig:
    return generator_config.model_copy(update={"n_clusters": n_clusters}, deep=True)


def simulation_mechanism_seed(config: SimulateApplicationConfig, *, cluster_index: int) -> int:
    base_seed = config.mechanism_seed or config.seed
    return base_seed + cluster_index * 100


def simulation_sample_seed(
    config: SimulateApplicationConfig,
    *,
    size_index: int,
    cluster_index: int,
    repeat_index: int,
) -> int:
    return config.seed + size_index * 10_000 + cluster_index * 100 + repeat_index


def simulation_split_summary(
    dataset: ClinicalTimeSeriesDataset,
    *,
    clusters: int,
    out: Path,
    seed: int,
) -> dict[str, Any]:
    event_rate = sum(float(sample.event) for sample in dataset) / len(dataset)
    return {
        "censoring_rate": 1.0 - event_rate,
        "clusters": clusters,
        "features": dataset.feature_names,
        "n_features": dataset.n_features,
        "n_patients": len(dataset),
        "out": str(out),
        "seed": seed,
    }

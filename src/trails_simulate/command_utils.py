from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from trails.data import ClinicalTimeSeriesDataset

from .config import PathsConfig
from .path import DatasetRunPaths


class DatasetSourceConfig(Protocol):
    paths: PathsConfig


def metric_row(
    run_paths: DatasetRunPaths,
    *,
    method: str,
    prediction_path: Path,
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "data_root": str(run_paths.data_root),
        "method": method,
        "prediction_path": str(prediction_path),
        "run_id": run_paths.run_id,
        **metrics,
    }


def dataset_source_payload(
    config: DatasetSourceConfig,
    runs: list[DatasetRunPaths],
) -> dict[str, Any]:
    if not runs:
        return {"data_root": None, "source": "empty"}
    if config.paths.explicit_split.enabled:
        root = runs[0].data_root
        source = "explicit split"
    else:
        root = config.paths.data_root
        source = "data root"
    return {
        "data_root": str(root),
        "source": source,
    }


def dataset_n_clusters(dataset: ClinicalTimeSeriesDataset, *, fallback: int) -> int:
    params = dataset.metadata.get("generation_params")
    if isinstance(params, Mapping) and "n_clusters" in params:
        return int(params["n_clusters"])
    return fallback


def format_completed_train_run(
    *,
    run_id: str,
    n_clusters: int,
    seed: int,
    prediction_path: Path,
    metrics: Mapping[str, float],
) -> str:
    metric_names = ("cindex", "ari", "nmi", "acc", "cluster_empty_count")
    metric_text = " ".join(
        f"{name}={float(metrics[name]):.4g}" for name in metric_names if name in metrics
    )
    fields = [f"Completed train run: {run_id}", f"k={n_clusters}", f"seed={seed}"]
    if metric_text:
        fields.append(metric_text)
    fields.append(f"prediction={compact_log_path(prediction_path)}")
    return " ".join(fields)


def compact_log_path(path: Path, *, keep_parts: int = 4) -> str:
    parts = path.parts
    if len(parts) <= keep_parts:
        return str(path)
    return str(Path("...").joinpath(*parts[-keep_parts:]))

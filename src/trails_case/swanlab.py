from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import swanlab

from trails.config import TrailsConfig
from trails.estimator import (
    KSelectionMetrics,
    best_selection_metrics,
    selected_k_from_selection_metrics,
)
from trails.trainer import HistoryEntry

from .config import CaseApplicationConfig, DiagnosticsConfig, SwanLabConfig
from .outputs import CaseOutputPaths, k_selection_payload, output_payload


def start_swanlab_run(
    swanlab_config: SwanLabConfig,
    trails_config: TrailsConfig,
    app_config: CaseApplicationConfig,
    outputs: CaseOutputPaths,
    artifacts: frozenset[str],
    diagnostics_config: DiagnosticsConfig,
    k_selection_metrics: KSelectionMetrics | None = None,
    k_selection_result_dir: Path | None = None,
) -> None:
    if not swanlab_config.enabled:
        return

    experiment_name = swanlab_config.experiment or datetime.now().astimezone().strftime(
        "trails-case-%Y%m%d-%H%M%S"
    )
    init_kwargs: dict[str, Any] = {
        "project": swanlab_config.project,
        "experiment_name": experiment_name,
        "config": {
            "case": app_config.model_dump(mode="json"),
            "config": trails_config.model_dump(mode="json"),
            "diagnostics": diagnostics_config.model_dump(mode="json"),
            "outputs": output_payload(outputs, k_selection_result_dir),
            "save_artifacts": sorted(artifacts),
            "swanlab": swanlab_config.model_dump(mode="json"),
        },
    }
    if k_selection_metrics is not None:
        init_kwargs["config"]["k_selection"] = k_selection_payload(
            k_selection_metrics,
            k_selection_result_dir,
        )
    if swanlab_config.mode is not None:
        init_kwargs["mode"] = swanlab_config.mode
    swanlab.init(**init_kwargs)


def log_swanlab_history(entry: HistoryEntry) -> None:
    metrics = {
        "epoch/global": entry["global_epoch"],
        "epoch/local": entry["epoch"],
        **{f"train/{key}": value for key, value in entry["train"].items()},
    }
    if "valid" in entry:
        metrics.update({f"val/{key}": value for key, value in entry["valid"].items()})
    swanlab.log(metrics, step=entry["global_epoch"])


def log_swanlab_case_metrics(
    metrics: Mapping[str, float],
    history: list[HistoryEntry],
    k_selection_metrics: KSelectionMetrics | None = None,
) -> None:
    step = int(float(history[-1]["global_epoch"])) if history else 0
    payload = {f"case/{name}": value for name, value in metrics.items()}
    if k_selection_metrics is not None:
        payload.update(swanlab_k_selection_metrics(k_selection_metrics))
    swanlab.log(payload, step=step)


def swanlab_k_selection_metrics(metrics: KSelectionMetrics) -> dict[str, float | int]:
    best = best_selection_metrics(metrics)
    return {
        "selection/selected_k": selected_k_from_selection_metrics(metrics),
        "selection/best_cindex": float(best["cindex"]),
        "selection/best_bic": float(best["bic"]),
        "selection/best_score": float(best["selection_score"]),
    }

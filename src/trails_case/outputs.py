from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from trails.artifacts import (
    plot_history,
    save_history_csv,
    save_json,
    save_latent_embedding_artifacts,
)
from trails.config import TrailsConfig
from trails.data import ClinicalTimeSeriesDataset
from trails.estimator import (
    KSelectionMetrics,
    TrailsEstimator,
    best_selection_metrics,
    selected_k_from_selection_metrics,
)

from .config import CaseApplicationConfig
from .evaluation import json_safe_metrics


@dataclass(frozen=True)
class CaseOutputPaths:
    dataset: Path
    dataset_summary: Path
    predictions: Path
    patient_clusters: Path
    cluster_summary: Path
    cluster_feature_summary: Path
    summary: Path

    @classmethod
    def from_config(cls, config: CaseApplicationConfig) -> CaseOutputPaths:
        run_dir = config.paths.dir
        return cls(
            dataset=resolve_output_path(config.outputs.dataset, run_dir),
            dataset_summary=resolve_output_path(config.outputs.dataset_summary, run_dir),
            predictions=resolve_output_path(config.outputs.predictions, run_dir),
            patient_clusters=resolve_output_path(config.outputs.patient_clusters, run_dir),
            cluster_summary=resolve_output_path(config.outputs.cluster_summary, run_dir),
            cluster_feature_summary=resolve_output_path(
                config.outputs.cluster_feature_summary,
                run_dir,
            ),
            summary=resolve_output_path(config.outputs.summary, run_dir),
        )


def configure_torch_threads(torch_threads: int | None) -> None:
    if torch_threads is None:
        return
    torch.set_num_threads(torch_threads)


def resolve_input_path(path: Path, base_dir: Path | None = None) -> Path:
    return path if path.is_absolute() else (base_dir or Path.cwd()) / path


def resolve_output_path(path: Path, run_dir: Path) -> Path:
    return path if path.is_absolute() else run_dir / path


def save_case_training_artifacts(
    *,
    config: CaseApplicationConfig,
    trails_config: TrailsConfig,
    estimator: TrailsEstimator,
    dataset: ClinicalTimeSeriesDataset,
    metrics: Mapping[str, float],
    outputs: CaseOutputPaths,
    artifacts: frozenset[str],
    k_selection_metrics: KSelectionMetrics | None = None,
    k_selection_result_dir: Path | None = None,
) -> None:
    should_save_diagnostics = config.diagnostics.latent_embeddings.enabled
    if not artifacts and not should_save_diagnostics:
        return

    created_at = datetime.now().astimezone()
    run_dir = outputs.summary.parent

    if "config" in artifacts:
        save_json(
            run_dir / "config.json",
            case_run_config(
                app_config=config,
                trails_config=trails_config,
                outputs=outputs,
                artifacts=artifacts,
                created_at=created_at,
                k_selection_metrics=k_selection_metrics,
                k_selection_result_dir=k_selection_result_dir,
            ),
        )
    if "history" in artifacts:
        save_json(run_dir / "history.json", estimator.history)
        save_history_csv(run_dir / "history.csv", estimator.history)
    if "test" in artifacts:
        save_json(run_dir / "case_metrics.json", json_safe_metrics(metrics))
    if "model" in artifacts:
        estimator.save(run_dir / "model.pt")
    if "plot" in artifacts:
        plot_history(run_dir / "history.png", estimator.history)
    if should_save_diagnostics:
        diagnostics = estimator.latent_diagnostics(dataset)
        save_latent_embedding_artifacts(
            run_dir,
            "case",
            diagnostics,
            random_state=trails_config.seed,
        )


def output_payload(
    outputs: CaseOutputPaths,
    k_selection_result_dir: Path | None = None,
) -> dict[str, str]:
    payload = {
        "case_summary": str(outputs.summary),
        "cluster_feature_summary": str(outputs.cluster_feature_summary),
        "cluster_summary": str(outputs.cluster_summary),
        "dataset": str(outputs.dataset),
        "dataset_summary": str(outputs.dataset_summary),
        "patient_clusters": str(outputs.patient_clusters),
        "predictions": str(outputs.predictions),
    }
    if k_selection_result_dir is not None:
        payload["k_selection"] = str(k_selection_result_dir)
    return payload


def case_run_config(
    *,
    app_config: CaseApplicationConfig,
    trails_config: TrailsConfig,
    outputs: CaseOutputPaths,
    artifacts: frozenset[str],
    created_at: datetime,
    k_selection_metrics: KSelectionMetrics | None = None,
    k_selection_result_dir: Path | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "artifacts": sorted(artifacts),
        "case": app_config.model_dump(mode="json"),
        "config": trails_config.model_dump(mode="json"),
        "created_at": created_at.isoformat(timespec="seconds"),
        "diagnostics": app_config.diagnostics.model_dump(mode="json"),
        "outputs": output_payload(outputs, k_selection_result_dir),
        "swanlab": app_config.swanlab.model_dump(mode="json"),
        "train_args": {
            "batch_size": trails_config.trainer.batch_size,
            "clusters": trails_config.model.n_clusters,
            "decoder_conditioning": trails_config.model.decoder.conditioning,
            "decoder_hidden_dim": trails_config.model.decoder.hidden_dim,
            "decoder_kind": trails_config.model.decoder.kind,
            "decoder_n_layers": trails_config.model.decoder.n_layers,
            "dropout": trails_config.model.dropout,
            "encoder_input_hidden_dim": trails_config.model.encoder.input.hidden_dim,
            "encoder_input_kind": trails_config.model.encoder.input.kind,
            "encoder_mapping_hidden_dim": trails_config.model.encoder.mapping.hidden_dim,
            "encoder_mapping_kind": trails_config.model.encoder.mapping.kind,
            "encoder_mapping_n_layers": trails_config.model.encoder.mapping.n_layers,
            "epochs": trails_config.trainer.max_epochs,
            "latent_dim": trails_config.model.latent_dim,
            "learning_rate": trails_config.trainer.learning_rate,
            "loss_cluster_weight": trails_config.model.loss.cluster_weight,
            "loss_reconstruction_weight": trails_config.model.loss.reconstruction_weight,
            "loss_survival_weight": trails_config.model.loss.survival_weight,
            "loss_weighting": trails_config.model.loss.weighting,
            "seed": trails_config.seed,
            "survival_head_hidden_layers": trails_config.model.survival_head_hidden_layers,
            "warmup_epochs": trails_config.trainer.warmup_epochs,
        },
    }
    if k_selection_metrics is not None:
        payload["k_selection"] = k_selection_payload(k_selection_metrics, k_selection_result_dir)
    return payload


def k_selection_payload(
    metrics: KSelectionMetrics,
    result_dir: Path | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "selected_k": selected_k_from_selection_metrics(metrics),
        "best": best_selection_metrics(metrics),
        "metrics": metrics,
    }
    if result_dir is not None:
        payload["result_dir"] = str(result_dir)
    return payload


def compact_log_path(path: Path, *, keep_parts: int = 4) -> str:
    parts = path.parts
    if len(parts) <= keep_parts:
        return str(path)
    return str(Path("...").joinpath(*parts[-keep_parts:]))

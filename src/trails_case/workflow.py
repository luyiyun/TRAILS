from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import swanlab
import torch

from trails.artifacts import (
    plot_history,
    resolve_artifact_names,
    save_history_csv,
    save_json,
    save_latent_embedding_artifacts,
)
from trails.config import DataConfig, TrailsConfig, resolve_batch_size
from trails.data import ClinicalTimeSeriesDataset
from trails.estimator import TrailsEstimator
from trails.trainer import HistoryEntry

from .config import CaseApplicationConfig, DiagnosticsConfig, SwanLabConfig
from .data import case_dataset_summary, load_case_dataset_from_csv
from .evaluation import (
    cluster_feature_summary_rows,
    cluster_summary_rows,
    evaluate_case_predictions,
    json_safe_metrics,
    prediction_payload_from_case_dataset,
    save_cluster_feature_summary_csv,
    save_cluster_summary_csv,
    save_patient_clusters_csv,
    save_prediction_payload,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaseOutputPaths:
    dataset: Path
    dataset_summary: Path
    predictions: Path
    patient_clusters: Path
    cluster_summary: Path
    cluster_feature_summary: Path
    summary: Path


def run_case_command(
    config: CaseApplicationConfig,
    hydra_run_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    configure_torch_threads(config.training.parallel.torch_threads)
    hydra_run_dir.mkdir(parents=True, exist_ok=True)
    outputs = case_output_paths(config, hydra_run_dir)
    patients_csv = resolve_project_path(config.case.patients_csv, project_root)
    observations_csv = resolve_project_path(config.case.observations_csv, project_root)

    imported = load_case_dataset_from_csv(
        patients_csv=patients_csv,
        observations_csv=observations_csv,
        columns=config.case.columns,
        description=config.case.description,
        feature_order=config.case.feature_order,
    )
    dataset = imported.dataset
    dataset.save(outputs.dataset)
    dataset_summary = case_dataset_summary(imported)
    save_json(outputs.dataset_summary, dataset_summary)

    seed = config.training.trainer.seed
    artifacts = resolve_artifact_names(config.training.artifacts.names)
    trails_config = TrailsConfig(
        data=DataConfig(n_features=dataset.n_features),
        model=config.training.model,
        trainer=config.training.trainer.model_copy(
            update={
                "batch_size": resolve_batch_size(len(dataset), config.training.trainer.batch_size),
                "seed": seed,
            }
        ),
        seed=seed,
    )

    start_swanlab_run(
        config.training.swanlab,
        trails_config,
        config,
        outputs,
        artifacts,
        config.training.diagnostics,
    )
    estimator = TrailsEstimator(trails_config)
    try:
        estimator.fit(
            dataset,
            history_callback=log_swanlab_history if config.training.swanlab.enabled else None,
        )
        prediction = prediction_payload_from_case_dataset(
            dataset,
            patient_ids=list(dataset.metadata["patient_ids"]),
            pred_cluster=estimator.predict(dataset),
            risk_score=estimator.predict_risk(dataset),
            cluster_probabilities=estimator.predict_proba(dataset),
        )
        metrics = evaluate_case_predictions(
            prediction,
            n_clusters=trails_config.model.n_clusters,
        )
        if config.training.swanlab.enabled:
            log_swanlab_case_metrics(metrics, estimator.history)
    finally:
        if config.training.swanlab.enabled:
            swanlab.finish()

    save_case_training_artifacts(
        config=config,
        trails_config=trails_config,
        estimator=estimator,
        dataset=dataset,
        metrics=metrics,
        outputs=outputs,
        artifacts=artifacts,
    )
    if config.training.artifacts.save is not None:
        estimator.save(resolve_project_path(config.training.artifacts.save, project_root))

    save_prediction_payload(outputs.predictions, prediction)
    save_patient_clusters_csv(
        outputs.patient_clusters,
        payload=prediction,
        patient_summaries=imported.patient_summaries,
    )
    cluster_rows = cluster_summary_rows(prediction, n_clusters=trails_config.model.n_clusters)
    feature_rows = cluster_feature_summary_rows(
        dataset,
        prediction,
        n_clusters=trails_config.model.n_clusters,
    )
    save_cluster_summary_csv(outputs.cluster_summary, cluster_rows)
    save_cluster_feature_summary_csv(outputs.cluster_feature_summary, feature_rows)

    summary = {
        "case": config.case.model_dump(mode="json"),
        "command": "case",
        "config": config.model_dump(mode="json"),
        "data": dataset_summary,
        "hydra_run_dir": str(hydra_run_dir),
        "metrics": json_safe_metrics(metrics),
        "outputs": output_payload(outputs),
        "seed": seed,
        "training": {
            "history": estimator.history,
            "n_clusters": trails_config.model.n_clusters,
            "trails_config": trails_config.model_dump(mode="json"),
        },
    }
    save_json(outputs.summary, summary)
    LOGGER.info(
        "Completed case run: patients=%s features=%s k=%s prediction=%s",
        len(dataset),
        dataset.n_features,
        trails_config.model.n_clusters,
        compact_log_path(outputs.patient_clusters),
    )
    return summary


def save_case_training_artifacts(
    *,
    config: CaseApplicationConfig,
    trails_config: TrailsConfig,
    estimator: TrailsEstimator,
    dataset: ClinicalTimeSeriesDataset,
    metrics: Mapping[str, float],
    outputs: CaseOutputPaths,
    artifacts: frozenset[str],
) -> None:
    should_save_diagnostics = config.training.diagnostics.latent_embeddings.enabled
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


def case_output_paths(config: CaseApplicationConfig, hydra_run_dir: Path) -> CaseOutputPaths:
    return CaseOutputPaths(
        dataset=resolve_output_path(config.case.outputs.dataset, hydra_run_dir),
        dataset_summary=resolve_output_path(config.case.outputs.dataset_summary, hydra_run_dir),
        predictions=resolve_output_path(config.case.outputs.predictions, hydra_run_dir),
        patient_clusters=resolve_output_path(config.case.outputs.patient_clusters, hydra_run_dir),
        cluster_summary=resolve_output_path(config.case.outputs.cluster_summary, hydra_run_dir),
        cluster_feature_summary=resolve_output_path(
            config.case.outputs.cluster_feature_summary,
            hydra_run_dir,
        ),
        summary=resolve_output_path(config.case.outputs.summary, hydra_run_dir),
    )


def output_payload(outputs: CaseOutputPaths) -> dict[str, str]:
    return {
        "case_summary": str(outputs.summary),
        "cluster_feature_summary": str(outputs.cluster_feature_summary),
        "cluster_summary": str(outputs.cluster_summary),
        "dataset": str(outputs.dataset),
        "dataset_summary": str(outputs.dataset_summary),
        "patient_clusters": str(outputs.patient_clusters),
        "predictions": str(outputs.predictions),
    }


def case_run_config(
    *,
    app_config: CaseApplicationConfig,
    trails_config: TrailsConfig,
    outputs: CaseOutputPaths,
    artifacts: frozenset[str],
    created_at: datetime,
) -> dict[str, Any]:
    return {
        "artifacts": sorted(artifacts),
        "case": app_config.case.model_dump(mode="json"),
        "config": trails_config.model_dump(mode="json"),
        "created_at": created_at.isoformat(timespec="seconds"),
        "diagnostics": app_config.training.diagnostics.model_dump(mode="json"),
        "outputs": output_payload(outputs),
        "swanlab": app_config.training.swanlab.model_dump(mode="json"),
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


def start_swanlab_run(
    swanlab_config: SwanLabConfig,
    trails_config: TrailsConfig,
    app_config: CaseApplicationConfig,
    outputs: CaseOutputPaths,
    artifacts: frozenset[str],
    diagnostics_config: DiagnosticsConfig,
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
            "case": app_config.case.model_dump(mode="json"),
            "config": trails_config.model_dump(mode="json"),
            "diagnostics": diagnostics_config.model_dump(mode="json"),
            "outputs": output_payload(outputs),
            "save_artifacts": sorted(artifacts),
            "swanlab": swanlab_config.model_dump(mode="json"),
        },
    }
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


def log_swanlab_case_metrics(metrics: Mapping[str, float], history: list[HistoryEntry]) -> None:
    step = int(float(history[-1]["global_epoch"])) if history else 0
    swanlab.log({f"case/{name}": value for name, value in metrics.items()}, step=step)


def configure_torch_threads(torch_threads: int | None) -> None:
    if torch_threads is None:
        return
    torch.set_num_threads(torch_threads)


def resolve_project_path(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def resolve_output_path(path: Path, hydra_run_dir: Path) -> Path:
    return path if path.is_absolute() else hydra_run_dir / path


def compact_log_path(path: Path, *, keep_parts: int = 4) -> str:
    parts = path.parts
    if len(parts) <= keep_parts:
        return str(path)
    return str(Path("...").joinpath(*parts[-keep_parts:]))

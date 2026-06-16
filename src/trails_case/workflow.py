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
from trails.estimator import (
    KSelectionMetrics,
    TrailsEstimator,
    best_selection_metrics,
    selected_k_from_selection_metrics,
)
from trails.trainer import HistoryEntry

from .config import CaseApplicationConfig, DiagnosticsConfig, SwanLabConfig
from .data import case_dataset_summary, patient_summaries_from_metadata
from .evaluation import (
    CaseResultTables,
    evaluate_case_predictions,
    json_safe_metrics,
    prediction_payload_from_case_dataset,
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

    @classmethod
    def from_config(
        cls,
        config: CaseApplicationConfig,
        hydra_run_dir: Path,
    ) -> CaseOutputPaths:
        return cls(
            dataset=resolve_path(config.case.outputs.dataset, hydra_run_dir),
            dataset_summary=resolve_path(config.case.outputs.dataset_summary, hydra_run_dir),
            predictions=resolve_path(config.case.outputs.predictions, hydra_run_dir),
            patient_clusters=resolve_path(config.case.outputs.patient_clusters, hydra_run_dir),
            cluster_summary=resolve_path(config.case.outputs.cluster_summary, hydra_run_dir),
            cluster_feature_summary=resolve_path(
                config.case.outputs.cluster_feature_summary,
                hydra_run_dir,
            ),
            summary=resolve_path(config.case.outputs.summary, hydra_run_dir),
        )


def configure_torch_threads(torch_threads: int | None) -> None:
    if torch_threads is None:
        return
    torch.set_num_threads(torch_threads)


def resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def run_case_command(
    config: CaseApplicationConfig,
    hydra_run_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    # --------------------------------------------------------------
    # 1. config and output paths
    # --------------------------------------------------------------
    configure_torch_threads(config.training.parallel.torch_threads)
    hydra_run_dir.mkdir(parents=True, exist_ok=True)
    outputs = CaseOutputPaths.from_config(config, hydra_run_dir)

    # --------------------------------------------------------------
    # 2. load dataset from case CSVs
    # --------------------------------------------------------------
    patient_columns = config.case.columns.patients
    observation_columns = config.case.columns.observations
    dataset = ClinicalTimeSeriesDataset.load_from_csv(
        patients_csv=resolve_path(config.case.patients_csv, project_root),
        observations_csv=resolve_path(config.case.observations_csv, project_root),
        patient_id_col=patient_columns.patient_id,
        survival_time_col=patient_columns.survival_time,
        event_col=patient_columns.event,
        cluster_label_col=patient_columns.cluster_label,
        observation_id_col=observation_columns.patient_id,
        time_col=observation_columns.time,
        feature_col=observation_columns.feature,
        value_col=observation_columns.value,
        use_features=config.case.feature_order,
        description=config.case.description,
        metadata={
            "case_columns": config.case.columns.model_dump(mode="json"),
            "source": "case_csv",
        },
    )
    dataset.save(outputs.dataset)
    dataset_summary = case_dataset_summary(dataset)
    save_json(outputs.dataset_summary, dataset_summary)
    patient_summaries = patient_summaries_from_metadata(dataset.metadata)

    # --------------------------------------------------------------
    # 3. configure TRAILS
    # --------------------------------------------------------------
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
    k_selection_metrics: KSelectionMetrics | None = None
    k_selection_result_dir: Path | None = None
    estimator = TrailsEstimator(trails_config)
    if config.case.k_selection.enabled:
        k_selection_result_dir = resolve_path(config.case.k_selection.result_dir, hydra_run_dir)
        k_selection_metrics = estimator.select_n_clusters(
            dataset,
            candidate_clusters=case_k_selection_candidates(config),
            valid_fraction=case_k_selection_valid_fraction(config),
            inherit_best=True,
            result_dir=k_selection_result_dir,
        )
        trails_config = estimator.config
        selected_k = selected_k_from_selection_metrics(k_selection_metrics)
        best_metrics = best_selection_metrics(k_selection_metrics)
        LOGGER.info(
            "Selected case K=%s score=%.4g cindex=%.4g bic=%.4g",
            selected_k,
            best_metrics["selection_score"],
            best_metrics["cindex"],
            best_metrics["bic"],
        )

    # --------------------------------------------------------------
    # 4. train and predict
    # --------------------------------------------------------------
    start_swanlab_run(
        config.training.swanlab,
        trails_config,
        config,
        outputs,
        artifacts,
        config.training.diagnostics,
        k_selection_metrics=k_selection_metrics,
        k_selection_result_dir=k_selection_result_dir,
    )
    try:
        if config.case.k_selection.enabled:
            if config.training.swanlab.enabled:
                for entry in estimator.history:
                    log_swanlab_history(entry)
        else:
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
            log_swanlab_case_metrics(metrics, estimator.history, k_selection_metrics)
    finally:
        if config.training.swanlab.enabled:
            swanlab.finish()

    # --------------------------------------------------------------
    # 5. save artifacts and prediction tables
    # --------------------------------------------------------------
    save_case_training_artifacts(
        config=config,
        trails_config=trails_config,
        estimator=estimator,
        dataset=dataset,
        metrics=metrics,
        outputs=outputs,
        artifacts=artifacts,
        k_selection_metrics=k_selection_metrics,
        k_selection_result_dir=k_selection_result_dir,
    )
    if config.training.artifacts.save is not None:
        estimator.save(resolve_path(config.training.artifacts.save, project_root))

    save_prediction_payload(outputs.predictions, prediction)
    tables = CaseResultTables(prediction)
    tables.save_patient_clusters_csv(
        outputs.patient_clusters,
        patient_summaries=patient_summaries,
    )
    tables.save_cluster_summary_csv(
        outputs.cluster_summary, n_clusters=trails_config.model.n_clusters
    )
    tables.save_cluster_feature_summary_csv(
        outputs.cluster_feature_summary,
        dataset,
        n_clusters=trails_config.model.n_clusters,
    )

    # --------------------------------------------------------------
    # 6. summarize
    # --------------------------------------------------------------
    summary = {
        "case": config.case.model_dump(mode="json"),
        "command": "case",
        "config": config.model_dump(mode="json"),
        "data": dataset_summary,
        "hydra_run_dir": str(hydra_run_dir),
        "metrics": json_safe_metrics(metrics),
        "outputs": output_payload(outputs, k_selection_result_dir),
        "seed": seed,
        "training": {
            "history": estimator.history,
            "n_clusters": trails_config.model.n_clusters,
            "trails_config": trails_config.model_dump(mode="json"),
        },
    }
    if k_selection_metrics is not None:
        summary["k_selection"] = k_selection_payload(k_selection_metrics, k_selection_result_dir)
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
    k_selection_metrics: KSelectionMetrics | None = None,
    k_selection_result_dir: Path | None = None,
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
        "case": app_config.case.model_dump(mode="json"),
        "config": trails_config.model_dump(mode="json"),
        "created_at": created_at.isoformat(timespec="seconds"),
        "diagnostics": app_config.training.diagnostics.model_dump(mode="json"),
        "outputs": output_payload(outputs, k_selection_result_dir),
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
    if k_selection_metrics is not None:
        payload["k_selection"] = k_selection_payload(k_selection_metrics, k_selection_result_dir)
    return payload


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
            "case": app_config.case.model_dump(mode="json"),
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


def case_k_selection_candidates(config: CaseApplicationConfig) -> tuple[int, ...]:
    if config.case.k_selection.candidate_clusters:
        return config.case.k_selection.candidate_clusters
    return tuple(range(2, config.training.model.n_clusters + 1))


def case_k_selection_valid_fraction(config: CaseApplicationConfig) -> float:
    valid_fraction = (
        config.training.trainer.valid_size
        if config.case.k_selection.valid_size is None
        else config.case.k_selection.valid_size
    )
    if valid_fraction <= 0.0 or valid_fraction >= 1.0:
        raise ValueError(
            "case K selection requires case.k_selection.valid_size or "
            "training.trainer.valid_size to be greater than 0 and less than 1."
        )
    return valid_fraction


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


def swanlab_k_selection_metrics(metrics: KSelectionMetrics) -> dict[str, float | int]:
    best = best_selection_metrics(metrics)
    return {
        "selection/selected_k": selected_k_from_selection_metrics(metrics),
        "selection/best_cindex": float(best["cindex"]),
        "selection/best_bic": float(best["bic"]),
        "selection/best_score": float(best["selection_score"]),
    }


def compact_log_path(path: Path, *, keep_parts: int = 4) -> str:
    parts = path.parts
    if len(parts) <= keep_parts:
        return str(path)
    return str(Path("...").joinpath(*parts[-keep_parts:]))

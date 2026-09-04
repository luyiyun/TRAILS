from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import swanlab
import torch
from omegaconf import DictConfig

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
from trails.progress import configure_tqdm_logging
from trails.selection import ClusterNumberSelectionResult, ClusterNumberSelector
from trails.trainer import HistoryEntry
from trails_case.config import CaseApplicationConfig
from trails_case.evaluation import (
    CasePatientSummary,
    CaseResultTables,
    evaluate_case_predictions,
    json_safe_metrics,
    prediction_payload_from_case_dataset,
    save_prediction_payload,
)
from trails_case.selection import case_k_selection_candidates, case_k_selection_valid_fraction
from trails_simulate.config import resolved_payload

LOGGER = logging.getLogger(__name__)


def resolve_input_path(path: Path, base_dir: Path | None = None) -> Path:
    return path if path.is_absolute() else (base_dir or Path.cwd()) / path


def resolve_output_path(path: Path, run_dir: Path) -> Path:
    return path if path.is_absolute() else run_dir / path


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


def k_selection_payload(
    result: ClusterNumberSelectionResult,
    result_dir: Path | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = result.to_payload()
    if result_dir is not None:
        payload["result_dir"] = str(result_dir)
    return payload


def format_case_summary(result: Mapping[str, Any]) -> str:
    outputs = dict(result["outputs"])
    data = dict(result["data"])
    metrics = dict(result["metrics"])
    lines = [
        "TRAILS case complete",
        f"Run dir: {result['run_dir']}",
        f"Patients: {data['n_patients']}",
        f"Features: {data['n_features']}",
        f"Observations: {data['n_observations']}",
        "",
        "Saved outputs:",
        f"  summary: {outputs['case_summary']}",
        f"  patient clusters: {outputs['patient_clusters']}",
        f"  cluster summary: {outputs['cluster_summary']}",
        f"  feature summary: {outputs['cluster_feature_summary']}",
        f"  model predictions: {outputs['predictions']}",
        f"  dataset: {outputs['dataset']}",
    ]
    metric_parts: list[str] = []
    for name in (
        "cindex",
        "acc",
        "ari",
        "nmi",
        "cluster_empty_count",
        "cluster_min_fraction",
        "cluster_max_fraction",
        "cluster_entropy",
    ):
        value = metrics.get(name)
        if not isinstance(value, int | float):
            continue
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            formatted = str(number)
        elif abs(number) >= 1000 or 0 < abs(number) < 0.001:
            formatted = f"{number:.4e}"
        else:
            formatted = f"{number:.4f}"
        metric_parts.append(f"{name}={formatted}")
    if metric_parts:
        lines.extend(["", f"Metrics: {', '.join(metric_parts)}"])
    return "\n".join(lines)


def log_swanlab_history(entry: HistoryEntry) -> None:
    metrics = {
        "epoch/global": entry["global_epoch"],
        "epoch/local": entry["epoch"],
        **{f"train/{key}": value for key, value in entry["train"].items()},
    }
    if "valid" in entry:
        metrics.update({f"val/{key}": value for key, value in entry["valid"].items()})
    swanlab.log(metrics, step=entry["global_epoch"])


def run(config: CaseApplicationConfig) -> dict[str, Any]:
    # 1. config and output paths
    if config.parallel.torch_threads is not None:
        torch.set_num_threads(config.parallel.torch_threads)
    run_dir = config.paths.dir
    run_dir.mkdir(parents=True, exist_ok=True)
    outputs = CaseOutputPaths.from_config(config)

    # 2. load dataset from case CSVs
    patient_columns = config.columns.patients
    observation_columns = config.columns.observations
    dataset = ClinicalTimeSeriesDataset.load_from_csv(
        patients_csv=resolve_input_path(config.patients_csv),
        observations_csv=resolve_input_path(config.observations_csv),
        patient_id_col=patient_columns.patient_id,
        survival_time_col=patient_columns.survival_time,
        event_col=patient_columns.event,
        cluster_label_col=patient_columns.cluster_label,
        observation_id_col=observation_columns.patient_id,
        time_col=observation_columns.time,
        feature_col=observation_columns.feature,
        value_col=observation_columns.value,
        use_features=config.feature_order,
        description=config.description,
        metadata={
            "case_columns": config.columns.model_dump(mode="json"),
            "source": "case_csv",
        },
    )
    dataset.save(outputs.dataset)

    raw_summaries = dataset.metadata.get("patient_summaries", [])
    if not isinstance(raw_summaries, Sequence) or isinstance(raw_summaries, str | bytes):
        raise ValueError("dataset metadata patient_summaries must be a sequence.")
    patient_summaries: list[CasePatientSummary] = []
    for raw_summary in raw_summaries:
        if not isinstance(raw_summary, Mapping):
            raise ValueError("dataset metadata patient_summaries entries must be mappings.")
        patient_summaries.append(
            CasePatientSummary(
                patient_id=str(raw_summary["patient_id"]),
                sample_index=int(raw_summary["sample_index"]),
                n_observations=int(raw_summary["n_observations"]),
                n_visits=int(raw_summary["n_visits"]),
                first_time=float(raw_summary["first_time"]),
                last_time=float(raw_summary["last_time"]),
                missing_fraction=float(raw_summary["missing_fraction"]),
            )
        )

    event_count = sum(float(sample.event) for sample in dataset)
    feature_observation_counts = {
        feature: int(
            sum(float(sample.to_aligned().mask[:, index].sum()) for sample in dataset.samples)
        )
        for index, feature in enumerate(dataset.feature_names)
    }
    dataset_summary = {
        "censoring_rate": 1.0 - event_count / len(dataset),
        "description": dataset.description,
        "event_rate": event_count / len(dataset),
        "feature_observation_counts": feature_observation_counts,
        "features": dataset.feature_names,
        "has_cluster_labels": dataset.has_cluster_labels,
        "n_features": dataset.n_features,
        "n_observations": int(sum(summary.n_observations for summary in patient_summaries)),
        "n_patients": len(dataset),
        "patient_summaries": [asdict(summary) for summary in patient_summaries],
        "source": {
            "observations_csv": dataset.metadata.get("observations_csv"),
            "patients_csv": dataset.metadata.get("patients_csv"),
        },
    }
    save_json(outputs.dataset_summary, dataset_summary)

    # 3. configure TRAILS
    seed = config.trainer.seed
    artifacts = resolve_artifact_names(config.artifacts.names)
    trails_config = TrailsConfig(
        data=DataConfig(n_features=dataset.n_features),
        model=config.model,
        trainer=config.trainer.model_copy(
            update={
                "batch_size": resolve_batch_size(len(dataset), config.trainer.batch_size),
                "seed": seed,
            }
        ),
        seed=seed,
    )
    k_selection_result: ClusterNumberSelectionResult | None = None
    k_selection_result_dir: Path | None = None
    estimator = TrailsEstimator(trails_config)
    if config.k_selection.enabled:
        selection_seeds = config.k_selection.seeds or (seed,)
        if seed not in selection_seeds:
            raise ValueError("trainer.seed must be included in k_selection.seeds.")
        k_selection_result_dir = resolve_output_path(
            config.k_selection.result_dir,
            run_dir,
        )
        selector = ClusterNumberSelector(
            case_k_selection_candidates(config),
            seeds=selection_seeds,
            valid_fraction=case_k_selection_valid_fraction(config),
            selection_rule=config.k_selection.selection_rule,
            require_non_empty=config.k_selection.require_non_empty,
            min_cluster_fraction=config.k_selection.min_cluster_fraction,
            min_mean_pairwise_ari=config.k_selection.min_mean_pairwise_ari,
            estimator_config=trails_config,
        )
        k_selection_result = selector.select(dataset)
        k_selection_result.save(k_selection_result_dir)
        if k_selection_result.selected_k is None:
            raise RuntimeError("No candidate K passed the configured selection gates.")
        estimator = k_selection_result.selected_estimators[seed]
        trails_config = estimator.config
        best_metrics = k_selection_result.run_metrics.loc[
            k_selection_result.run_metrics["n_clusters"] == k_selection_result.selected_k
        ].iloc[0]
        LOGGER.info(
            "Selected case K=%s score=%.4g cindex=%.4g bic=%.4g",
            k_selection_result.selected_k,
            best_metrics["selection_score"],
            best_metrics["cindex"],
            best_metrics["bic"],
        )

    # 4. train and predict
    if config.swanlab.enabled:
        experiment_name = config.swanlab.experiment or datetime.now().astimezone().strftime(
            "trails-case-%Y%m%d-%H%M%S"
        )
        init_kwargs: dict[str, Any] = {
            "project": config.swanlab.project,
            "experiment_name": experiment_name,
            "config": {
                "case": config.model_dump(mode="json"),
                "config": trails_config.model_dump(mode="json"),
                "diagnostics": config.diagnostics.model_dump(mode="json"),
                "outputs": output_payload(outputs, k_selection_result_dir),
                "save_artifacts": sorted(artifacts),
                "swanlab": config.swanlab.model_dump(mode="json"),
            },
        }
        if k_selection_result is not None:
            init_kwargs["config"]["k_selection"] = k_selection_payload(
                k_selection_result,
                k_selection_result_dir,
            )
        if config.swanlab.mode is not None:
            init_kwargs["mode"] = config.swanlab.mode
        swanlab.init(**init_kwargs)

    try:
        if config.k_selection.enabled:
            if config.swanlab.enabled:
                for entry in estimator.history:
                    log_swanlab_history(entry)
        else:
            estimator.fit(
                dataset,
                history_callback=log_swanlab_history if config.swanlab.enabled else None,
            )
        model_prediction = estimator.predict(dataset)
        prediction = prediction_payload_from_case_dataset(
            dataset,
            patient_ids=list(dataset.metadata["patient_ids"]),
            pred_cluster=model_prediction.predict(),
            risk_score=model_prediction.risk_score(trails_config.trainer.risk_horizon),
            cluster_probabilities=model_prediction.predict_proba(),
        )
        metrics = evaluate_case_predictions(
            dataset,
            model_prediction,
            trails_config.trainer.risk_horizon,
        )
        if config.swanlab.enabled:
            step = int(float(estimator.history[-1]["global_epoch"])) if estimator.history else 0
            swanlab_metrics: dict[str, float | int] = {
                f"case/{name}": value for name, value in metrics.items()
            }
            if k_selection_result is not None:
                if k_selection_result.selected_k is None:
                    raise ValueError("Cannot log K selection metrics without a selected K.")
                selected_k = k_selection_result.selected_k
                selected = k_selection_result.run_metrics.loc[
                    k_selection_result.run_metrics["n_clusters"] == selected_k
                ]
                swanlab_metrics.update(
                    {
                        "selection/selected_k": selected_k,
                        "selection/best_cindex": float(selected["cindex"].mean()),
                        "selection/best_bic": float(selected["bic"].mean()),
                        "selection/best_score": float(selected["selection_score"].mean()),
                    }
                )
            swanlab.log(swanlab_metrics, step=step)
    finally:
        if config.swanlab.enabled:
            swanlab.finish()

    # 5. save artifacts and prediction tables
    should_save_diagnostics = config.diagnostics.latent_embeddings.enabled
    if artifacts or should_save_diagnostics:
        artifact_run_dir = outputs.summary.parent
        if "config" in artifacts:
            run_config: dict[str, Any] = {
                "artifacts": sorted(artifacts),
                "case": config.model_dump(mode="json"),
                "config": trails_config.model_dump(mode="json"),
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "diagnostics": config.diagnostics.model_dump(mode="json"),
                "outputs": output_payload(outputs, k_selection_result_dir),
                "swanlab": config.swanlab.model_dump(mode="json"),
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
                    "survival_head_hidden_layers": (
                        trails_config.model.survival_head_hidden_layers
                    ),
                    "warmup_epochs": trails_config.trainer.warmup_epochs,
                },
            }
            if k_selection_result is not None:
                run_config["k_selection"] = k_selection_payload(
                    k_selection_result,
                    k_selection_result_dir,
                )
            save_json(artifact_run_dir / "config.json", run_config)
        if "history" in artifacts:
            save_json(artifact_run_dir / "history.json", estimator.history)
            save_history_csv(artifact_run_dir / "history.csv", estimator.history)
        if "test" in artifacts:
            save_json(artifact_run_dir / "case_metrics.json", json_safe_metrics(metrics))
        if "model" in artifacts:
            estimator.save(artifact_run_dir / "model.pt")
        if "plot" in artifacts:
            plot_history(artifact_run_dir / "history.png", estimator.history)
        if should_save_diagnostics:
            diagnostics = model_prediction.latent_diagnostics()
            save_latent_embedding_artifacts(
                artifact_run_dir,
                "case",
                diagnostics,
                random_state=trails_config.seed,
            )
    if config.artifacts.save is not None:
        estimator.save(resolve_output_path(config.artifacts.save, run_dir))

    save_prediction_payload(outputs.predictions, prediction)
    tables = CaseResultTables(prediction)
    tables.save_patient_clusters_csv(
        outputs.patient_clusters,
        patient_summaries=patient_summaries,
    )
    tables.save_cluster_summary_csv(
        outputs.cluster_summary,
        n_clusters=trails_config.model.n_clusters,
    )
    tables.save_cluster_feature_summary_csv(
        outputs.cluster_feature_summary,
        dataset,
        n_clusters=trails_config.model.n_clusters,
    )

    # 6. summarize
    summary = {
        "case": config.model_dump(mode="json"),
        "command": "case",
        "config": config.model_dump(mode="json"),
        "data": dataset_summary,
        "run_dir": str(run_dir),
        "metrics": json_safe_metrics(metrics),
        "outputs": output_payload(outputs, k_selection_result_dir),
        "seed": seed,
        "training": {
            "history": estimator.history,
            "n_clusters": trails_config.model.n_clusters,
            "trails_config": trails_config.model_dump(mode="json"),
        },
    }
    if k_selection_result is not None:
        summary["k_selection"] = k_selection_payload(k_selection_result, k_selection_result_dir)
    save_json(outputs.summary, summary)
    path_parts = outputs.patient_clusters.parts
    prediction_log_path = (
        str(outputs.patient_clusters)
        if len(path_parts) <= 4
        else str(Path("...").joinpath(*path_parts[-4:]))
    )
    LOGGER.info(
        "Completed case run: patients=%s features=%s k=%s prediction=%s",
        len(dataset),
        dataset.n_features,
        trails_config.model.n_clusters,
        prediction_log_path,
    )
    return summary


@hydra.main(config_path="../configs", config_name="case", version_base="1.3")
def main(raw_config: DictConfig) -> None:
    configure_tqdm_logging()
    config = CaseApplicationConfig.model_validate(resolved_payload(raw_config))
    result = run(config)
    LOGGER.info(format_case_summary(result))


if __name__ == "__main__":
    main()

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra
import swanlab
from omegaconf import DictConfig

from trails.artifacts import resolve_artifact_names, save_json
from trails.config import DataConfig, TrailsConfig, resolve_batch_size
from trails.data import ClinicalTimeSeriesDataset
from trails.estimator import (
    KSelectionMetrics,
    TrailsEstimator,
    best_selection_metrics,
    selected_k_from_selection_metrics,
)
from trails.progress import configure_tqdm_logging
from trails_case.config import CaseApplicationConfig
from trails_case.data import case_dataset_summary, patient_summaries_from_metadata
from trails_case.evaluation import (
    CaseResultTables,
    evaluate_case_predictions,
    json_safe_metrics,
    prediction_payload_from_case_dataset,
    save_prediction_payload,
)
from trails_case.outputs import (
    CaseOutputPaths,
    compact_log_path,
    configure_torch_threads,
    k_selection_payload,
    output_payload,
    resolve_input_path,
    resolve_output_path,
    save_case_training_artifacts,
)
from trails_case.selection import case_k_selection_candidates, case_k_selection_valid_fraction
from trails_case.summary import format_case_summary
from trails_case.swanlab import (
    log_swanlab_case_metrics,
    log_swanlab_history,
    start_swanlab_run,
)
from trails_simulate.config import resolved_payload

LOGGER = logging.getLogger(__name__)


def run(config: CaseApplicationConfig) -> dict[str, Any]:
    # 1. config and output paths
    configure_torch_threads(config.parallel.torch_threads)
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
    dataset_summary = case_dataset_summary(dataset)
    save_json(outputs.dataset_summary, dataset_summary)
    patient_summaries = patient_summaries_from_metadata(dataset.metadata)

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
    k_selection_metrics: KSelectionMetrics | None = None
    k_selection_result_dir: Path | None = None
    estimator = TrailsEstimator(trails_config)
    if config.k_selection.enabled:
        k_selection_result_dir = resolve_output_path(
            config.k_selection.result_dir,
            run_dir,
        )
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

    # 4. train and predict
    start_swanlab_run(
        config.swanlab,
        trails_config,
        config,
        outputs,
        artifacts,
        config.diagnostics,
        k_selection_metrics=k_selection_metrics,
        k_selection_result_dir=k_selection_result_dir,
    )
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
        if config.swanlab.enabled:
            log_swanlab_case_metrics(metrics, estimator.history, k_selection_metrics)
    finally:
        if config.swanlab.enabled:
            swanlab.finish()

    # 5. save artifacts and prediction tables
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


@hydra.main(config_path="../configs", config_name="case", version_base="1.3")
def main(raw_config: DictConfig) -> None:
    configure_tqdm_logging()
    config = CaseApplicationConfig.model_validate(resolved_payload(raw_config))
    result = run(config)
    LOGGER.info(format_case_summary(result))


if __name__ == "__main__":
    main()

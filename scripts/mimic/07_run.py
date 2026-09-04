from __future__ import annotations

import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
import swanlab
import torch
from omegaconf import DictConfig

from trails import (
    ClinicalTimeSeriesDataset,
    ClusterNumberSelectionResult,
    ClusterNumberSelector,
    DataConfig,
    TrailsConfig,
    TrailsEstimator,
    resolve_batch_size,
)
from trails.artifacts import plot_history, resolve_artifact_names, save_history_csv, save_json
from trails_case.evaluation import (
    evaluate_case_predictions,
)
from trails_simulate.config import resolved_payload
from trails_simulate.training import log_swanlab_history

from .config import MimicApplicationConfig
from .data import BASELINE_COVARIATE_COLUMNS
from .paths import resolve_input_path, resolve_output_path


def _save_split_outputs(
    run_dir: Path,
    split_name: str,
    dataset: ClinicalTimeSeriesDataset,
    estimator: TrailsEstimator,
    risk_horizon: float,
) -> dict[str, float]:
    split_dir = run_dir / split_name
    patient_ids = list(dataset.metadata["patient_ids"])
    model_prediction = estimator.predict(dataset)
    pred_cluster = model_prediction.predict()
    probabilities = model_prediction.predict_proba()
    risk_score = model_prediction.risk_score(risk_horizon)
    survival_time = torch.stack([dataset[index].survival_time for index in range(len(dataset))])
    event = torch.stack([dataset[index].event for index in range(len(dataset))])
    metrics = evaluate_case_predictions(dataset, model_prediction, risk_horizon)

    dataset.save(split_dir / "dataset.pt")
    model_prediction.save(split_dir / "model_prediction.pt")
    frame = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "pred_cluster": pred_cluster.numpy(),
            "risk_score": risk_score.numpy(),
            "survival_time": survival_time.numpy(),
            "event": event.numpy(),
        }
    )
    raw_covariates = dataset.metadata.get("baseline_covariates")
    if not isinstance(raw_covariates, list):
        raise ValueError(f"{split_name} dataset缺少baseline_covariates")
    covariates = pd.DataFrame(raw_covariates)
    required_covariates = {"patient_id", *BASELINE_COVARIATE_COLUMNS}
    if missing := sorted(required_covariates - set(covariates.columns)):
        raise ValueError(f"{split_name} baseline_covariates缺少字段：{missing}")
    frame = frame.merge(
        covariates.loc[:, ["patient_id", *BASELINE_COVARIATE_COLUMNS]],
        on="patient_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    for index in range(probabilities.shape[1]):
        frame[f"cluster_probability_{index}"] = probabilities[:, index]
    for index in range(model_prediction.latent_representation.shape[1]):
        frame[f"latent_{index}"] = model_prediction.latent_representation[:, index]
    frame.to_csv(split_dir / "patient_outputs.csv", index=False)
    save_json(split_dir / "metrics.json", metrics)
    return metrics


def run(config: MimicApplicationConfig) -> dict[str, object]:
    run_dir = config.paths.dir
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = resolve_artifact_names(config.artifacts.names)
    split_dir = resolve_input_path(config.split.dir)
    dataset_paths = {
        name: split_dir / name / "dataset.pt" for name in ("train", "validation", "test")
    }
    preprocessing_parameters = split_dir / "preprocessing_parameters.csv"
    required = [*dataset_paths.values(), preprocessing_parameters]
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(f"缺少 MIMIC tensor dataset；请先运行06_split：{missing}")
    device = config.trainer.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("MIMIC TRAILS 分析要求可用 CUDA GPU")

    load_started = time.perf_counter()
    datasets = {
        name: ClinicalTimeSeriesDataset.load(dataset_paths[name])
        for name in ("train", "validation")
    }
    for name, dataset in datasets.items():
        if (
            dataset.metadata.get("split_name") != name
            or dataset.metadata.get("split_seed") != config.split.seed
        ):
            raise ValueError(f"{dataset_paths[name]} 的split元数据与当前配置不一致")
    train = datasets["train"]
    validation = datasets["validation"]
    load_seconds = time.perf_counter() - load_started
    initial_k = config.n_clusters or config.k_selection.candidate_clusters[0]
    trails_config = TrailsConfig(
        data=DataConfig(n_features=train.n_features),
        model=config.model.model_copy(update={"n_clusters": initial_k}),
        trainer=config.trainer.model_copy(
            update={
                "batch_size": resolve_batch_size(len(train), config.trainer.batch_size),
                "valid_size": 0.0,
            }
        ),
        seed=config.trainer.seed,
    )
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
        torch.cuda.reset_peak_memory_stats(torch.device(device))

    k_selection_result: ClusterNumberSelectionResult | None = None
    k_selection_result_dir: Path | None = None
    selection_seconds = 0.0
    estimator: TrailsEstimator
    if config.n_clusters is None:
        selection_started = time.perf_counter()
        k_selection_result_dir = resolve_output_path(config.k_selection.result_dir, run_dir)
        selector = ClusterNumberSelector(
            config.k_selection.candidate_clusters,
            seeds=config.k_selection.seeds,
            selection_rule=config.k_selection.selection_rule,
            require_non_empty=config.k_selection.require_non_empty,
            min_cluster_fraction=config.k_selection.min_cluster_fraction,
            min_mean_pairwise_ari=config.k_selection.min_mean_pairwise_ari,
            estimator_config=trails_config,
        )
        k_selection_result = selector.select(train, validation_data=validation)
        selection_seconds = time.perf_counter() - selection_started
        k_selection_result.save(k_selection_result_dir)
        if k_selection_result.selected_k is None:
            raise RuntimeError("没有候选K通过配置的选择门槛；test尚未评估")
        estimator = k_selection_result.selected_estimators[config.trainer.seed]
        trails_config = estimator.config
    else:
        estimator = TrailsEstimator(trails_config)

    resolved_application = config.model_copy(update={"model": trails_config.model})
    if config.swanlab.enabled:
        experiment_name = config.swanlab.experiment or datetime.now().astimezone().strftime(
            "trails-mimic-%Y%m%d-%H%M%S"
        )
        init_kwargs: dict[str, Any] = {
            "project": config.swanlab.project,
            "experiment_name": experiment_name,
            "config": {
                "application": resolved_application.model_dump(mode="json"),
                "training": trails_config.model_dump(mode="json"),
                "artifacts": sorted(artifacts),
            },
        }
        if config.swanlab.mode is not None:
            init_kwargs["mode"] = config.swanlab.mode
        swanlab.init(**init_kwargs)

    try:
        training_started = time.perf_counter()
        if k_selection_result is None:
            estimator.fit(
                train,
                validation_data=validation,
                history_callback=log_swanlab_history if config.swanlab.enabled else None,
            )
            training_seconds = time.perf_counter() - training_started
        else:
            if config.swanlab.enabled:
                for entry in estimator.history:
                    log_swanlab_history(entry)
            training_seconds = selection_seconds

        # K与最终模型锁定后才读取封存测试集，避免选择失败时接触test。
        test_load_started = time.perf_counter()
        test = ClinicalTimeSeriesDataset.load(dataset_paths["test"])
        if (
            test.metadata.get("split_name") != "test"
            or test.metadata.get("split_seed") != config.split.seed
        ):
            raise ValueError(f"{dataset_paths['test']} 的split元数据与当前配置不一致")
        datasets["test"] = test
        load_seconds += time.perf_counter() - test_load_started

        inference_started = time.perf_counter()
        split_metrics = {
            name: _save_split_outputs(
                run_dir,
                name,
                dataset,
                estimator,
                trails_config.trainer.risk_horizon,
            )
            for name, dataset in datasets.items()
        }
        inference_seconds = time.perf_counter() - inference_started
        if config.swanlab.enabled:
            step = int(estimator.history[-1]["global_epoch"]) if estimator.history else 0
            swanlab.log(
                {
                    f"{split_name}/{metric_name}": value
                    for split_name, metrics in split_metrics.items()
                    for metric_name, value in metrics.items()
                },
                step=step,
            )
    finally:
        if config.swanlab.enabled:
            swanlab.finish()

    estimator.save(run_dir / "model.pt")
    shutil.copy2(preprocessing_parameters, run_dir / "preprocessing_parameters.csv")
    save_history_csv(run_dir / "training_history.csv", estimator.history)
    save_json(run_dir / "training_history.json", estimator.history)
    if "plot" in artifacts:
        plot_history(run_dir / "training_history.png", estimator.history)
    peak_memory = (
        torch.cuda.max_memory_allocated(torch.device(device)) / 1024**3
        if device.startswith("cuda")
        else 0.0
    )
    summary = {
        "data": {
            "split_seed": config.split.seed,
            "split_dir": str(split_dir),
            "split_sizes": {name: len(data) for name, data in datasets.items()},
            "n_features": train.n_features,
        },
        "metrics": split_metrics,
        "resources": {
            "device": device,
            "batch_size": trails_config.trainer.batch_size,
            "load_seconds": load_seconds,
            "training_seconds": training_seconds,
            "inference_seconds": inference_seconds,
            "k_selection_seconds": selection_seconds,
            "peak_gpu_memory_gib": peak_memory,
        },
        "application": resolved_application.model_dump(mode="json"),
        "training": trails_config.model_dump(mode="json"),
        "k_resolution": {
            "mode": "automatic" if k_selection_result is not None else "fixed",
            "requested_n_clusters": config.n_clusters,
            "effective_n_clusters": trails_config.model.n_clusters,
            "selection_seeds": (
                list(config.k_selection.seeds) if k_selection_result is not None else []
            ),
            "final_model_seed": config.trainer.seed,
        },
        "k_selection": (
            None
            if k_selection_result is None
            else {
                **k_selection_result.to_payload(),
                "result_dir": str(k_selection_result_dir),
            }
        ),
    }
    save_json(resolve_output_path(config.outputs.summary, run_dir), summary)
    return summary


@hydra.main(config_path="../../configs", config_name="mimic/run", version_base="1.3")
def main(raw_config: DictConfig) -> None:
    config = MimicApplicationConfig.model_validate(resolved_payload(raw_config))
    run(config)


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import swanlab

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

from .config import (
    Command,
    DiagnosticsConfig,
    SwanLabConfig,
    TrainingApplicationConfig,
)
from .evaluation import PredictionPayload, evaluate_predictions, prediction_payload_from_dataset
from .path import TrainPaths


@dataclass(frozen=True)
class TrainResult:
    history: list[HistoryEntry]
    metrics: dict[str, float]
    prediction: PredictionPayload
    run_dir: Path | None


def fit_training_run(
    config: TrainingApplicationConfig,
    *,
    train_paths: TrainPaths,
    seed: int,
    swanlab_repeat_label: str | None,
) -> TrainResult:
    artifacts = resolve_artifact_names(config.artifacts.names)
    dataset = ClinicalTimeSeriesDataset.load(train_paths.data)
    test_dataset = (
        dataset
        if train_paths.test_data is None
        else ClinicalTimeSeriesDataset.load(train_paths.test_data)
    )
    trails_config = TrailsConfig(
        data=DataConfig(n_features=dataset.n_features),
        model=config.model,
        trainer=config.trainer.model_copy(
            update={
                "batch_size": resolve_batch_size(
                    len(dataset),
                    config.trainer.batch_size,
                ),
                "seed": seed,
            }
        ),
        seed=seed,
    )

    start_swanlab_run(
        config.swanlab,
        trails_config,
        train_paths,
        artifacts,
        config.diagnostics,
        repeat_label=swanlab_repeat_label,
    )
    try:
        estimator = TrailsEstimator(trails_config).fit(
            dataset,
            history_callback=log_swanlab_history if config.swanlab.enabled else None,
        )
        model_prediction = estimator.predict(test_dataset)
        prediction = prediction_payload_from_dataset(
            test_dataset,
            pred_cluster=model_prediction.predict(),
            risk_score=model_prediction.risk_score(),
            cluster_probabilities=model_prediction.predict_proba(),
        )
        metrics = evaluate_predictions(
            prediction,
            n_clusters=trails_config.model.n_clusters,
        )
        if config.swanlab.enabled:
            log_swanlab_test_metrics(metrics, estimator.history)
    finally:
        if config.swanlab.enabled:
            swanlab.finish()

    run_dir = save_training_artifacts(
        config=config,
        train_paths=train_paths,
        trails_config=trails_config,
        estimator=estimator,
        train_dataset=dataset,
        test_dataset=test_dataset,
        metrics=metrics,
        artifacts=artifacts,
    )

    if train_paths.save is not None:
        estimator.save(train_paths.save)

    return TrainResult(
        history=estimator.history,
        metrics=metrics,
        prediction=prediction,
        run_dir=run_dir,
    )


def save_training_artifacts(
    *,
    config: TrainingApplicationConfig,
    train_paths: TrainPaths,
    trails_config: TrailsConfig,
    estimator: TrailsEstimator,
    train_dataset: ClinicalTimeSeriesDataset,
    test_dataset: ClinicalTimeSeriesDataset,
    metrics: dict[str, float],
    artifacts: frozenset[str],
) -> Path | None:
    should_save_diagnostics = config.diagnostics.latent_embeddings.enabled
    if not artifacts and not should_save_diagnostics:
        return None

    created_at = datetime.now().astimezone()
    run_dir = train_paths.train_root
    run_dir.mkdir(parents=True, exist_ok=True)

    # artifacts.names 与 diagnostics 开关共同决定训练后落盘产物边界。
    if "config" in artifacts:
        save_json(
            run_dir / "config.json",
            training_run_config(
                app_config=config,
                trails_config=trails_config,
                train_paths=train_paths,
                artifacts=artifacts,
                created_at=created_at,
                run_dir=run_dir,
            ),
        )
    if "history" in artifacts:
        save_json(run_dir / "history.json", estimator.history)
        save_history_csv(run_dir / "history.csv", estimator.history)
    if "test" in artifacts:
        save_json(run_dir / "test_metrics.json", metrics)
    if "model" in artifacts:
        estimator.save(run_dir / "model.pt")
    if "plot" in artifacts:
        plot_history(run_dir / "history.png", estimator.history)
    if should_save_diagnostics:
        save_latent_embedding_diagnostics(
            run_dir=run_dir,
            estimator=estimator,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            seed=trails_config.seed,
        )

    return run_dir


def save_latent_embedding_diagnostics(
    *,
    run_dir: Path,
    estimator: TrailsEstimator,
    train_dataset: ClinicalTimeSeriesDataset,
    test_dataset: ClinicalTimeSeriesDataset,
    seed: int,
) -> None:
    split_datasets: list[tuple[str, ClinicalTimeSeriesDataset]] = [("train", train_dataset)]
    split_datasets.append(("test", test_dataset))

    for split_name, split_dataset in split_datasets:
        diagnostics = estimator.predict(split_dataset).latent_diagnostics()
        save_latent_embedding_artifacts(
            run_dir,
            split_name,
            diagnostics,
            random_state=seed,
        )


def start_swanlab_run(
    swanlab_config: SwanLabConfig,
    trails_config: TrailsConfig,
    train_paths: TrainPaths,
    artifacts: frozenset[str],
    diagnostics_config: DiagnosticsConfig,
    *,
    repeat_label: str | None,
) -> None:
    if not swanlab_config.enabled:
        return

    # 每次训练单独打开/关闭 SwanLab run，repeat 标签避免多轮实验同名。
    experiment_name = swanlab_config.experiment or datetime.now().astimezone().strftime(
        "trails-%Y%m%d-%H%M%S"
    )
    if repeat_label is not None:
        experiment_name = f"{experiment_name}-{repeat_label}"

    init_kwargs: dict[str, Any] = {
        "project": swanlab_config.project,
        "experiment_name": experiment_name,
        "config": swanlab_config_payload(
            swanlab_config,
            trails_config,
            train_paths,
            artifacts,
            diagnostics_config,
        ),
    }
    if swanlab_config.mode is not None:
        init_kwargs["mode"] = swanlab_config.mode
    swanlab.init(**init_kwargs)


def log_swanlab_history(entry: HistoryEntry) -> None:
    metrics = {
        "epoch/global": entry["global_epoch"],
        "epoch/local": entry["epoch"],
        **{f"train/{k}": v for k, v in entry["train"].items()},
    }
    if "valid" in entry:
        metrics.update({f"val/{k}": v for k, v in entry["valid"].items()})
    step = entry["global_epoch"]
    swanlab.log(metrics, step=step)


def log_swanlab_test_metrics(metrics: dict[str, float], history: list[HistoryEntry]) -> None:
    step = int(float(history[-1]["global_epoch"])) if history else 0
    swanlab.log({f"test/{name}": value for name, value in metrics.items()}, step=step)


def swanlab_config_payload(
    swanlab_config: SwanLabConfig,
    trails_config: TrailsConfig,
    train_paths: TrainPaths,
    artifacts: frozenset[str],
    diagnostics_config: DiagnosticsConfig,
) -> dict[str, Any]:
    return {
        "config": trails_config.model_dump(mode="json"),
        "diagnostics": diagnostics_config.model_dump(mode="json"),
        "paths": train_paths_payload(train_paths),
        "save_artifacts": sorted(artifacts),
        "swanlab": swanlab_config.model_dump(mode="json"),
    }


def training_run_config(
    *,
    app_config: TrainingApplicationConfig,
    trails_config: TrailsConfig,
    train_paths: TrainPaths,
    artifacts: frozenset[str],
    created_at: datetime,
    run_dir: Path,
) -> dict[str, Any]:
    return {
        "artifacts": sorted(artifacts),
        "config": trails_config.model_dump(mode="json"),
        "created_at": created_at.isoformat(timespec="seconds"),
        "paths": {
            **train_paths_payload(train_paths),
            "run_dir": str(run_dir),
        },
        "diagnostics": app_config.diagnostics.model_dump(mode="json"),
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


def train_output_payload(
    *,
    command: Command,
    run_dir: Path,
    train_paths: TrainPaths,
    result: TrainResult,
    seed: int,
) -> dict[str, Any]:
    return {
        "command": command,
        "history": result.history,
        "run_dir": str(run_dir),
        "paths": train_paths_payload(train_paths),
        "seed": seed,
        "test": result.metrics,
        "trainer_run_dir": None if result.run_dir is None else str(result.run_dir),
    }


def train_paths_payload(train_paths: TrainPaths) -> dict[str, str | None]:
    return {
        "data": str(train_paths.data),
        "save": None if train_paths.save is None else str(train_paths.save),
        "save_dir": str(train_paths.train_root),
        "test_data": None if train_paths.test_data is None else str(train_paths.test_data),
    }


def summarize_repeat_metrics(repeats: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metric_names = sorted(
        {
            name
            for repeat in repeats
            for name, value in dict(repeat["metrics"]).items()
            if isinstance(value, int | float)
        }
    )
    summary: dict[str, Any] = {}
    for name in metric_names:
        values = [
            float(dict(repeat["metrics"])[name])
            for repeat in repeats
            if name in dict(repeat["metrics"])
            and isinstance(dict(repeat["metrics"])[name], int | float)
            and math.isfinite(float(dict(repeat["metrics"])[name]))
        ]
        if not values:
            continue
        array = np.asarray(values, dtype=float)
        summary[name] = {
            "max": float(array.max()),
            "mean": float(array.mean()),
            "min": float(array.min()),
            "n": len(values),
            "std": float(array.std(ddof=0)),
        }
    return summary


def save_repeat_metrics_csv(path: Path, repeats: Sequence[Mapping[str, Any]]) -> None:
    metric_names = sorted(
        {
            name
            for repeat in repeats
            for name, value in dict(repeat["metrics"]).items()
            if isinstance(value, int | float)
        }
    )
    fieldnames = ["repeat", "index", "seed", "data_dir", "train_run_dir", *metric_names]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for repeat in repeats:
            metrics = dict(repeat["metrics"])
            row = {name: repeat.get(name) for name in fieldnames}
            row.update({name: metrics.get(name, "") for name in metric_names})
            writer.writerow(row)

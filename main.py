from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import hydra
import swanlab
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from trails.artifacts import (
    create_timestamped_run_dir,
    plot_history,
    resolve_artifact_names,
    save_history_csv,
    save_json,
)
from trails.config import DataConfig, ModelConfig, TrailsConfig, TrainerConfig
from trails.data import ClinicalTimeSeriesDataset
from trails.estimator import TrailsEstimator
from trails.trainer import HistoryEntry
from trails_simulate import generate_clinical_time_series_dataset

Command = Literal["experiment", "simulate", "train"]
SplitNames = tuple[Literal["train"], Literal["val"], Literal["test"]]


# ---------------------------------------------------------------------------
# Config schema


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "quick"
    repeats: int = Field(default=1, gt=0)
    seed: int = 20260517
    seed_stride: int = Field(default=100, gt=0)


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_root: Path | None = None
    data: Path | None = None
    val_data: Path | None = None
    test_data: Path | None = None
    train_root: Path | None = None


class SimulatorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patients: int = Field(default=128, gt=0)
    split_patients: tuple[int, int, int] | None = None
    n_clusters: int = Field(default=3, gt=1)
    min_visits: int = Field(default=4, gt=0)
    max_visits: int = Field(default=8, gt=0)
    hidden_size: int = Field(default=100, gt=0)
    latent_dim: int = Field(default=5, gt=0)
    attention_layers: int = Field(default=3, gt=0)
    attention_heads: int | None = Field(default=None, gt=0)
    censoring_rate: float = Field(default=0.3, ge=0.0, le=1.0)
    weibull_shape: float = Field(default=1.0, gt=0.0)
    x_low: float = -10.0
    x_high: float = 10.0
    beta_low: float = -2.5
    beta_high: float = 2.5


class ArtifactsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    names: tuple[str, ...] = ("all",)
    save: Path | None = None


class SwanLabConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    project: str = "TRAILS"
    experiment: str | None = None
    mode: str | None = None


class ApplicationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: Command = "experiment"
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    simulator: SimulatorConfig = Field(default_factory=SimulatorConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig)
    swanlab: SwanLabConfig = Field(default_factory=SwanLabConfig)


@dataclass(frozen=True)
class TrainPaths:
    data: Path
    val_data: Path | None
    test_data: Path | None
    test_data_used: Path
    train_root: Path
    save: Path | None


@dataclass(frozen=True)
class TrainResult:
    history: list[HistoryEntry]
    metrics: dict[str, float]
    run_dir: Path | None


# ---------------------------------------------------------------------------
# Hydra entrypoint and command dispatch


def _load_app_config(raw_config: DictConfig) -> ApplicationConfig:
    payload = OmegaConf.to_container(raw_config, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError("Hydra config must resolve to a mapping.")
    try:
        return ApplicationConfig.model_validate(payload)
    except ValidationError as error:
        raise ValueError(str(error)) from error


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(raw_config: DictConfig) -> None:
    config = _load_app_config(raw_config)
    project_root = Path(get_original_cwd())
    hydra_run_dir = Path(HydraConfig.get().runtime.output_dir)
    result = run(config, hydra_run_dir=hydra_run_dir, project_root=project_root)
    print(format_run_summary(result))


def run(
    config: ApplicationConfig,
    *,
    hydra_run_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    if config.command == "simulate":
        return _run_simulate_command(config, hydra_run_dir, project_root)
    if config.command == "train":
        return _run_train_command(config, hydra_run_dir, project_root)
    if config.command == "experiment":
        return _run_experiment_command(config, hydra_run_dir, project_root)

    raise ValueError(f"Unsupported command: {config.command}")


# ---------------------------------------------------------------------------
# Command flows


def _run_simulate_command(
    config: ApplicationConfig,
    hydra_run_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    seed = config.experiment.seed

    if config.simulator.split_patients is not None:
        if config.paths.data is not None:
            raise ValueError("paths.data cannot be combined with simulator.split_patients.")
        data_root = _data_root(config, hydra_run_dir, project_root)
        payload = _simulate_splits(
            config.simulator,
            out_dir=data_root,
            seed=seed,
        )
        return {
            "command": "simulate",
            "hydra_run_dir": str(hydra_run_dir),
            **payload,
        }

    out = _single_dataset_path(config, hydra_run_dir, project_root)
    payload = _simulate_one_dataset(
        config.simulator,
        out=out,
        n_patients=config.simulator.patients,
        seed=seed,
    )
    return {
        "command": "simulate",
        "hydra_run_dir": str(hydra_run_dir),
        **payload,
    }


def _run_train_command(
    config: ApplicationConfig,
    hydra_run_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    seed = config.experiment.seed
    train_paths = _train_paths_from_config(config, hydra_run_dir, project_root)
    result = _fit_training_run(
        config,
        train_paths=train_paths,
        seed=seed,
        swanlab_repeat_label=None,
    )
    return _train_output_payload(
        command="train",
        hydra_run_dir=hydra_run_dir,
        train_paths=train_paths,
        result=result,
        seed=seed,
    )


def _run_experiment_command(
    config: ApplicationConfig,
    hydra_run_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    if config.simulator.split_patients is None:
        raise ValueError("command=experiment requires simulator.split_patients.")
    if (
        config.paths.data_root is not None
        or config.paths.data is not None
        or config.paths.val_data is not None
        or config.paths.test_data is not None
        or config.paths.train_root is not None
    ):
        raise ValueError("command=experiment generates its own split data per repeat.")

    repeats: list[dict[str, Any]] = []
    for index in range(config.experiment.repeats):
        # paired repeat: 同一个 repeat seed 同时驱动该轮 split 生成和模型训练。
        repeat_name = f"repeat_{index:03d}"
        repeat_seed = config.experiment.seed + index * config.experiment.seed_stride
        repeat_dir = hydra_run_dir / repeat_name
        data_dir = repeat_dir / "data"
        train_root = repeat_dir / "train"

        split_payload = _simulate_splits(
            config.simulator,
            out_dir=data_dir,
            seed=repeat_seed,
        )
        train_paths = TrainPaths(
            data=data_dir / "train.pt",
            val_data=data_dir / "val.pt",
            test_data=data_dir / "test.pt",
            test_data_used=data_dir / "test.pt",
            train_root=train_root,
            save=_repeat_checkpoint_path(config, repeat_dir, project_root, index),
        )
        train_result = _fit_training_run(
            config,
            train_paths=train_paths,
            seed=repeat_seed,
            swanlab_repeat_label=f"r{index:03d}",
        )
        repeats.append(
            {
                "data_dir": str(data_dir),
                "index": index,
                "metrics": train_result.metrics,
                "repeat": repeat_name,
                "seed": repeat_seed,
                "splits": split_payload["splits"],
                "train_run_dir": None
                if train_result.run_dir is None
                else str(train_result.run_dir),
            }
        )

    metrics_summary = _summarize_repeat_metrics(repeats)
    summary = {
        "command": "experiment",
        "config": config.model_dump(mode="json"),
        "experiment": config.experiment.model_dump(mode="json"),
        "hydra_run_dir": str(hydra_run_dir),
        "metrics_summary": metrics_summary,
        "repeats": repeats,
    }

    save_json(hydra_run_dir / "experiment_summary.json", summary)
    _save_repeat_metrics_csv(hydra_run_dir / "test_metrics.csv", repeats)
    save_json(hydra_run_dir / "test_metrics_summary.json", metrics_summary)
    return summary


# ---------------------------------------------------------------------------
# Simulation


def _simulate_splits(
    simulator: SimulatorConfig,
    *,
    out_dir: Path,
    seed: int,
) -> dict[str, Any]:
    split_names: SplitNames = ("train", "val", "test")
    if simulator.split_patients is None:
        raise ValueError("simulator.split_patients is required for split simulation.")
    out_dir.mkdir(parents=True, exist_ok=True)

    # split seed 固定为 repeat seed + 0/1/2，便于复现实验中的 train/val/test。
    summaries: dict[str, dict[str, Any]] = {}
    for offset, (name, patient_count) in enumerate(
        zip(split_names, simulator.split_patients, strict=True)
    ):
        split_seed = seed + offset
        path = out_dir / f"{name}.pt"
        dataset = _generate_simulated_dataset(
            simulator,
            n_patients=patient_count,
            seed=split_seed,
        )
        dataset.save(path)
        summaries[name] = _simulation_summary(
            dataset,
            clusters=simulator.n_clusters,
            out=path,
            seed=split_seed,
        )

    return {
        "out_dir": str(out_dir),
        "split_patients": {
            name: count for name, count in zip(split_names, simulator.split_patients, strict=True)
        },
        "splits": summaries,
    }


def _simulate_one_dataset(
    simulator: SimulatorConfig,
    *,
    out: Path,
    n_patients: int,
    seed: int,
) -> dict[str, Any]:
    dataset = _generate_simulated_dataset(
        simulator,
        n_patients=n_patients,
        seed=seed,
    )
    dataset.save(out)
    return _simulation_summary(
        dataset,
        clusters=simulator.n_clusters,
        out=out,
        seed=seed,
    )


def _generate_simulated_dataset(
    simulator: SimulatorConfig,
    *,
    n_patients: int,
    seed: int,
) -> ClinicalTimeSeriesDataset:
    return generate_clinical_time_series_dataset(
        n_patients=n_patients,
        n_clusters=simulator.n_clusters,
        min_visits=simulator.min_visits,
        max_visits=simulator.max_visits,
        latent_dim=simulator.latent_dim,
        hidden_size=simulator.hidden_size,
        attention_layers=simulator.attention_layers,
        attention_heads=simulator.attention_heads,
        censoring_rate=simulator.censoring_rate,
        weibull_shape=simulator.weibull_shape,
        x_low=simulator.x_low,
        x_high=simulator.x_high,
        beta_low=simulator.beta_low,
        beta_high=simulator.beta_high,
        seed=seed,
    )


def _simulation_summary(
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


# ---------------------------------------------------------------------------
# Training


def _fit_training_run(
    config: ApplicationConfig,
    *,
    train_paths: TrainPaths,
    seed: int,
    swanlab_repeat_label: str | None,
) -> TrainResult:
    artifacts = resolve_artifact_names(config.artifacts.names)
    dataset = ClinicalTimeSeriesDataset.load(train_paths.data)
    validation_dataset = (
        None
        if train_paths.val_data is None
        else ClinicalTimeSeriesDataset.load(train_paths.val_data)
    )
    test_dataset = (
        dataset
        if train_paths.test_data is None
        else ClinicalTimeSeriesDataset.load(train_paths.test_data)
    )
    trails_config = TrailsConfig(
        data=DataConfig(n_features=dataset.n_features),
        model=config.model,
        trainer=config.trainer.model_copy(update={"seed": seed}),
        seed=seed,
    )

    _start_swanlab_run(
        config.swanlab,
        trails_config,
        train_paths,
        artifacts,
        repeat_label=swanlab_repeat_label,
    )
    try:
        estimator = TrailsEstimator(trails_config).fit(
            dataset,
            validation_data=validation_dataset,
            history_callback=_log_swanlab_history if config.swanlab.enabled else None,
        )
        metrics = estimator.test(test_dataset)
        if config.swanlab.enabled:
            _log_swanlab_test_metrics(metrics, estimator.history)
    finally:
        if config.swanlab.enabled:
            swanlab.finish()

    run_dir = _save_training_artifacts(
        config=config,
        train_paths=train_paths,
        trails_config=trails_config,
        estimator=estimator,
        metrics=metrics,
        artifacts=artifacts,
    )

    if train_paths.save is not None:
        estimator.save(train_paths.save)

    return TrainResult(history=estimator.history, metrics=metrics, run_dir=run_dir)


def _save_training_artifacts(
    *,
    config: ApplicationConfig,
    train_paths: TrainPaths,
    trails_config: TrailsConfig,
    estimator: TrailsEstimator,
    metrics: dict[str, float],
    artifacts: frozenset[str],
) -> Path | None:
    if not artifacts:
        return None

    created_at = datetime.now().astimezone()
    run_dir = create_timestamped_run_dir(train_paths.train_root, created_at)

    # artifacts.names 决定训练产物边界；不在这里隐式保存额外文件。
    if "config" in artifacts:
        save_json(
            run_dir / "config.json",
            _training_run_config(
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

    return run_dir


# ---------------------------------------------------------------------------
# SwanLab integration


def _start_swanlab_run(
    swanlab_config: SwanLabConfig,
    trails_config: TrailsConfig,
    train_paths: TrainPaths,
    artifacts: frozenset[str],
    *,
    repeat_label: str | None,
) -> None:
    if not swanlab_config.enabled:
        return

    import swanlab

    # 每次训练单独打开/关闭 SwanLab run，repeat 标签避免多轮实验同名。
    experiment_name = swanlab_config.experiment or datetime.now().astimezone().strftime(
        "trails-%Y%m%d-%H%M%S"
    )
    if repeat_label is not None:
        experiment_name = f"{experiment_name}-{repeat_label}"

    init_kwargs: dict[str, Any] = {
        "project": swanlab_config.project,
        "experiment_name": experiment_name,
        "config": _swanlab_config(swanlab_config, trails_config, train_paths, artifacts),
    }
    if swanlab_config.mode is not None:
        init_kwargs["mode"] = swanlab_config.mode
    swanlab.init(**init_kwargs)


def _log_swanlab_history(entry: HistoryEntry) -> None:
    metrics = {
        "epoch/global": entry["global_epoch"],
        "epoch/local": entry["epoch"],
        **{f"train/{k}": v for k, v in entry["train"].items()},
    }
    if "valid" in entry:
        metrics.update({f"val/{k}": v for k, v in entry["valid"].items()})
    step = entry["global_epoch"]
    swanlab.log(metrics, step=step)


def _log_swanlab_test_metrics(metrics: dict[str, float], history: list[HistoryEntry]) -> None:
    step = int(float(history[-1]["global_epoch"])) if history else 0
    swanlab.log({f"test/{name}": value for name, value in metrics.items()}, step=step)


def _swanlab_config(
    swanlab_config: SwanLabConfig,
    trails_config: TrailsConfig,
    train_paths: TrainPaths,
    artifacts: frozenset[str],
) -> dict[str, Any]:
    return {
        "config": trails_config.model_dump(mode="json"),
        "paths": _train_paths_payload(train_paths),
        "save_artifacts": sorted(artifacts),
        "swanlab": swanlab_config.model_dump(mode="json"),
    }


def _training_run_config(
    *,
    app_config: ApplicationConfig,
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
            **_train_paths_payload(train_paths),
            "run_dir": str(run_dir),
        },
        "swanlab": app_config.swanlab.model_dump(mode="json"),
        "train_args": {
            "batch_size": trails_config.trainer.batch_size,
            "clusters": trails_config.model.n_clusters,
            "decoder_hidden_dim": trails_config.model.decoder_hidden_dim,
            "dropout": trails_config.model.dropout,
            "encoder_hidden_dim": trails_config.model.encoder_hidden_dim,
            "epochs": trails_config.trainer.max_epochs,
            "latent_dim": trails_config.model.latent_dim,
            "learning_rate": trails_config.trainer.learning_rate,
            "n_layers": trails_config.model.n_layers,
            "seed": trails_config.seed,
            "warmup_epochs": trails_config.trainer.warmup_epochs,
        },
    }


def _train_output_payload(
    *,
    command: Command,
    hydra_run_dir: Path,
    train_paths: TrainPaths,
    result: TrainResult,
    seed: int,
) -> dict[str, Any]:
    return {
        "command": command,
        "history": result.history,
        "hydra_run_dir": str(hydra_run_dir),
        "paths": _train_paths_payload(train_paths),
        "run_dir": None if result.run_dir is None else str(result.run_dir),
        "seed": seed,
        "test": result.metrics,
    }


def _train_paths_payload(train_paths: TrainPaths) -> dict[str, str | None]:
    return {
        "data": str(train_paths.data),
        "save": None if train_paths.save is None else str(train_paths.save),
        "save_dir": str(train_paths.train_root),
        "test_data": None if train_paths.test_data is None else str(train_paths.test_data),
        "test_data_used": str(train_paths.test_data_used),
        "val_data": None if train_paths.val_data is None else str(train_paths.val_data),
    }


# ---------------------------------------------------------------------------
# Path resolution


def _train_paths_from_config(
    config: ApplicationConfig,
    hydra_run_dir: Path,
    project_root: Path,
) -> TrainPaths:
    if config.paths.data is None and config.paths.data_root is None:
        raise ValueError("command=train requires paths.data=... or paths.data_root=...")

    data_root = (
        None
        if config.paths.data_root is None
        else _resolve_path(config.paths.data_root, project_root)
    )
    if config.paths.data is not None:
        data_path = _resolve_path(config.paths.data, project_root)
    else:
        if data_root is None:
            raise ValueError("paths.data_root is required when paths.data is not set.")
        data_path = data_root / "train.pt"

    val_data = _optional_split_path(config.paths.val_data, data_root, "val", project_root)
    test_data = _optional_split_path(config.paths.test_data, data_root, "test", project_root)
    train_root = _train_root(config, hydra_run_dir, project_root)
    save = None
    if config.artifacts.save is not None:
        save = _resolve_path(config.artifacts.save, project_root)
    return TrainPaths(
        data=data_path,
        val_data=val_data,
        test_data=test_data,
        test_data_used=data_path if test_data is None else test_data,
        train_root=train_root,
        save=save,
    )


def _optional_split_path(
    explicit_path: Path | None,
    data_root: Path | None,
    name: str,
    project_root: Path,
) -> Path | None:
    if explicit_path is not None:
        return _resolve_path(explicit_path, project_root)
    if data_root is None:
        return None
    candidate = data_root / f"{name}.pt"
    return candidate if candidate.exists() else None


def _data_root(
    config: ApplicationConfig,
    hydra_run_dir: Path,
    project_root: Path,
) -> Path:
    if config.paths.data_root is not None:
        return _resolve_path(config.paths.data_root, project_root)
    return hydra_run_dir / "data"


def _single_dataset_path(
    config: ApplicationConfig,
    hydra_run_dir: Path,
    project_root: Path,
) -> Path:
    if config.paths.data is not None:
        return _resolve_path(config.paths.data, project_root)
    return _data_root(config, hydra_run_dir, project_root) / "dataset.pt"


def _train_root(
    config: ApplicationConfig,
    hydra_run_dir: Path,
    project_root: Path,
) -> Path:
    if config.paths.train_root is not None:
        return _resolve_path(config.paths.train_root, project_root)
    return hydra_run_dir / "train"


def _repeat_checkpoint_path(
    config: ApplicationConfig,
    repeat_dir: Path,
    project_root: Path,
    index: int,
) -> Path | None:
    if config.artifacts.save is None:
        return None
    configured = _resolve_path(config.artifacts.save, project_root)
    if config.experiment.repeats == 1:
        return configured
    suffix = configured.suffix
    stem = configured.stem if suffix else configured.name
    return repeat_dir / "train" / f"{stem}-r{index:03d}{suffix}"


def _resolve_path(path: Path, project_root: Path) -> Path:
    if path.is_absolute():
        return path
    return project_root / path


# ---------------------------------------------------------------------------
# Repeat metric summaries


def _summarize_repeat_metrics(repeats: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
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
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        summary[name] = {
            "max": max(values),
            "mean": mean,
            "min": min(values),
            "n": len(values),
            "std": math.sqrt(variance),
        }
    return summary


def _save_repeat_metrics_csv(path: Path, repeats: Sequence[Mapping[str, Any]]) -> None:
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


# ---------------------------------------------------------------------------
# Human-readable output


def format_run_summary(result: Mapping[str, Any]) -> str:
    command = str(result["command"])
    if command == "simulate":
        return _format_simulate_summary(result)
    if command == "train":
        return _format_train_summary(result)
    if command == "experiment":
        return _format_experiment_summary(result)
    raise ValueError(f"Unsupported command summary: {command}")


def _format_simulate_summary(result: Mapping[str, Any]) -> str:
    lines = ["TRAILS simulate complete", f"Hydra run: {result['hydra_run_dir']}"]
    if "splits" in result:
        lines.append(f"Data root: {result['out_dir']}")
        lines.append("")
        lines.append("Splits:")
        for name in ("train", "val", "test"):
            split = dict(dict(result["splits"])[name])
            lines.append(
                "  "
                f"{name:<5} patients={split['n_patients']} "
                f"seed={split['seed']} "
                f"censoring={_format_float(split['censoring_rate'])} "
                f"path={split['out']}"
            )
    else:
        lines.extend(
            [
                f"Dataset: {result['out']}",
                f"Patients: {result['n_patients']}",
                f"Clusters: {result['clusters']}",
                f"Features: {result['n_features']}",
                f"Seed: {result['seed']}",
                f"Censoring rate: {_format_float(result['censoring_rate'])}",
            ]
        )
    return "\n".join(lines)


def _format_train_summary(result: Mapping[str, Any]) -> str:
    paths = dict(result["paths"])
    lines = [
        "TRAILS train complete",
        f"Hydra run: {result['hydra_run_dir']}",
        f"Seed: {result['seed']}",
        f"Train data: {paths['data']}",
    ]
    if paths.get("val_data") is not None:
        lines.append(f"Validation data: {paths['val_data']}")
    lines.append(f"Test data: {paths['test_data_used']}")
    lines.append(f"Artifacts: {result['run_dir'] or 'not saved'}")
    lines.extend(_format_metrics_block("Test metrics", dict(result["test"])))
    return "\n".join(lines)


def _format_experiment_summary(result: Mapping[str, Any]) -> str:
    run_dir = Path(str(result["hydra_run_dir"]))
    repeats = [dict(repeat) for repeat in result["repeats"]]
    lines = [
        "TRAILS experiment complete",
        f"Hydra run: {run_dir}",
        f"Repeats: {len(repeats)}",
        f"Seeds: {_format_seed_list([int(repeat['seed']) for repeat in repeats])}",
        "",
        "Saved summaries:",
        f"  experiment: {run_dir / 'experiment_summary.json'}",
        f"  metrics csv: {run_dir / 'test_metrics.csv'}",
        f"  metrics summary: {run_dir / 'test_metrics_summary.json'}",
    ]
    lines.extend(_format_metric_summary_block(dict(result["metrics_summary"])))
    lines.extend(_format_repeat_block(repeats))
    return "\n".join(lines)


def _format_metrics_block(title: str, metrics: Mapping[str, Any]) -> list[str]:
    names = _ordered_metric_names(metrics.keys())
    if not names:
        return []
    lines = ["", f"{title}:"]
    for name in names:
        value = metrics[name]
        if isinstance(value, int | float):
            lines.append(f"  {name:<22} {_format_float(value)}")
    return lines


def _format_metric_summary_block(summary: Mapping[str, Any]) -> list[str]:
    names = _ordered_metric_names(summary.keys())
    if not names:
        return []
    lines = ["", "Metric summary:"]
    header = f"  {'metric':<22} {'mean':>10} {'std':>10} {'min':>10} {'max':>10} {'n':>4}"
    lines.append(header)
    for name in names:
        stats = dict(summary[name])
        lines.append(
            "  "
            f"{name:<22} "
            f"{_format_float(stats['mean']):>10} "
            f"{_format_float(stats['std']):>10} "
            f"{_format_float(stats['min']):>10} "
            f"{_format_float(stats['max']):>10} "
            f"{int(stats['n']):>4}"
        )
    return lines


def _format_repeat_block(repeats: Sequence[Mapping[str, Any]]) -> list[str]:
    if not repeats:
        return []
    lines = ["", "Repeat results:"]
    for repeat in repeats:
        metrics = dict(repeat["metrics"])
        metric_text = ", ".join(
            f"{name}={_format_float(metrics[name])}"
            for name in _ordered_metric_names(metrics.keys())[:4]
            if isinstance(metrics.get(name), int | float)
        )
        lines.append(
            "  "
            f"{repeat['repeat']} seed={repeat['seed']} "
            f"{metric_text} "
            f"artifacts={repeat['train_run_dir'] or 'not saved'}"
        )
    return lines


def _ordered_metric_names(names: Iterable[str]) -> list[str]:
    preferred = [
        "loss",
        "c_index",
        "ari",
        "nmi",
        "reconstruction_loss",
        "survival_loss",
        "vade_kl_loss",
    ]
    available = set(names)
    ordered = [name for name in preferred if name in available]
    ordered.extend(sorted(available - set(ordered)))
    return ordered


def _format_seed_list(seeds: Sequence[int]) -> str:
    if len(seeds) <= 8:
        return ", ".join(str(seed) for seed in seeds)
    head = ", ".join(str(seed) for seed in seeds[:4])
    tail = ", ".join(str(seed) for seed in seeds[-2:])
    return f"{head}, ..., {tail}"


def _format_float(value: Any) -> str:
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return str(number)
    if abs(number) >= 1000 or (0 < abs(number) < 0.001):
        return f"{number:.4e}"
    return f"{number:.4f}"


if __name__ == "__main__":
    main()

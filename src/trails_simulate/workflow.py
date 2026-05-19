from __future__ import annotations

from pathlib import Path
from typing import Any

from trails.artifacts import save_json
from trails.data import ClinicalTimeSeriesDataset

from .config import ApplicationConfig
from .generators import ClinicalTimeSeriesDatasetGenerator
from .optim import run_optim_command
from .path import (
    TrainPaths,
    data_root,
    repeat_checkpoint_path,
    resolve_path,
    train_paths_from_config,
)
from .training import (
    fit_training_run,
    save_repeat_metrics_csv,
    summarize_repeat_metrics,
    train_output_payload,
)


def simulation_summary(
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


def run(
    config: ApplicationConfig,
    *,
    hydra_run_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    if config.command == "simulate":
        return run_simulate_command(config, hydra_run_dir, project_root)
    if config.command == "train":
        return run_train_command(config, hydra_run_dir, project_root)
    if config.command == "experiment":
        return run_experiment_command(config, hydra_run_dir, project_root)
    if config.command == "optim":
        return run_optim_command(config, hydra_run_dir, project_root)

    raise ValueError(f"Unsupported command: {config.command}")


def run_simulate_command(
    config: ApplicationConfig,
    hydra_run_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    seed = config.experiment.seed

    generator = ClinicalTimeSeriesDatasetGenerator(config.simulator)
    data = generator.simulate(seed)

    if config.paths.data is not None:
        out = resolve_path(config.paths.data, project_root)
    else:
        out = data_root(config, hydra_run_dir, project_root) / "dataset.pt"

    data.save(out)
    return simulation_summary(
        data,
        clusters=config.simulator.n_clusters,
        out=out,
        seed=seed,
    )


def run_train_command(
    config: ApplicationConfig,
    hydra_run_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    seed = config.experiment.seed
    train_paths = train_paths_from_config(config, hydra_run_dir, project_root)
    result = fit_training_run(
        config,
        train_paths=train_paths,
        seed=seed,
        swanlab_repeat_label=None,
    )
    return train_output_payload(
        command="train",
        hydra_run_dir=hydra_run_dir,
        train_paths=train_paths,
        result=result,
        seed=seed,
    )


def run_experiment_command(
    config: ApplicationConfig,
    hydra_run_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    if (
        config.paths.data_root is not None
        or config.paths.data is not None
        or config.paths.test_data is not None
        or config.paths.train_root is not None
    ):
        raise ValueError("command=experiment generates its own split data per repeat.")

    repeats: list[dict[str, Any]] = []
    for index in range(config.experiment.repeats):
        # paired repeat: 同一个 repeat seed 同时驱动该轮 split 生成和模型训练。
        repeat_name = f"repeat_{index:03d}"
        repeat_dir = hydra_run_dir / repeat_name
        data_dir = repeat_dir / "data"
        train_root = repeat_dir / "train"

        data_dir.mkdir(parents=True, exist_ok=True)

        generator = ClinicalTimeSeriesDatasetGenerator(config.simulator)
        for i, split in enumerate(["train", "val", "test"]):
            data_i = generator.simulate(config.experiment.seed + i)
            data_i.save(data_dir / f"{split}.pt")

        train_paths = TrainPaths(
            data=data_dir / "train.pt",
            test_data=data_dir / "test.pt",
            train_root=train_root,
            save=repeat_checkpoint_path(config, repeat_dir, project_root, index),
        )
        train_result = fit_training_run(
            config,
            train_paths=train_paths,
            seed=config.experiment.seed,
            swanlab_repeat_label=f"r{index:03d}",
        )
        repeats.append(
            {
                "data_dir": str(data_dir),
                "index": index,
                "metrics": train_result.metrics,
                "repeat": repeat_name,
                "seed": config.experiment.seed,
                "train_run_dir": None
                if train_result.run_dir is None
                else str(train_result.run_dir),
            }
        )

    metrics_summary = summarize_repeat_metrics(repeats)
    summary = {
        "command": "experiment",
        "config": config.model_dump(mode="json"),
        "experiment": config.experiment.model_dump(mode="json"),
        "hydra_run_dir": str(hydra_run_dir),
        "metrics_summary": metrics_summary,
        "repeats": repeats,
    }

    save_json(hydra_run_dir / "experiment_summary.json", summary)
    save_repeat_metrics_csv(hydra_run_dir / "test_metrics.csv", repeats)
    save_json(hydra_run_dir / "test_metrics_summary.json", metrics_summary)
    return summary

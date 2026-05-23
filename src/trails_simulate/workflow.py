from __future__ import annotations

from pathlib import Path
from typing import Any

from trails.artifacts import save_json
from trails.data import ClinicalTimeSeriesDataset

from .baselines import make_baseline
from .config import ApplicationConfig
from .evaluation import (
    evaluate_predictions,
    json_safe_metrics,
    save_metrics_csv,
    save_prediction_payload,
    summarize_metric_rows,
)
from .generators import ClinicalTimeSeriesDatasetGenerator
from .optim import run_optim_command
from .path import (
    DatasetRunPaths,
    TrainPaths,
    checkpoint_path_for_run,
    data_root,
    discover_dataset_runs,
)
from .training import fit_training_run


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
    if config.command == "optim":
        return run_optim_command(config, hydra_run_dir, project_root)
    if config.command == "baseline":
        return run_baseline_command(config, hydra_run_dir, project_root)

    raise ValueError(f"Unsupported command: {config.command}")


def run_simulate_command(
    config: ApplicationConfig,
    hydra_run_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    if config.paths.data is not None or config.paths.test_data is not None:
        raise ValueError(
            "command=simulate writes split data under paths.data_root, not paths.data."
        )
    if config.paths.train_root is not None:
        raise ValueError("command=simulate does not use paths.train_root.")

    out_root = data_root(config, hydra_run_dir, project_root)
    repeats: list[dict[str, Any]] = []
    generator = ClinicalTimeSeriesDatasetGenerator(
        config.simulation.generator,
        mechanism_seed=config.simulation.mechanism_seed,
    )
    for index in range(config.simulation.repeats):
        repeat_seed = config.simulation.seed + index
        run_id = str(index)
        split_root = out_root / run_id
        split_root.mkdir(parents=True, exist_ok=True)

        total_patients = config.simulation.train_size + config.simulation.test_size
        source_dataset = generator.simulate(n_patients=total_patients, seed=repeat_seed)
        train_dataset, test_dataset = source_dataset.split_counts(
            [config.simulation.train_size, config.simulation.test_size],
            seed=repeat_seed,
        )
        train_path = split_root / "train.pt"
        test_path = split_root / "test.pt"
        train_dataset.save(train_path)
        test_dataset.save(test_path)
        repeats.append(
            {
                "data_root": str(split_root),
                "run_id": run_id,
                "seed": repeat_seed,
                "source_size": total_patients,
                "splits": {
                    "train": simulation_summary(
                        train_dataset,
                        clusters=config.simulation.generator.n_clusters,
                        out=train_path,
                        seed=repeat_seed,
                    ),
                    "test": simulation_summary(
                        test_dataset,
                        clusters=config.simulation.generator.n_clusters,
                        out=test_path,
                        seed=repeat_seed,
                    ),
                },
            }
        )

    summary = {
        "command": "simulate",
        "config": config.model_dump(mode="json"),
        "data_root": str(out_root),
        "hydra_run_dir": str(hydra_run_dir),
        "outputs": {
            "summary": str(out_root / "simulation_summary.json"),
        },
        "repeats": repeats,
        "simulation": config.simulation.model_dump(mode="json"),
    }
    save_json(out_root / "simulation_summary.json", summary)
    return summary


def run_train_command(
    config: ApplicationConfig,
    hydra_run_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    runs = discover_dataset_runs(config, hydra_run_dir, project_root)
    metric_rows: list[dict[str, Any]] = []
    run_payloads: list[dict[str, Any]] = []
    for index, run_paths in enumerate(runs):
        seed = config.simulation.seed + index
        train_paths = TrainPaths(
            data=run_paths.train_data,
            test_data=run_paths.test_data,
            train_root=hydra_run_dir / run_paths.run_id,
            save=checkpoint_path_for_run(
                config,
                hydra_run_dir=hydra_run_dir,
                project_root=project_root,
                run_id=run_paths.run_id,
                n_runs=len(runs),
            ),
        )
        train_result = fit_training_run(
            config,
            train_paths=train_paths,
            seed=seed,
            swanlab_repeat_label=None if len(runs) == 1 else run_paths.run_id,
        )
        prediction_path = hydra_run_dir / run_paths.run_id / "trails.pt"
        save_prediction_payload(prediction_path, train_result.prediction)
        row = metric_row(
            run_paths,
            method="trails",
            prediction_path=prediction_path,
            metrics=train_result.metrics,
        )
        metric_rows.append(row)
        run_payloads.append(
            {
                "history": train_result.history,
                "metrics": json_safe_metrics(train_result.metrics),
                "prediction_path": str(prediction_path),
                "run_dir": None if train_result.run_dir is None else str(train_result.run_dir),
                "run_id": run_paths.run_id,
                "seed": seed,
            }
        )

    summary_path = hydra_run_dir / "train_summary.json"
    metrics_csv_path = hydra_run_dir / "train_metrics.csv"
    summary = {
        "command": "train",
        "config": config.model_dump(mode="json"),
        "data_source": dataset_source_payload(config, runs),
        "hydra_run_dir": str(hydra_run_dir),
        "metrics_summary": summarize_metric_rows(metric_rows),
        "outputs": {
            "metrics_csv": str(metrics_csv_path),
            "summary": str(summary_path),
        },
        "runs": run_payloads,
        "simulation": config.simulation.model_dump(mode="json"),
    }

    save_json(summary_path, summary)
    save_metrics_csv(metrics_csv_path, metric_rows)
    return summary


def run_baseline_command(
    config: ApplicationConfig,
    hydra_run_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    runs = discover_dataset_runs(config, hydra_run_dir, project_root)
    n_clusters = config.baseline.n_clusters or config.simulation.generator.n_clusters
    metric_rows: list[dict[str, Any]] = []
    run_payloads: list[dict[str, Any]] = []

    for index, run_paths in enumerate(runs):
        seed = config.simulation.seed + index
        train_dataset = ClinicalTimeSeriesDataset.load(run_paths.train_data)
        test_dataset = ClinicalTimeSeriesDataset.load(run_paths.test_data)
        method_payloads: list[dict[str, Any]] = []
        for method in config.baseline.methods:
            baseline = make_baseline(
                method,
                n_clusters=n_clusters,
                random_state=seed,
                kmeans_iters=config.baseline.kmeans_iters,
                ridge_alpha=config.baseline.ridge_alpha,
                risk_feature_weight=config.baseline.risk_feature_weight,
            )
            prediction = baseline.fit(train_dataset).predict(test_dataset)
            metrics = evaluate_predictions(prediction, n_clusters=n_clusters)
            prediction_path = hydra_run_dir / run_paths.run_id / f"{method}.pt"
            save_prediction_payload(prediction_path, prediction)
            metric_rows.append(
                metric_row(
                    run_paths,
                    method=method,
                    prediction_path=prediction_path,
                    metrics=metrics,
                )
            )
            method_payloads.append(
                {
                    "method": method,
                    "metrics": json_safe_metrics(metrics),
                    "prediction_path": str(prediction_path),
                }
            )
        run_payloads.append(
            {
                "methods": method_payloads,
                "run_id": run_paths.run_id,
                "seed": seed,
            }
        )

    summary_path = hydra_run_dir / "baseline_summary.json"
    metrics_csv_path = hydra_run_dir / "baseline_metrics.csv"
    summary = {
        "baseline": {
            **config.baseline.model_dump(mode="json"),
            "n_clusters_resolved": n_clusters,
        },
        "command": "baseline",
        "config": config.model_dump(mode="json"),
        "data_source": dataset_source_payload(config, runs),
        "hydra_run_dir": str(hydra_run_dir),
        "metrics_summary": summarize_metric_rows(metric_rows),
        "outputs": {
            "metrics_csv": str(metrics_csv_path),
            "summary": str(summary_path),
        },
        "runs": run_payloads,
        "simulation": config.simulation.model_dump(mode="json"),
    }
    save_json(summary_path, summary)
    save_metrics_csv(metrics_csv_path, metric_rows)
    return summary


def metric_row(
    run_paths: DatasetRunPaths,
    *,
    method: str,
    prediction_path: Path,
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "data_root": str(run_paths.data_root),
        "method": method,
        "prediction_path": str(prediction_path),
        "run_id": run_paths.run_id,
        **metrics,
    }


def dataset_source_payload(
    config: ApplicationConfig,
    runs: list[DatasetRunPaths],
) -> dict[str, Any]:
    if not runs:
        return {"auto_selected": False, "data_root": None}
    if config.paths.data is not None:
        root = runs[0].data_root
    elif config.paths.data_root is not None:
        root = common_dataset_root(runs)
    else:
        root = common_dataset_root(runs)
    return {
        "auto_selected": config.paths.data is None and config.paths.data_root is None,
        "data_root": str(root),
    }


def common_dataset_root(runs: list[DatasetRunPaths]) -> Path:
    if runs[0].run_id.isdigit() and runs[0].data_root.name == runs[0].run_id:
        return runs[0].data_root.parent
    return runs[0].data_root if len(runs) == 1 else runs[0].data_root.parent

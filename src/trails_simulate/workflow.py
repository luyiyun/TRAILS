from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tqdm import tqdm

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
from .generators import ClinicalTimeSeriesDatasetGenerator, ClinicalTimeSeriesDatasetGeneratorConfig
from .optim import run_optim_command
from .path import (
    DatasetRunPaths,
    TrainPaths,
    checkpoint_path_for_run,
    data_root,
    discover_dataset_runs,
)
from .result_summary import run_summary_command
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
    if config.command == "summary":
        return run_summary_command(config, hydra_run_dir, project_root)

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

    out_root = data_root(config, hydra_run_dir, project_root) / config.simulation.name
    manifest_path = out_root / "simulation_manifest.csv"
    summary_path = out_root / "simulation_summary.json"
    runs: list[dict[str, Any]] = []

    n_iter = (
        len(config.simulation.generator.n_clusters_tuple_)
        * len(config.simulation.train_size)
        * config.simulation.repeats
    )
    iter_bar = tqdm(desc="Simulation", total=n_iter)

    for cluster_index, n_clusters in enumerate(config.simulation.generator.n_clusters_tuple_):
        generator_config = generator_config_for_cluster(
            config.simulation.generator,
            n_clusters=n_clusters,
        )
        mechanism_seed = simulation_mechanism_seed(config, cluster_index=cluster_index)
        generator = ClinicalTimeSeriesDatasetGenerator(
            generator_config,
            mechanism_seed=mechanism_seed,
        )
        for size_index, (train_size, test_size) in enumerate(
            zip(config.simulation.train_size, config.simulation.test_size, strict=True)
        ):
            total_patients = train_size + test_size
            for repeat_index in range(config.simulation.repeats):
                repeat_seed = simulation_sample_seed(
                    config,
                    size_index=size_index,
                    cluster_index=cluster_index,
                    repeat_index=repeat_index,
                )
                run_id = f"train_{train_size}_test_{test_size}/k{n_clusters}/{repeat_index}"
                split_root = out_root / run_id
                split_root.mkdir(parents=True, exist_ok=True)

                source_dataset = generator.simulate(n_patients=total_patients, seed=repeat_seed)
                train_dataset, test_dataset = source_dataset.split_counts(
                    [train_size, test_size],
                    seed=repeat_seed,
                )
                train_path = split_root / "train.pt"
                test_path = split_root / "test.pt"
                train_dataset.save(train_path)
                test_dataset.save(test_path)
                train_summary = simulation_summary(
                    train_dataset,
                    clusters=n_clusters,
                    out=train_path,
                    seed=repeat_seed,
                )
                test_summary = simulation_summary(
                    test_dataset,
                    clusters=n_clusters,
                    out=test_path,
                    seed=repeat_seed,
                )
                runs.append(
                    {
                        "data_root": str(split_root),
                        "n_clusters": n_clusters,
                        "n_features": train_dataset.n_features,
                        "repeat": repeat_index,
                        "run_id": run_id,
                        "seed": repeat_seed,
                        "mechanism_seed": mechanism_seed,
                        "source_size": total_patients,
                        "train_size": train_size,
                        "test_size": test_size,
                        "train_censoring_rate": train_summary["censoring_rate"],
                        "test_censoring_rate": test_summary["censoring_rate"],
                        "train_path": str(train_path),
                        "test_path": str(test_path),
                        "splits": {
                            "train": train_summary,
                            "test": test_summary,
                        },
                    }
                )

                iter_bar.update()

    summary = {
        "command": "simulate",
        "config": config.model_dump(mode="json"),
        "data_root": str(out_root),
        "hydra_run_dir": str(hydra_run_dir),
        "outputs": {
            "manifest": str(manifest_path),
            "summary": str(summary_path),
        },
        "runs": runs,
        "simulation": config.simulation.model_dump(mode="json"),
    }
    save_metrics_csv(manifest_path, runs)
    save_json(summary_path, summary)
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
        print(f"Training run {index + 1}/{len(runs)}: {run_paths.run_id}")
        seed = config.training.trainer.seed + index
        run_config = config_for_dataset_clusters(config, run_paths.train_data)
        train_paths = TrainPaths(
            data=run_paths.train_data,
            test_data=run_paths.test_data,
            train_root=hydra_run_dir / run_paths.run_id,
            save=checkpoint_path_for_run(
                run_config,
                hydra_run_dir=hydra_run_dir,
                project_root=project_root,
                run_id=run_paths.run_id,
                n_runs=len(runs),
            ),
        )
        train_result = fit_training_run(
            run_config,
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
        tqdm.write(
            format_completed_train_run(
                run_id=run_paths.run_id,
                n_clusters=run_config.training.model.n_clusters,
                seed=seed,
                prediction_path=prediction_path,
                metrics=train_result.metrics,
            )
        )
        run_payloads.append(
            {
                "history": train_result.history,
                "metrics": json_safe_metrics(train_result.metrics),
                "prediction_path": str(prediction_path),
                "run_dir": None if train_result.run_dir is None else str(train_result.run_dir),
                "run_id": run_paths.run_id,
                "seed": seed,
                "n_clusters": run_config.training.model.n_clusters,
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
    metric_rows: list[dict[str, Any]] = []
    run_payloads: list[dict[str, Any]] = []

    iter_bar = tqdm(desc="Baseline", total=len(runs) * len(config.baseline.methods))

    for index, run_paths in enumerate(runs):
        seed = config.training.trainer.seed + index
        train_dataset = ClinicalTimeSeriesDataset.load(run_paths.train_data)
        test_dataset = ClinicalTimeSeriesDataset.load(run_paths.test_data)
        n_clusters = config.baseline.n_clusters or dataset_n_clusters(
            train_dataset,
            fallback=config.training.model.n_clusters,
        )
        method_payloads: list[dict[str, Any]] = []
        for method in config.baseline.methods:
            baseline = make_baseline(
                method,
                n_clusters=n_clusters,
                random_state=seed,
                kmeans_iters=config.baseline.kmeans_iters,
                ridge_alpha=config.baseline.ridge_alpha,
                risk_feature_weight=config.baseline.risk_feature_weight,
                fpca_components=config.baseline.fpca_components,
                fpca_grid_size=config.baseline.fpca_grid_size,
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
            iter_bar.update()
        run_payloads.append(
            {
                "methods": method_payloads,
                "n_clusters": n_clusters,
                "run_id": run_paths.run_id,
                "seed": seed,
            }
        )

    summary_path = hydra_run_dir / "baseline_summary.json"
    metrics_csv_path = hydra_run_dir / "baseline_metrics.csv"
    summary = {
        "baseline": {
            **config.baseline.model_dump(mode="json"),
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


def format_completed_train_run(
    *,
    run_id: str,
    n_clusters: int,
    seed: int,
    prediction_path: Path,
    metrics: Mapping[str, float],
) -> str:
    metric_names = ("cindex", "ari", "nmi", "acc", "cluster_empty_count")
    metric_text = " ".join(
        f"{name}={float(metrics[name]):.4g}" for name in metric_names if name in metrics
    )
    return (
        f"Completed train run: {run_id} "
        f"k={n_clusters} seed={seed} {metric_text} prediction={prediction_path}"
    )


def dataset_source_payload(
    config: ApplicationConfig,
    runs: list[DatasetRunPaths],
) -> dict[str, Any]:
    if not runs:
        return {"data_root": None, "source": "empty"}
    if config.paths.data is not None:
        root = runs[0].data_root
        source = "explicit split"
    elif config.paths.data_root is not None:
        root = config.paths.data_root
        source = "data root"
    else:
        root = common_dataset_root(runs)
        source = "discovered split"
    return {
        "data_root": str(root),
        "source": source,
    }


def common_dataset_root(runs: list[DatasetRunPaths]) -> Path:
    if runs[0].run_id.isdigit() and runs[0].data_root.name == runs[0].run_id:
        return runs[0].data_root.parent
    return runs[0].data_root if len(runs) == 1 else runs[0].data_root.parent


def generator_config_for_cluster(
    generator_config: ClinicalTimeSeriesDatasetGeneratorConfig,
    *,
    n_clusters: int,
) -> ClinicalTimeSeriesDatasetGeneratorConfig:
    payload = generator_config.model_dump(mode="json")
    payload["n_clusters"] = n_clusters
    return ClinicalTimeSeriesDatasetGeneratorConfig.model_validate(payload)


def simulation_mechanism_seed(config: ApplicationConfig, *, cluster_index: int) -> int:
    base_seed = config.simulation.mechanism_seed or config.simulation.seed
    return base_seed + cluster_index * 100


def simulation_sample_seed(
    config: ApplicationConfig,
    *,
    size_index: int,
    cluster_index: int,
    repeat_index: int,
) -> int:
    return config.simulation.seed + size_index * 10_000 + cluster_index * 100 + repeat_index


def dataset_n_clusters(dataset: ClinicalTimeSeriesDataset, *, fallback: int) -> int:
    params = dataset.metadata.get("generation_params")
    if isinstance(params, Mapping) and "n_clusters" in params:
        return int(params["n_clusters"])
    return fallback


def config_for_dataset_clusters(
    config: ApplicationConfig,
    train_data: Path,
) -> ApplicationConfig:
    dataset = ClinicalTimeSeriesDataset.load(train_data)
    n_clusters = dataset_n_clusters(dataset, fallback=config.training.model.n_clusters)
    return config.model_copy(
        update={
            "training": config.training.model_copy(
                update={
                    "model": config.training.model.model_copy(update={"n_clusters": n_clusters})
                }
            )
        }
    )

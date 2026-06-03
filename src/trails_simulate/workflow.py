from __future__ import annotations

import logging
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trails.artifacts import save_json
from trails.data import ClinicalTimeSeriesDataset
from trails.progress import ProgressBar, ProgressManager

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
    discover_dataset_runs,
)
from .result_summary import run_summary_command
from .training import fit_training_run

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainRunJob:
    config: ApplicationConfig
    hydra_run_dir: Path
    index: int
    project_root: Path
    run_paths: DatasetRunPaths
    total_runs: int
    worker_slot: int
    device: str | None


@dataclass(frozen=True)
class TrainRunResult:
    index: int
    metric_row: dict[str, Any]
    metrics: dict[str, float]
    n_clusters: int
    prediction_path: Path
    run_id: str
    run_payload: dict[str, Any]
    seed: int


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
    if config.paths.explicit_split.enabled:
        raise ValueError(
            "command=simulate generates new split data in the Hydra run directory; "
            "paths.explicit_split is only for commands that read existing split data."
        )

    out_root = hydra_run_dir
    manifest_path = hydra_run_dir / "simulation_manifest.csv"
    summary_path = hydra_run_dir / "simulation_summary.json"
    runs: list[dict[str, Any]] = []

    n_iter = (
        len(config.simulation.generator.n_clusters_tuple_)
        * len(config.simulation.train_size)
        * config.simulation.repeats
    )
    iter_bar = ProgressBar(desc="Simulation", total=n_iter)

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
    train_results = run_train_jobs(config, runs, hydra_run_dir, project_root)
    metric_rows = [result.metric_row for result in train_results]
    run_payloads = [result.run_payload for result in train_results]

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


def run_train_jobs(
    config: ApplicationConfig,
    runs: list[DatasetRunPaths],
    hydra_run_dir: Path,
    project_root: Path,
) -> list[TrainRunResult]:
    if not runs:
        return []

    workers = effective_train_workers(config, n_runs=len(runs))
    results: list[TrainRunResult] = []

    if workers == 1:
        with ProgressBar(desc="Train splits", total=len(runs)) as run_bar:
            for index, run_paths in enumerate(runs):
                job = build_train_run_job(
                    config,
                    hydra_run_dir=hydra_run_dir,
                    index=index,
                    project_root=project_root,
                    run_paths=run_paths,
                    total_runs=len(runs),
                    worker_slot=0,
                )
                result = run_train_split_job(job)
                record_train_result(
                    result,
                    results=results,
                    run_bar=run_bar,
                    total_runs=len(runs),
                )
    else:
        with ProgressManager(workers=workers) as progress_manager:
            with ProgressBar(desc="Train splits", total=len(runs)) as run_bar:
                results.extend(
                    run_train_jobs_parallel(
                        config,
                        runs,
                        hydra_run_dir=hydra_run_dir,
                        project_root=project_root,
                        workers=workers,
                        run_bar=run_bar,
                        progress_manager=progress_manager,
                    )
                )

    return sorted(results, key=lambda result: result.index)


def run_train_jobs_parallel(
    config: ApplicationConfig,
    runs: list[DatasetRunPaths],
    *,
    hydra_run_dir: Path,
    project_root: Path,
    workers: int,
    run_bar: ProgressBar[Any],
    progress_manager: ProgressManager,
) -> list[TrainRunResult]:
    results: list[TrainRunResult] = []
    next_index = 0
    futures: dict[Future[TrainRunResult], TrainRunJob] = {}

    def submit_next(executor: ProcessPoolExecutor, worker_slot: int) -> None:
        nonlocal next_index
        if next_index >= len(runs):
            return
        index = next_index
        next_index += 1
        job = build_train_run_job(
            config,
            hydra_run_dir=hydra_run_dir,
            index=index,
            project_root=project_root,
            run_paths=runs[index],
            total_runs=len(runs),
            worker_slot=worker_slot,
        )
        futures[executor.submit(run_train_split_job, job)] = job

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=progress_manager.mp_context,
        initializer=ProgressManager.initialize_worker,
        initargs=progress_manager.worker_initargs(),
    ) as executor:
        for worker_slot in range(workers):
            submit_next(executor, worker_slot)

        while futures:
            done, _pending = wait(set(futures), return_when=FIRST_COMPLETED)
            for future in done:
                job = futures.pop(future)
                try:
                    result = future.result()
                except Exception:
                    LOGGER.exception("Train run failed: %s", job.run_paths.run_id)
                    for pending_future in futures:
                        pending_future.cancel()
                    raise
                record_train_result(
                    result,
                    results=results,
                    run_bar=run_bar,
                    total_runs=len(runs),
                )
                submit_next(executor, job.worker_slot)

    return results


def build_train_run_job(
    config: ApplicationConfig,
    *,
    hydra_run_dir: Path,
    index: int,
    project_root: Path,
    run_paths: DatasetRunPaths,
    total_runs: int,
    worker_slot: int,
) -> TrainRunJob:
    return TrainRunJob(
        config=config,
        hydra_run_dir=hydra_run_dir,
        index=index,
        project_root=project_root,
        run_paths=run_paths,
        total_runs=total_runs,
        worker_slot=worker_slot,
        device=device_for_worker_slot(config, worker_slot),
    )


def run_train_split_job(job: TrainRunJob) -> TrainRunResult:
    configure_torch_threads(job.config.training.parallel.torch_threads)
    seed = job.config.training.trainer.seed + job.index
    run_config = config_for_dataset_clusters(job.config, job.run_paths.train_data)
    if job.device is not None:
        run_config = config_with_training_device(run_config, job.device)
    train_paths = TrainPaths(
        data=job.run_paths.train_data,
        test_data=job.run_paths.test_data,
        train_root=job.hydra_run_dir / job.run_paths.run_id,
        save=checkpoint_path_for_run(
            run_config,
            hydra_run_dir=job.hydra_run_dir,
            project_root=job.project_root,
            run_id=job.run_paths.run_id,
            n_runs=job.total_runs,
        ),
    )
    with ProgressManager.worker_scope(
        worker_slot=job.worker_slot,
        description_prefix=short_run_label(job.run_paths.run_id),
        leave=False,
    ):
        train_result = fit_training_run(
            run_config,
            train_paths=train_paths,
            seed=seed,
            swanlab_repeat_label=None if job.total_runs == 1 else job.run_paths.run_id,
        )
    prediction_path = job.hydra_run_dir / job.run_paths.run_id / "trails.pt"
    save_prediction_payload(prediction_path, train_result.prediction)
    row = metric_row(
        job.run_paths,
        method="trails",
        prediction_path=prediction_path,
        metrics=train_result.metrics,
    )
    return TrainRunResult(
        index=job.index,
        metric_row=row,
        metrics=train_result.metrics,
        n_clusters=run_config.training.model.n_clusters,
        prediction_path=prediction_path,
        run_id=job.run_paths.run_id,
        run_payload={
            "history": train_result.history,
            "metrics": json_safe_metrics(train_result.metrics),
            "prediction_path": str(prediction_path),
            "run_dir": None if train_result.run_dir is None else str(train_result.run_dir),
            "run_id": job.run_paths.run_id,
            "seed": seed,
            "n_clusters": run_config.training.model.n_clusters,
        },
        seed=seed,
    )


def record_train_result(
    result: TrainRunResult,
    *,
    results: list[TrainRunResult],
    run_bar: ProgressBar[Any],
    total_runs: int,
) -> None:
    results.append(result)
    run_bar.update()
    run_bar.set_postfix(completed=f"{len(results)}/{total_runs}")
    LOGGER.info(
        format_completed_train_run(
            run_id=result.run_id,
            n_clusters=result.n_clusters,
            seed=result.seed,
            prediction_path=result.prediction_path,
            metrics=result.metrics,
        )
    )


def effective_train_workers(config: ApplicationConfig, *, n_runs: int) -> int:
    return min(config.training.parallel.workers, max(1, n_runs))


def device_for_worker_slot(config: ApplicationConfig, worker_slot: int) -> str | None:
    devices = config.training.parallel.devices
    if not devices:
        return None
    return devices[worker_slot % len(devices)]


def config_with_training_device(config: ApplicationConfig, device: str) -> ApplicationConfig:
    return config.model_copy(
        update={
            "training": config.training.model_copy(
                update={"trainer": config.training.trainer.model_copy(update={"device": device})}
            )
        }
    )


def configure_torch_threads(torch_threads: int | None) -> None:
    if torch_threads is None:
        return
    import torch

    torch.set_num_threads(torch_threads)


def short_run_label(run_id: str) -> str:
    parts = Path(run_id).parts
    label = "/".join(parts[-3:])
    if len(label) <= 36:
        return label
    return f"...{label[-33:]}"


def run_baseline_command(
    config: ApplicationConfig,
    hydra_run_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    runs = discover_dataset_runs(config, hydra_run_dir, project_root)
    metric_rows: list[dict[str, Any]] = []
    run_payloads: list[dict[str, Any]] = []

    iter_bar = ProgressBar(desc="Baseline", total=len(runs) * len(config.baseline.methods))

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
    fields = [f"Completed train run: {run_id}", f"k={n_clusters}", f"seed={seed}"]
    if metric_text:
        fields.append(metric_text)
    fields.append(f"prediction={compact_log_path(prediction_path)}")
    return " ".join(fields)


def compact_log_path(path: Path, *, keep_parts: int = 4) -> str:
    parts = path.parts
    if len(parts) <= keep_parts:
        return str(path)
    return str(Path("...").joinpath(*parts[-keep_parts:]))


def dataset_source_payload(
    config: ApplicationConfig,
    runs: list[DatasetRunPaths],
) -> dict[str, Any]:
    if not runs:
        return {"data_root": None, "source": "empty"}
    if config.paths.explicit_split.enabled:
        root = runs[0].data_root
        source = "explicit split"
    else:
        root = config.paths.data_root
        source = "data root"
    return {
        "data_root": str(root),
        "source": source,
    }


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

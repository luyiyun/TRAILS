from __future__ import annotations

import logging
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trails.data import ClinicalTimeSeriesDataset
from trails.progress import ProgressBar, ProgressManager

from .command_utils import dataset_n_clusters, format_completed_train_run, metric_row
from .config import TrainApplicationConfig
from .evaluation import json_safe_metrics, save_prediction_payload
from .path import DatasetRunPaths, TrainPaths, checkpoint_path_for_run
from .training import fit_training_run

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainRunJob:
    config: TrainApplicationConfig
    index: int
    run_dir: Path
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


def run_train_jobs(
    config: TrainApplicationConfig,
    runs: list[DatasetRunPaths],
) -> list[TrainRunResult]:
    if not runs:
        return []

    workers = effective_train_workers(config, n_runs=len(runs))
    run_dir = config.paths.dir
    results: list[TrainRunResult] = []

    if workers == 1:
        with ProgressBar(desc="Train splits", total=len(runs)) as run_bar:
            for index, run_paths in enumerate(runs):
                job = build_train_run_job(
                    config,
                    index=index,
                    run_dir=run_dir,
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
                        run_dir=run_dir,
                        workers=workers,
                        run_bar=run_bar,
                        progress_manager=progress_manager,
                    )
                )

    return sorted(results, key=lambda result: result.index)


def run_train_jobs_parallel(
    config: TrainApplicationConfig,
    runs: list[DatasetRunPaths],
    *,
    run_dir: Path,
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
            index=index,
            run_dir=run_dir,
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
    config: TrainApplicationConfig,
    *,
    index: int,
    run_dir: Path,
    run_paths: DatasetRunPaths,
    total_runs: int,
    worker_slot: int,
) -> TrainRunJob:
    return TrainRunJob(
        config=config,
        index=index,
        run_dir=run_dir,
        run_paths=run_paths,
        total_runs=total_runs,
        worker_slot=worker_slot,
        device=device_for_worker_slot(config, worker_slot),
    )


def run_train_split_job(job: TrainRunJob) -> TrainRunResult:
    configure_torch_threads(job.config.parallel.torch_threads)
    seed = job.config.trainer.seed + job.index
    run_config = config_for_dataset_clusters(job.config, job.run_paths.train_data)
    if job.device is not None:
        run_config = config_with_training_device(run_config, job.device)
    train_paths = TrainPaths(
        data=job.run_paths.train_data,
        test_data=job.run_paths.test_data,
        train_root=job.run_dir / job.run_paths.run_id,
        save=checkpoint_path_for_run(
            run_config,
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
    prediction_path = job.run_dir / job.run_paths.run_id / "trails.pt"
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
        n_clusters=run_config.model.n_clusters,
        prediction_path=prediction_path,
        run_id=job.run_paths.run_id,
        run_payload={
            "history": train_result.history,
            "metrics": json_safe_metrics(train_result.metrics),
            "prediction_path": str(prediction_path),
            "run_dir": None if train_result.run_dir is None else str(train_result.run_dir),
            "run_id": job.run_paths.run_id,
            "seed": seed,
            "n_clusters": run_config.model.n_clusters,
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


def effective_train_workers(config: TrainApplicationConfig, *, n_runs: int) -> int:
    return min(config.parallel.workers, max(1, n_runs))


def device_for_worker_slot(config: TrainApplicationConfig, worker_slot: int) -> str | None:
    devices = config.parallel.devices
    if not devices:
        return None
    return devices[worker_slot % len(devices)]


def config_with_training_device(
    config: TrainApplicationConfig,
    device: str,
) -> TrainApplicationConfig:
    return config.model_copy(
        update={"trainer": config.trainer.model_copy(update={"device": device})}
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


def config_for_dataset_clusters(
    config: TrainApplicationConfig,
    train_data: Path,
) -> TrainApplicationConfig:
    dataset = ClinicalTimeSeriesDataset.load(train_data)
    n_clusters = dataset_n_clusters(dataset, fallback=config.model.n_clusters)
    return config.model_copy(
        update={"model": config.model.model_copy(update={"n_clusters": n_clusters})}
    )

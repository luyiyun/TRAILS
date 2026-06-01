from __future__ import annotations

import csv
import hashlib
import logging
import math
import multiprocessing as mp
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import matplotlib

from trails.artifacts import save_json
from trails.data import ClinicalTimeSeriesDataset

from .config import (
    OPTIM_PARAM_NAMES,
    ApplicationConfig,
    FloatSearchRangeConfig,
)
from .path import DatasetRunPaths, TrainPaths, discover_dataset_runs, resolve_path
from .training import fit_training_run

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimSplitJob:
    config: ApplicationConfig
    device: str
    optim_root: Path
    run_paths: DatasetRunPaths
    seed: int
    split_index: int
    total_splits: int
    trial_number: int


@dataclass(frozen=True)
class OptimSplitResult:
    metrics: dict[str, float]
    run_id: str
    seed: int
    split_index: int
    trial_number: int


@dataclass
class ActiveOptimTrial:
    trial: Any
    results: list[OptimSplitResult] = field(default_factory=list)
    futures: set[Future[OptimSplitResult]] = field(default_factory=set)


def run_optim_command(
    config: ApplicationConfig,
    hydra_run_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    optuna = load_optuna()
    runs = discover_dataset_runs(config, hydra_run_dir, project_root)
    selected_runs, selection_source = select_optim_runs(runs, config.optim.run_ids)
    for run_paths in selected_runs:
        validate_optim_test_data(run_paths.test_data)

    fingerprint = dataset_fingerprint(selected_runs)
    fingerprint_path = hydra_run_dir / "dataset_fingerprint.json"
    validate_or_write_dataset_fingerprint(
        config=config,
        fingerprint_path=fingerprint_path,
        fingerprint=fingerprint,
        hydra_run_dir=hydra_run_dir,
    )

    storage_url = optim_storage_url(config.optim.storage, hydra_run_dir, project_root)
    sampler = optuna.samplers.TPESampler(seed=config.training.trainer.seed)
    study = optuna.create_study(
        directions=["maximize", "maximize"],
        load_if_exists=True,
        sampler=sampler,
        storage=storage_url,
        study_name=config.optim.study_name,
    )

    completed_before = count_completed_trials(study.trials)
    run_optim_study(
        study,
        config=config,
        selected_runs=selected_runs,
        optim_root=hydra_run_dir,
    )
    completed_after = count_completed_trials(study.trials)

    summary_path = hydra_run_dir / "optim_summary.json"
    trials_csv_path = hydra_run_dir / "trials.csv"
    pareto_path = hydra_run_dir / "pareto_trials.json"
    top_trials_csv_path = hydra_run_dir / "top_trials.csv"
    figures_dir = hydra_run_dir / "figures"
    pareto_trials = serialize_optim_trials(best_trials(study))
    all_trials = serialize_optim_trials(study.trials)
    figures = save_optim_figures(
        figures_dir,
        trials=study.trials,
        pareto_trials=best_trials(study),
        selected_runs=selected_runs,
    )

    save_optim_trials_csv(trials_csv_path, study.trials)
    save_top_trials_csv(top_trials_csv_path, study.trials)
    save_json(pareto_path, pareto_trials)

    summary = {
        "command": "optim",
        "completed_after": completed_after,
        "completed_before": completed_before,
        "config": config.model_dump(mode="json"),
        "data_fingerprint": fingerprint,
        "hydra_run_dir": str(hydra_run_dir),
        "n_trials_requested": config.optim.n_trials,
        "outputs": {
            "dataset_fingerprint": str(fingerprint_path),
            "figures": figures,
            "pareto_trials": str(pareto_path),
            "summary": str(summary_path),
            "top_trials_csv": str(top_trials_csv_path),
            "trials_csv": str(trials_csv_path),
        },
        "paths": {
            "optim_summary": str(summary_path),
        },
        "pareto_trials": pareto_trials,
        "selected_run_ids": [run.run_id for run in selected_runs],
        "selection": {
            "available_run_ids": [run.run_id for run in runs],
            "run_ids": [run.run_id for run in selected_runs],
            "source": selection_source,
        },
        "splits": [optim_split_payload(run_paths) for run_paths in selected_runs],
        "storage": storage_url,
        "study_name": study.study_name,
        "trials": all_trials,
    }

    save_json(summary_path, summary)
    return summary


def select_optim_runs(
    runs: Sequence[DatasetRunPaths],
    configured_run_ids: Sequence[str],
) -> tuple[list[DatasetRunPaths], str]:
    if not configured_run_ids:
        return list(runs), "all"

    configured = set(configured_run_ids)
    selected = [run for run in runs if run.run_id in configured]
    missing = sorted(configured - {run.run_id for run in selected})
    if missing:
        raise ValueError(
            "optim.run_ids contained values that did not match discovered dataset splits: "
            f"{', '.join(missing)}. Available run_id values: {format_available_run_ids(runs)}"
        )
    return selected, "configured"


def format_available_run_ids(runs: Sequence[DatasetRunPaths]) -> str:
    return ", ".join(run.run_id for run in runs)


def run_optim_study(
    study: Any,
    *,
    config: ApplicationConfig,
    selected_runs: Sequence[DatasetRunPaths],
    optim_root: Path,
) -> None:
    if config.optim.parallel.workers == 1:
        run_optim_study_serial(
            study, config=config, selected_runs=selected_runs, optim_root=optim_root
        )
        return

    context = mp.get_context("spawn")
    futures: dict[Future[OptimSplitResult], ActiveOptimTrial] = {}
    active_trials: dict[int, ActiveOptimTrial] = {}
    started_trials = 0
    next_task_slot = 0

    def submit_trial(executor: ProcessPoolExecutor) -> None:
        nonlocal started_trials, next_task_slot
        trial = study.ask()
        trial_config = optim_trial_config(config, trial)
        active_trial = ActiveOptimTrial(trial=trial)
        active_trials[int(trial.number)] = active_trial
        for split_index, run_paths in enumerate(selected_runs):
            job = build_optim_split_job(
                trial_config,
                optim_root=optim_root,
                run_paths=run_paths,
                split_index=split_index,
                task_slot=next_task_slot,
                total_splits=len(selected_runs),
                trial_number=int(trial.number),
            )
            next_task_slot += 1
            future = executor.submit(run_optim_split_job, job)
            active_trial.futures.add(future)
            futures[future] = active_trial
        started_trials += 1

    with ProcessPoolExecutor(
        max_workers=config.optim.parallel.workers, mp_context=context
    ) as executor:
        while started_trials < config.optim.n_trials and len(
            active_trials
        ) < effective_active_trials(config):
            submit_trial(executor)

        while futures:
            done, _pending = wait(set(futures), return_when=FIRST_COMPLETED)
            for future in done:
                active_trial = futures.pop(future)
                active_trial.futures.remove(future)
                try:
                    active_trial.results.append(future.result())
                except Exception:
                    LOGGER.exception("Optim split failed.")
                    for pending_future in futures:
                        pending_future.cancel()
                    raise
                if not active_trial.futures:
                    active_trials.pop(int(active_trial.trial.number), None)
                    complete_optim_trial(study, active_trial)
                    if started_trials < config.optim.n_trials:
                        submit_trial(executor)


def run_optim_study_serial(
    study: Any,
    *,
    config: ApplicationConfig,
    selected_runs: Sequence[DatasetRunPaths],
    optim_root: Path,
) -> None:
    for _trial_index in range(config.optim.n_trials):
        trial = study.ask()
        trial_config = optim_trial_config(config, trial)
        active_trial = ActiveOptimTrial(trial=trial)
        for split_index, run_paths in enumerate(selected_runs):
            job = build_optim_split_job(
                trial_config,
                optim_root=optim_root,
                run_paths=run_paths,
                split_index=split_index,
                task_slot=split_index,
                total_splits=len(selected_runs),
                trial_number=int(trial.number),
            )
            active_trial.results.append(run_optim_split_job(job))
        complete_optim_trial(study, active_trial)


def effective_active_trials(config: ApplicationConfig) -> int:
    return max(1, min(config.optim.parallel.max_active_trials, config.optim.n_trials))


def build_optim_split_job(
    config: ApplicationConfig,
    *,
    optim_root: Path,
    run_paths: DatasetRunPaths,
    split_index: int,
    task_slot: int,
    total_splits: int,
    trial_number: int,
) -> OptimSplitJob:
    return OptimSplitJob(
        config=config,
        device=optim_device_for_task(config, task_slot),
        optim_root=optim_root,
        run_paths=run_paths,
        seed=config.training.trainer.seed + trial_number * total_splits + split_index,
        split_index=split_index,
        total_splits=total_splits,
        trial_number=trial_number,
    )


def optim_device_for_task(config: ApplicationConfig, task_slot: int) -> str:
    devices = config.optim.parallel.devices
    if devices:
        return devices[task_slot % len(devices)]
    return config.training.trainer.device


def run_optim_split_job(job: OptimSplitJob) -> OptimSplitResult:
    configure_torch_threads(job.config.optim.parallel.torch_threads)
    run_config = config_for_dataset_clusters(job.config, job.run_paths.train_data)
    run_config = config_with_training_device(run_config, job.device)
    train_paths = TrainPaths(
        data=job.run_paths.train_data,
        test_data=job.run_paths.test_data,
        train_root=job.optim_root
        / "trial_runs"
        / f"trial_{job.trial_number}"
        / job.run_paths.run_id,
        save=None,
    )
    result = fit_training_run(
        run_config,
        train_paths=train_paths,
        seed=job.seed,
        swanlab_repeat_label=None,
    )
    required_metric(result.metrics, "cindex", job.trial_number)
    required_metric(result.metrics, "ari", job.trial_number)
    return OptimSplitResult(
        metrics=result.metrics,
        run_id=job.run_paths.run_id,
        seed=job.seed,
        split_index=job.split_index,
        trial_number=job.trial_number,
    )


def complete_optim_trial(study: Any, active_trial: ActiveOptimTrial) -> None:
    aggregate = aggregate_split_results(active_trial.results)
    split_metrics = [
        split_result_payload(result)
        for result in sorted(active_trial.results, key=lambda item: item.split_index)
    ]
    trial = active_trial.trial
    set_trial_user_attr(trial, "seed", aggregate["seed"])
    set_trial_user_attr(trial, "metrics", aggregate["metrics"])
    set_trial_user_attr(trial, "split_metrics", split_metrics)
    study.tell(
        trial, values=(aggregate["metrics"]["mean_cindex"], aggregate["metrics"]["mean_ari"])
    )
    LOGGER.info(
        "Completed optim trial %s mean_cindex=%.4g mean_ari=%.4g",
        trial.number,
        aggregate["metrics"]["mean_cindex"],
        aggregate["metrics"]["mean_ari"],
    )


def aggregate_split_results(results: Sequence[OptimSplitResult]) -> dict[str, Any]:
    cindex_values = [
        required_metric(result.metrics, "cindex", result.trial_number) for result in results
    ]
    ari_values = [required_metric(result.metrics, "ari", result.trial_number) for result in results]
    mean_cindex = float(fmean(cindex_values))
    mean_ari = float(fmean(ari_values))
    std_cindex = float(pstdev(cindex_values)) if len(cindex_values) > 1 else 0.0
    std_ari = float(pstdev(ari_values)) if len(ari_values) > 1 else 0.0
    return {
        "metrics": {
            "mean_ari": mean_ari,
            "mean_cindex": mean_cindex,
            "mean_objective": (mean_cindex + mean_ari) / 2.0,
            "std_ari": std_ari,
            "std_cindex": std_cindex,
        },
        "seed": min(result.seed for result in results),
    }


def split_result_payload(result: OptimSplitResult) -> dict[str, Any]:
    return {
        "ari": required_metric(result.metrics, "ari", result.trial_number),
        "cindex": required_metric(result.metrics, "cindex", result.trial_number),
        "metrics": json_safe_metrics(result.metrics),
        "run_id": result.run_id,
        "seed": result.seed,
        "split_index": result.split_index,
    }


def load_optuna() -> Any:
    try:
        import optuna
    except ImportError as error:
        raise RuntimeError(
            "command=optim requires Optuna. Install the project dev dependencies with "
            "`uv sync --group dev` before running `uv run main.py command=optim`."
        ) from error
    return optuna


def optim_trial_config(config: ApplicationConfig, trial: Any) -> ApplicationConfig:
    search = config.optim.search
    encoder_input_kind = str(
        trial.suggest_categorical("encoder_input_kind", list(search.encoder_input_kind))
    )
    encoder_mapping_kind = str(
        trial.suggest_categorical("encoder_mapping_kind", list(search.encoder_mapping_kind))
    )
    decoder_kind = str(trial.suggest_categorical("decoder_kind", list(search.decoder_kind)))
    if decoder_kind == "transformer":
        decoder_conditioning = "concat_time"
    else:
        decoder_conditioning = str(
            trial.suggest_categorical(
                "decoder_conditioning",
                list(search.decoder_conditioning),
            )
        )
    set_trial_user_attr(trial, "decoder_conditioning", decoder_conditioning)
    hidden_dim = int(trial.suggest_categorical("hidden_dim", list(search.hidden_dim)))
    n_layers = int(trial.suggest_categorical("n_layers", list(search.n_layers)))
    model = config.training.model
    trainer = config.training.trainer
    encoder_config = model.encoder.model_copy(
        update={
            "input": model.encoder.input.model_copy(
                update={"kind": encoder_input_kind, "hidden_dim": hidden_dim}
            ),
            "mapping": model.encoder.mapping.model_copy(
                update={
                    "kind": encoder_mapping_kind,
                    "hidden_dim": hidden_dim,
                    "n_layers": n_layers,
                }
            ),
        }
    )
    decoder_config = model.decoder.model_copy(
        update={
            "kind": decoder_kind,
            "conditioning": decoder_conditioning,
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
        }
    )
    model_config = model.model_copy(
        update={
            "dropout": suggest_float_range(trial, "dropout", search.dropout),
            "encoder": encoder_config,
            "decoder": decoder_config,
            "latent_dim": int(trial.suggest_categorical("latent_dim", list(search.latent_dim))),
            "survival_head_hidden_layers": int(
                trial.suggest_categorical(
                    "survival_head_hidden_layers",
                    list(search.survival_head_hidden_layers),
                )
            ),
        }
    )
    trainer_config = trainer.model_copy(
        update={
            "batch_size": int(trial.suggest_categorical("batch_size", list(search.batch_size))),
            "gmm_init_iters": int(
                trial.suggest_categorical("gmm_init_iters", list(search.gmm_init_iters))
            ),
            "learning_rate": suggest_float_range(trial, "learning_rate", search.learning_rate),
            "warmup_epochs": int(
                trial.suggest_int(
                    "warmup_epochs",
                    search.warmup_epochs.low,
                    search.warmup_epochs.high,
                )
            ),
        }
    )

    # optim 只保留 Optuna bookkeeping，训练过程中的模型、图和诊断产物全部关闭。
    diagnostics_config = config.training.diagnostics.model_copy(
        update={
            "latent_embeddings": config.training.diagnostics.latent_embeddings.model_copy(
                update={"enabled": False}
            )
        }
    )
    training_config = config.training.model_copy(
        update={
            "artifacts": config.training.artifacts.model_copy(
                update={"names": ("none",), "save": None}
            ),
            "diagnostics": diagnostics_config,
            "model": model_config,
            "swanlab": config.training.swanlab.model_copy(update={"enabled": False}),
            "trainer": trainer_config,
        }
    )
    return config.model_copy(update={"training": training_config})


def suggest_float_range(trial: Any, name: str, search_range: FloatSearchRangeConfig) -> float:
    return float(
        trial.suggest_float(
            name,
            search_range.low,
            search_range.high,
            log=search_range.log,
        )
    )


def set_trial_user_attr(trial: Any, name: str, value: Any) -> None:
    set_user_attr = getattr(trial, "set_user_attr", None)
    if callable(set_user_attr):
        set_user_attr(name, value)


def config_for_dataset_clusters(config: ApplicationConfig, train_data: Path) -> ApplicationConfig:
    dataset = ClinicalTimeSeriesDataset.load(train_data)
    params = dataset.metadata.get("generation_params")
    if not isinstance(params, Mapping) or "n_clusters" not in params:
        return config
    n_clusters = int(params["n_clusters"])
    return config.model_copy(
        update={
            "training": config.training.model_copy(
                update={
                    "model": config.training.model.model_copy(update={"n_clusters": n_clusters})
                }
            )
        }
    )


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


def optim_split_payload(run_paths: DatasetRunPaths) -> dict[str, Any]:
    return {
        "data": optim_existing_split_summaries(run_paths.train_data, run_paths.test_data),
        "run_id": run_paths.run_id,
        "test_data": str(run_paths.test_data),
        "train_data": str(run_paths.train_data),
    }


def optim_existing_split_summaries(
    train_data: Path,
    test_data: Path,
) -> dict[str, dict[str, Any]]:
    return {
        "train": existing_dataset_summary(
            train_data,
            default_seed=0,
        ),
        "test": existing_dataset_summary(
            test_data,
            default_seed=0,
        ),
    }


def existing_dataset_summary(
    path: Path,
    *,
    default_seed: int,
) -> dict[str, Any]:
    dataset = ClinicalTimeSeriesDataset.load(path)
    metadata_params = dataset.metadata.get("generation_params")
    if isinstance(metadata_params, Mapping):
        seed = int(metadata_params.get("sample_seed", metadata_params.get("seed", default_seed)))
        clusters = int(metadata_params.get("n_clusters", 0))
    else:
        seed = default_seed
        clusters = 0
    return simulation_summary(dataset, clusters=clusters, out=path, seed=seed)


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


def validate_optim_test_data(test_data: Path) -> None:
    dataset = ClinicalTimeSeriesDataset.load(test_data)
    if not dataset.has_cluster_labels:
        raise ValueError("command=optim requires test data with cluster_label for ARI.")


def validate_or_write_dataset_fingerprint(
    *,
    config: ApplicationConfig,
    fingerprint_path: Path,
    fingerprint: dict[str, Any],
    hydra_run_dir: Path,
) -> None:
    existing_payload = load_json_if_exists(fingerprint_path)
    has_existing_results = (hydra_run_dir / "optim_summary.json").exists() or (
        hydra_run_dir / "study.db"
    ).exists()
    if existing_payload is None:
        if config.optim.resume and has_existing_results:
            raise ValueError(
                "optim.resume=true found existing results without dataset_fingerprint.json; "
                "cannot guarantee that the study uses the same dataset."
            )
        if not config.optim.resume and has_existing_results:
            raise ValueError(
                "Optim output directory already contains results. Use optim.resume=true with "
                "the same run.name to append trials."
            )
        save_json(fingerprint_path, fingerprint)
        return

    if existing_payload != fingerprint:
        raise ValueError(
            "Optim dataset fingerprint does not match the existing run directory. "
            "Use a new run.name for a different dataset."
        )
    if not config.optim.resume:
        raise ValueError(
            "Optim output directory already contains a dataset fingerprint. Use "
            "optim.resume=true to continue this run or choose a new run.name."
        )


def load_json_if_exists(path: Path) -> Any | None:
    if not path.exists():
        return None
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def dataset_fingerprint(runs: Sequence[DatasetRunPaths]) -> dict[str, Any]:
    split_payloads = [
        {
            "run_id": run.run_id,
            "test": file_fingerprint(run.test_data),
            "train": file_fingerprint(run.train_data),
        }
        for run in runs
    ]
    digest = hashlib.sha256()
    for split in split_payloads:
        digest.update(str(split["run_id"]).encode("utf-8"))
        digest.update(str(split["train"]["sha256"]).encode("utf-8"))
        digest.update(str(split["test"]["sha256"]).encode("utf-8"))
    return {
        "digest": digest.hexdigest(),
        "splits": split_payloads,
    }


def file_fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size_bytes": stat.st_size,
    }


def optim_storage_url(storage: str | None, optim_root: Path, project_root: Path) -> str:
    if storage is None:
        return f"sqlite:///{(optim_root / 'study.db').as_posix()}"
    if "://" in storage:
        return storage
    return f"sqlite:///{resolve_path(Path(storage), project_root).as_posix()}"


def required_metric(metrics: Mapping[str, float], name: str, trial_number: int) -> float:
    if name not in metrics:
        raise ValueError(f"Trial {trial_number} did not produce required metric '{name}'.")
    value = float(metrics[name])
    if not math.isfinite(value):
        raise ValueError(f"Trial {trial_number} produced non-finite metric '{name}={value}'.")
    return value


def json_safe_metrics(metrics: Mapping[str, float]) -> dict[str, float | str]:
    payload: dict[str, float | str] = {}
    for name, value in metrics.items():
        number = float(value)
        payload[name] = number if math.isfinite(number) else str(number)
    return payload


def count_completed_trials(trials: Sequence[Any]) -> int:
    return sum(1 for trial in trials if trial_state_name(trial) == "COMPLETE")


def best_trials(study: Any) -> list[Any]:
    return list(getattr(study, "best_trials", []))


def serialize_optim_trials(trials: Sequence[Any]) -> list[dict[str, Any]]:
    return [serialize_optim_trial(trial) for trial in trials]


def serialize_optim_trial(trial: Any) -> dict[str, Any]:
    values = None if trial.values is None else [float(value) for value in trial.values]
    user_attrs = dict(trial.user_attrs)
    params = dict(trial.params)
    for name in OPTIM_PARAM_NAMES:
        if name not in params and name in user_attrs:
            params[name] = user_attrs[name]
    return {
        "datetime_complete": None
        if trial.datetime_complete is None
        else trial.datetime_complete.isoformat(timespec="seconds"),
        "datetime_start": None
        if trial.datetime_start is None
        else trial.datetime_start.isoformat(timespec="seconds"),
        "duration_seconds": trial_duration_seconds(trial),
        "number": trial.number,
        "params": params,
        "state": trial_state_name(trial),
        "user_attrs": user_attrs,
        "values": values,
    }


def trial_duration_seconds(trial: Any) -> float | None:
    if trial.datetime_start is None or trial.datetime_complete is None:
        return None
    return (trial.datetime_complete - trial.datetime_start).total_seconds()


def save_optim_trials_csv(path: Path, trials: Sequence[Any]) -> None:
    fieldnames = [
        "number",
        "state",
        "mean_cindex",
        "mean_ari",
        "std_cindex",
        "std_ari",
        "mean_objective",
        "seed",
        *OPTIM_PARAM_NAMES,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trial in trials:
            metrics = trial_metric_attrs(trial)
            row: dict[str, Any] = {
                "mean_ari": trial_objective_value(trial, 1),
                "mean_cindex": trial_objective_value(trial, 0),
                "mean_objective": metrics.get("mean_objective", ""),
                "number": trial.number,
                "seed": trial.user_attrs.get("seed", ""),
                "state": trial_state_name(trial),
                "std_ari": metrics.get("std_ari", ""),
                "std_cindex": metrics.get("std_cindex", ""),
            }
            row.update(
                {
                    name: trial.params.get(name, trial.user_attrs.get(name, ""))
                    for name in OPTIM_PARAM_NAMES
                }
            )
            writer.writerow(row)


def save_top_trials_csv(path: Path, trials: Sequence[Any], *, limit: int = 20) -> None:
    top_trials = sorted(
        (trial for trial in trials if trial_objective_values(trial) is not None),
        key=lambda trial: mean_objective(trial),
        reverse=True,
    )[:limit]
    fieldnames = [
        "rank",
        "number",
        "mean_cindex",
        "mean_ari",
        "std_cindex",
        "std_ari",
        "mean_objective",
        *OPTIM_PARAM_NAMES,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, trial in enumerate(top_trials, start=1):
            metrics = trial_metric_attrs(trial)
            row: dict[str, Any] = {
                "mean_ari": trial_objective_value(trial, 1),
                "mean_cindex": trial_objective_value(trial, 0),
                "mean_objective": metrics.get("mean_objective", mean_objective(trial)),
                "number": trial.number,
                "rank": rank,
                "std_ari": metrics.get("std_ari", ""),
                "std_cindex": metrics.get("std_cindex", ""),
            }
            row.update(
                {
                    name: trial.params.get(name, trial.user_attrs.get(name, ""))
                    for name in OPTIM_PARAM_NAMES
                }
            )
            writer.writerow(row)


def trial_metric_attrs(trial: Any) -> dict[str, float]:
    metrics = trial.user_attrs.get("metrics", {})
    if not isinstance(metrics, Mapping):
        return {}
    return {str(name): float(value) for name, value in metrics.items() if is_float_like(value)}


def trial_objective_value(trial: Any, index: int) -> float | str:
    values = trial_objective_values(trial)
    if values is None or len(values) <= index:
        return ""
    return float(values[index])


def trial_objective_values(trial: Any) -> list[float] | None:
    if trial.values is None or len(trial.values) < 2:
        return None
    return [float(value) for value in trial.values]


def mean_objective(trial: Any) -> float:
    values = trial_objective_values(trial)
    if values is None:
        return float("-inf")
    return (values[0] + values[1]) / 2.0


def trial_state_name(trial: Any) -> str:
    return str(getattr(trial.state, "name", trial.state))


def is_float_like(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def save_optim_figures(
    figures_dir: Path,
    *,
    trials: Sequence[Any],
    pareto_trials: Sequence[Any],
    selected_runs: Sequence[DatasetRunPaths],
) -> dict[str, str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "objective_history_png": str(figures_dir / "objective_history.png"),
        "pareto_png": str(figures_dir / "pareto_front.png"),
        "split_heatmap_png": str(figures_dir / "split_metric_heatmap.png"),
        "top_trials_png": str(figures_dir / "top_trials_table.png"),
    }
    plot_pareto(figures_dir / "pareto_front.png", trials, pareto_trials)
    plot_objective_history(figures_dir / "objective_history.png", trials)
    plot_split_metric_heatmap(figures_dir / "split_metric_heatmap.png", trials, selected_runs)
    plot_top_trials_table(figures_dir / "top_trials_table.png", trials)
    return figures


def plot_pareto(path: Path, trials: Sequence[Any], pareto_trials: Sequence[Any]) -> None:
    completed = [trial for trial in trials if trial_objective_values(trial) is not None]
    pareto_numbers = {int(trial.number) for trial in pareto_trials}
    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    if completed:
        x = [float(trial_objective_value(trial, 0)) for trial in completed]
        y = [float(trial_objective_value(trial, 1)) for trial in completed]
        colors = [
            "#d17aad" if int(trial.number) in pareto_numbers else "#4c78a8" for trial in completed
        ]
        ax.scatter(x, y, c=colors, s=42, alpha=0.9)
    else:
        ax.text(0.5, 0.5, "No completed trials", ha="center", va="center")
    ax.set_xlabel("Mean C-index")
    ax.set_ylabel("Mean ARI")
    ax.set_title("Optim Pareto candidates", fontweight="bold")
    ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_objective_history(path: Path, trials: Sequence[Any]) -> None:
    completed = [trial for trial in trials if trial_objective_values(trial) is not None]
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    if completed:
        numbers = [int(trial.number) for trial in completed]
        ax.plot(
            numbers,
            [float(trial_objective_value(trial, 0)) for trial in completed],
            marker="o",
            label="mean_cindex",
        )
        ax.plot(
            numbers,
            [float(trial_objective_value(trial, 1)) for trial in completed],
            marker="s",
            label="mean_ari",
        )
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "No completed trials", ha="center", va="center")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Objective value")
    ax.set_title("Optim objective history", fontweight="bold")
    ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_split_metric_heatmap(
    path: Path,
    trials: Sequence[Any],
    selected_runs: Sequence[DatasetRunPaths],
) -> None:
    completed = [trial for trial in trials if trial_objective_values(trial) is not None]
    display_trials = completed[-50:]
    run_ids = [run.run_id for run in selected_runs]
    matrix = [
        [split_mean_objective(trial, run_id) for run_id in run_ids] for trial in display_trials
    ]
    fig, ax = plt.subplots(figsize=(max(7.0, len(run_ids) * 0.55), 5.5), constrained_layout=True)
    if matrix and run_ids:
        image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        fig.colorbar(image, ax=ax, label="(C-index + ARI) / 2")
        ax.set_xticks(range(len(run_ids)))
        ax.set_xticklabels([short_run_label(run_id) for run_id in run_ids], rotation=45, ha="right")
        ax.set_yticks(range(len(display_trials)))
        ax.set_yticklabels([str(trial.number) for trial in display_trials])
    else:
        ax.text(0.5, 0.5, "No split metrics", ha="center", va="center")
    ax.set_xlabel("Split")
    ax.set_ylabel("Trial")
    ax.set_title("Split-level objective heatmap", fontweight="bold")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def split_mean_objective(trial: Any, run_id: str) -> float:
    split_metrics = trial.user_attrs.get("split_metrics", [])
    if not isinstance(split_metrics, Sequence):
        return float("nan")
    for split in split_metrics:
        if not isinstance(split, Mapping) or split.get("run_id") != run_id:
            continue
        return (float(split["cindex"]) + float(split["ari"])) / 2.0
    return float("nan")


def plot_top_trials_table(path: Path, trials: Sequence[Any]) -> None:
    top_trials = sorted(
        (trial for trial in trials if trial_objective_values(trial) is not None),
        key=lambda trial: mean_objective(trial),
        reverse=True,
    )[:10]
    fig, ax = plt.subplots(figsize=(10.0, max(2.5, 0.45 * max(1, len(top_trials)) + 1.2)))
    ax.axis("off")
    if not top_trials:
        ax.text(0.5, 0.5, "No completed trials", ha="center", va="center")
    else:
        headers = ["rank", "trial", "mean_cindex", "mean_ari", "objective"]
        rows = [
            [
                str(rank),
                str(trial.number),
                f"{float(trial_objective_value(trial, 0)):.3f}",
                f"{float(trial_objective_value(trial, 1)):.3f}",
                f"{mean_objective(trial):.3f}",
            ]
            for rank, trial in enumerate(top_trials, start=1)
        ]
        table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.25)
    ax.set_title("Top optim trials", fontweight="bold")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def short_run_label(run_id: str) -> str:
    parts = Path(run_id).parts
    label = "/".join(parts[-3:])
    if len(label) <= 32:
        return label
    return f"...{label[-29:]}"

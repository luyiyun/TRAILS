from __future__ import annotations

import csv
import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from trails.artifacts import save_json
from trails.data import ClinicalTimeSeriesDataset

from .config import (
    OPTIM_PARAM_NAMES,
    ApplicationConfig,
    FloatSearchRangeConfig,
)
from .path import DatasetRunPaths, TrainPaths, discover_dataset_runs, resolve_path
from .training import fit_training_run

LOGGER = logging.getLogger(__name__)


def run_optim_command(
    config: ApplicationConfig,
    hydra_run_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    optuna = load_optuna()
    runs = discover_dataset_runs(config, hydra_run_dir, project_root)
    selected_run, selected_index, selection_source = select_optim_run(runs, config.optim.run_id)

    validate_optim_test_data(selected_run.test_data)
    run_root = hydra_run_dir / selected_run.run_id
    run_payload = run_optim_dataset(
        optuna,
        config=config,
        run_paths=selected_run,
        run_root=run_root,
        project_root=project_root,
        seed=config.training.trainer.seed + selected_index,
        study_name=optim_study_name(config.optim.study_name, selected_run, len(runs)),
    )

    summary_path = hydra_run_dir / "optim_summary.json"
    summary = {
        "command": "optim",
        "config": config.model_dump(mode="json"),
        "hydra_run_dir": str(hydra_run_dir),
        "n_trials_requested": config.optim.n_trials,
        "outputs": {
            "summary": str(summary_path),
        },
        "paths": {
            "optim_summary": str(summary_path),
        },
        "runs": [run_payload],
        "selection": {
            "available_run_ids": [run.run_id for run in runs],
            "run_id": selected_run.run_id,
            "source": selection_source,
        },
    }

    save_json(summary_path, summary)
    return summary


def select_optim_run(
    runs: Sequence[DatasetRunPaths],
    configured_run_id: str | None,
) -> tuple[DatasetRunPaths, int, str]:
    if len(runs) == 1:
        return runs[0], 0, "single"
    if configured_run_id is not None:
        for index, run in enumerate(runs):
            if run.run_id == configured_run_id:
                return run, index, "configured"
        raise ValueError(
            "optim.run_id did not match any discovered dataset split. Available run_id values: "
            f"{format_available_run_ids(runs)}"
        )
    return interactive_select_optim_run(runs)


def interactive_select_optim_run(
    runs: Sequence[DatasetRunPaths],
) -> tuple[DatasetRunPaths, int, str]:
    LOGGER.info("Multiple dataset splits discovered for command=optim. Select one split:")
    for index, run in enumerate(runs, start=1):
        LOGGER.info(
            f"  [{index}] {run.run_id}\n      train: {run.train_data}\n      test:  {run.test_data}"
        )
    try:
        choice = input("Enter split number: ").strip()
    except EOFError as error:
        raise ValueError(
            "command=optim discovered multiple dataset splits but no interactive selection was "
            "available. Use optim.run_id=... or explicit paths.data and paths.test_data. "
            f"Available run_id values: {format_available_run_ids(runs)}"
        ) from error
    if not choice.isdigit():
        raise ValueError(f"Invalid optim split selection: {choice!r}")
    selected_index = int(choice) - 1
    if selected_index < 0 or selected_index >= len(runs):
        raise ValueError(f"Invalid optim split selection: {choice!r}")
    return runs[selected_index], selected_index, "interactive"


def format_available_run_ids(runs: Sequence[DatasetRunPaths]) -> str:
    return ", ".join(run.run_id for run in runs)


def run_optim_dataset(
    optuna: Any,
    *,
    config: ApplicationConfig,
    run_paths: DatasetRunPaths,
    run_root: Path,
    project_root: Path,
    seed: int,
    study_name: str,
) -> dict[str, Any]:
    run_root.mkdir(parents=True, exist_ok=True)
    storage_url = optim_storage_url(config.optim.storage, run_root, project_root)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        directions=["maximize", "maximize"],
        load_if_exists=True,
        sampler=sampler,
        storage=storage_url,
        study_name=study_name,
    )

    completed_before = count_completed_trials(study.trials)
    study.optimize(
        lambda trial: run_optim_trial(
            trial,
            config=config,
            optim_root=run_root,
            train_data=run_paths.train_data,
            test_data=run_paths.test_data,
            seed=seed + trial.number,
        ),
        n_trials=config.optim.n_trials,
    )
    completed_after = count_completed_trials(study.trials)

    summary_path = run_root / "optim_summary.json"
    trials_csv_path = run_root / "trials.csv"
    pareto_path = run_root / "pareto_trials.json"
    pareto_trials = serialize_optim_trials(study.best_trials)
    all_trials = serialize_optim_trials(study.trials)
    payload = {
        "data": {
            "splits": optim_existing_split_summaries(
                run_paths.train_data,
                run_paths.test_data,
            ),
            "test_data": str(run_paths.test_data),
            "train_data": str(run_paths.train_data),
        },
        "n_completed_after": completed_after,
        "n_completed_before": completed_before,
        "outputs": {
            "optim_summary": str(summary_path),
            "pareto_trials": str(pareto_path),
            "trials_csv": str(trials_csv_path),
        },
        "run_id": run_paths.run_id,
        "run_root": str(run_root),
        "seed": seed,
        "storage": storage_url,
        "study_name": study.study_name,
        "trials": all_trials,
        "pareto_trials": pareto_trials,
    }
    save_json(summary_path, payload)
    save_optim_trials_csv(trials_csv_path, study.trials)
    save_json(pareto_path, pareto_trials)
    return payload


def load_optuna() -> Any:
    try:
        import optuna
    except ImportError as error:
        raise RuntimeError(
            "command=optim requires Optuna. Install the project dev dependencies with "
            "`uv sync --group dev` before running `uv run main.py command=optim`."
        ) from error
    return optuna


def run_optim_trial(
    trial: Any,
    *,
    config: ApplicationConfig,
    optim_root: Path,
    train_data: Path,
    test_data: Path,
    seed: int,
) -> tuple[float, float]:
    trial_config = optim_trial_config(config_for_dataset_clusters(config, train_data), trial)
    train_paths = TrainPaths(
        data=train_data,
        test_data=test_data,
        train_root=optim_root / str(trial.number),
        save=None,
    )
    result = fit_training_run(
        trial_config,
        train_paths=train_paths,
        seed=seed,
        swanlab_repeat_label=None,
    )
    cindex = required_metric(result.metrics, "cindex", trial.number)
    ari = required_metric(result.metrics, "ari", trial.number)

    trial.set_user_attr("seed", seed)
    trial.set_user_attr("metrics", json_safe_metrics(result.metrics))
    trial.set_user_attr("model_config", trial_config.training.model.model_dump(mode="json"))
    trial.set_user_attr("trainer_config", trial_config.training.trainer.model_dump(mode="json"))
    return cindex, ari


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
    return config.model_copy(
        update={
            "training": training_config,
        }
    )


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


def optim_study_name(study_name: str, run_paths: DatasetRunPaths, n_runs: int) -> str:
    if n_runs == 1:
        return study_name
    return f"{study_name}-{run_paths.run_id.replace('/', '-')}"


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
    return sum(1 for trial in trials if trial.state.name == "COMPLETE")


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
        "state": trial.state.name,
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
        "cindex",
        "ari",
        "seed",
        *OPTIM_PARAM_NAMES,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trial in trials:
            row: dict[str, Any] = {
                "ari": trial_objective_value(trial, 1),
                "cindex": trial_objective_value(trial, 0),
                "number": trial.number,
                "seed": trial.user_attrs.get("seed", ""),
                "state": trial.state.name,
            }
            row.update(
                {
                    name: trial.params.get(name, trial.user_attrs.get(name, ""))
                    for name in OPTIM_PARAM_NAMES
                }
            )
            writer.writerow(row)


def trial_objective_value(trial: Any, index: int) -> float | str:
    if trial.values is None or len(trial.values) <= index:
        return ""
    return float(trial.values[index])

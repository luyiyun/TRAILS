from __future__ import annotations

import logging
from typing import Any

import hydra
from omegaconf import DictConfig

from trails.artifacts import save_json
from trails.progress import configure_tqdm_logging
from trails_simulate.config import OptimApplicationConfig, resolved_payload
from trails_simulate.optim import (
    best_trials,
    count_completed_trials,
    dataset_fingerprint,
    load_optuna,
    optim_split_payload,
    optim_storage_url,
    run_optim_study,
    save_optim_figures,
    save_optim_trials_csv,
    save_top_trials_csv,
    select_optim_runs,
    serialize_optim_trials,
    validate_optim_test_data,
    validate_or_write_dataset_fingerprint,
)
from trails_simulate.path import discover_dataset_runs
from trails_simulate.summary import format_summary

LOGGER = logging.getLogger(__name__)


def run(config: OptimApplicationConfig) -> dict[str, Any]:
    run_dir = config.paths.dir
    optuna = load_optuna()
    runs = discover_dataset_runs(config)
    selected_runs, selection_source = select_optim_runs(runs, config.optim.run_ids)
    for run_paths in selected_runs:
        validate_optim_test_data(run_paths.test_data)

    fingerprint = dataset_fingerprint(selected_runs)
    fingerprint_path = run_dir / "dataset_fingerprint.json"
    validate_or_write_dataset_fingerprint(
        config=config,
        fingerprint_path=fingerprint_path,
        fingerprint=fingerprint,
        run_dir=run_dir,
    )

    storage_url = optim_storage_url(config.optim.storage, run_dir)
    sampler = optuna.samplers.TPESampler(seed=config.trainer.seed)
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
        optim_root=run_dir,
    )
    completed_after = count_completed_trials(study.trials)

    summary_path = run_dir / "optim_summary.json"
    trials_csv_path = run_dir / "trials.csv"
    pareto_path = run_dir / "pareto_trials.json"
    top_trials_csv_path = run_dir / "top_trials.csv"
    figures_dir = run_dir / "figures"
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
        "run_dir": str(run_dir),
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


@hydra.main(config_path="../configs", config_name="optim", version_base="1.3")
def main(raw_config: DictConfig) -> None:
    configure_tqdm_logging()
    config = OptimApplicationConfig.model_validate(resolved_payload(raw_config))
    result = run(config)
    LOGGER.info(format_summary("optim", result))


if __name__ == "__main__":
    main()

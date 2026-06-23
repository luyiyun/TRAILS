from __future__ import annotations

import logging
from typing import Any

import hydra
from omegaconf import DictConfig

from trails.artifacts import save_json
from trails.progress import configure_tqdm_logging
from trails_simulate.command_utils import dataset_source_payload
from trails_simulate.config import TrainApplicationConfig, resolved_payload
from trails_simulate.evaluation import save_metrics_csv, summarize_metric_rows
from trails_simulate.path import discover_dataset_runs
from trails_simulate.summary import format_summary
from trails_simulate.train_jobs import run_train_jobs

LOGGER = logging.getLogger(__name__)


def run(config: TrainApplicationConfig) -> dict[str, Any]:
    run_dir = config.paths.dir
    runs = discover_dataset_runs(config)
    train_results = run_train_jobs(config, runs)
    metric_rows = [result.metric_row for result in train_results]
    run_payloads = [result.run_payload for result in train_results]

    summary_path = run_dir / "train_summary.json"
    metrics_csv_path = run_dir / "train_metrics.csv"
    summary = {
        "command": "train",
        "config": config.model_dump(mode="json"),
        "data_source": dataset_source_payload(config, runs),
        "run_dir": str(run_dir),
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


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(raw_config: DictConfig) -> None:
    configure_tqdm_logging()
    config = TrainApplicationConfig.model_validate(resolved_payload(raw_config))
    result = run(config)
    LOGGER.info(format_summary("train", result))


if __name__ == "__main__":
    main()

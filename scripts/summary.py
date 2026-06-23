from __future__ import annotations

import logging
from typing import Any

import hydra
import pandas as pd
from omegaconf import DictConfig

from trails.artifacts import save_json
from trails.progress import configure_tqdm_logging
from trails_simulate.config import SummaryApplicationConfig, resolved_payload
from trails_simulate.result_summary import (
    add_method_labels,
    add_run_id_fields,
    available_numeric_metrics,
    group_metric_df,
    metric_input_payload,
    read_metric_df,
    save_summary_figures,
    summary_metric_inputs,
    write_metric_df,
)
from trails_simulate.summary import format_summary

LOGGER = logging.getLogger(__name__)


def run(config: SummaryApplicationConfig) -> dict[str, Any]:
    run_dir = config.paths.dir
    inputs = summary_metric_inputs(config)
    if not inputs:
        raise ValueError("summary requires at least one train or baseline metrics root.")

    metric_df = pd.concat(
        [read_metric_df(metric_input) for metric_input in inputs],
        ignore_index=True,
    )
    metric_df, parse_warnings = add_run_id_fields(metric_df)
    metric_df = add_method_labels(metric_df)
    requested_metrics = list(config.metrics)
    available_metrics = available_numeric_metrics(metric_df)
    skipped_metrics = [metric for metric in requested_metrics if metric not in available_metrics]
    grouped_df = group_metric_df(metric_df, metrics=available_metrics)

    metrics_path = run_dir / "summary_metrics.csv"
    grouped_path = run_dir / "summary_metrics_grouped.csv"
    summary_path = run_dir / "summary_summary.json"
    figures_dir = run_dir / "figures"
    figures = save_summary_figures(
        grouped_df,
        metrics=[metric for metric in requested_metrics if metric in available_metrics],
        figures_dir=figures_dir,
    )

    write_metric_df(metrics_path, metric_df)
    write_metric_df(grouped_path, grouped_df)
    payload = {
        "command": "summary",
        "config": config.model_dump(mode="json"),
        "run_dir": str(run_dir),
        "inputs": [metric_input_payload(metric_input) for metric_input in inputs],
        "metrics": {
            "available": available_metrics,
            "requested": requested_metrics,
            "skipped": skipped_metrics,
        },
        "n_groups": len(grouped_df),
        "n_rows": len(metric_df),
        "outputs": {
            "figures": figures,
            "grouped_csv": str(grouped_path),
            "metrics_csv": str(metrics_path),
            "summary": str(summary_path),
        },
        "parse_warnings": parse_warnings,
    }
    save_json(summary_path, payload)
    return payload


@hydra.main(config_path="../configs", config_name="summary", version_base="1.3")
def main(raw_config: DictConfig) -> None:
    configure_tqdm_logging()
    config = SummaryApplicationConfig.model_validate(resolved_payload(raw_config))
    result = run(config)
    LOGGER.info(format_summary("summary", result))


if __name__ == "__main__":
    main()

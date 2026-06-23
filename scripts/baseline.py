from __future__ import annotations

import logging
from typing import Any

import hydra
from omegaconf import DictConfig

from trails.artifacts import save_json
from trails.data import ClinicalTimeSeriesDataset
from trails.progress import ProgressBar, configure_tqdm_logging
from trails_simulate.baselines import make_baseline
from trails_simulate.command_utils import (
    dataset_n_clusters,
    dataset_source_payload,
    metric_row,
)
from trails_simulate.config import BaselineApplicationConfig, resolved_payload
from trails_simulate.evaluation import (
    evaluate_predictions,
    json_safe_metrics,
    save_metrics_csv,
    save_prediction_payload,
    summarize_metric_rows,
)
from trails_simulate.path import discover_dataset_runs
from trails_simulate.summary import format_summary

LOGGER = logging.getLogger(__name__)


def run(config: BaselineApplicationConfig) -> dict[str, Any]:
    run_dir = config.paths.dir
    runs = discover_dataset_runs(config)
    metric_rows: list[dict[str, Any]] = []
    run_payloads: list[dict[str, Any]] = []

    with ProgressBar(desc="Baseline", total=len(runs) * len(config.methods)) as iter_bar:
        for index, run_paths in enumerate(runs):
            seed = config.seed + index
            train_dataset = ClinicalTimeSeriesDataset.load(run_paths.train_data)
            test_dataset = ClinicalTimeSeriesDataset.load(run_paths.test_data)
            n_clusters = config.n_clusters or dataset_n_clusters(
                train_dataset,
                fallback=config.fallback_n_clusters,
            )
            method_payloads: list[dict[str, Any]] = []
            for method in config.methods:
                baseline = make_baseline(
                    method,
                    n_clusters=n_clusters,
                    random_state=seed,
                    kmeans_iters=config.kmeans_iters,
                    ridge_alpha=config.ridge_alpha,
                    risk_feature_weight=config.risk_feature_weight,
                    fpca_components=config.fpca_components,
                    fpca_grid_size=config.fpca_grid_size,
                )
                prediction = baseline.fit(train_dataset).predict(test_dataset)
                metrics = evaluate_predictions(prediction, n_clusters=n_clusters)

                prediction_path = run_dir / run_paths.run_id / f"{method}.pt"
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

    summary_path = run_dir / "baseline_summary.json"
    metrics_csv_path = run_dir / "baseline_metrics.csv"
    summary = {
        "baseline": config.model_dump(mode="json"),
        "command": "baseline",
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


@hydra.main(config_path="../configs", config_name="baseline", version_base="1.3")
def main(raw_config: DictConfig) -> None:
    configure_tqdm_logging()
    config = BaselineApplicationConfig.model_validate(resolved_payload(raw_config))
    result = run(config)
    LOGGER.info(format_summary("baseline", result))


if __name__ == "__main__":
    main()

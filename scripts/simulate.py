from __future__ import annotations

import logging
from itertools import product
from typing import Any

import hydra
from omegaconf import DictConfig

from trails.artifacts import save_json
from trails.progress import ProgressBar, configure_tqdm_logging
from trails_simulate.config import SimulateApplicationConfig, resolved_payload
from trails_simulate.evaluation import save_metrics_csv
from trails_simulate.generators import ClinicalTimeSeriesDatasetGenerator
from trails_simulate.split_generation import (
    generator_config_for_cluster,
    simulation_mechanism_seed,
    simulation_sample_seed,
    simulation_split_summary,
)
from trails_simulate.summary import format_summary

LOGGER = logging.getLogger(__name__)


def run(config: SimulateApplicationConfig) -> dict[str, Any]:
    run_dir = config.paths.dir
    out_root = run_dir
    manifest_path = run_dir / "simulation_manifest.csv"
    summary_path = run_dir / "simulation_summary.json"
    runs: list[dict[str, Any]] = []

    n_iter = len(config.generator.n_clusters_tuple_) * len(config.train_size) * config.repeats

    with ProgressBar(desc="Simulation", total=n_iter) as iter_bar:
        for cluster_index, n_clusters in enumerate(config.generator.n_clusters_tuple_):
            # 设置不同的n_cluster
            generator_config = generator_config_for_cluster(
                config.generator,
                n_clusters=n_clusters,
            )

            # 为不同的n_cluster设置不同的seed
            mechanism_seed = simulation_mechanism_seed(config, cluster_index=cluster_index)

            generator = ClinicalTimeSeriesDatasetGenerator(
                generator_config,
                mechanism_seed=mechanism_seed,
            )

            for (size_index, (train_size, test_size)), repeat_index in product(
                enumerate(zip(config.train_size, config.test_size, strict=True)),
                range(config.repeats),
            ):
                total_patients = train_size + test_size
                repeat_seed = simulation_sample_seed(
                    config,
                    size_index=size_index,
                    cluster_index=cluster_index,
                    repeat_index=repeat_index,
                )

                run_id = f"train_{train_size}_test_{test_size}/k{n_clusters}/{repeat_index}"
                split_root = out_root / run_id
                split_root.mkdir(parents=True, exist_ok=True)

                # 每个 repeat 生成一个源数据集，再按 train/test 数量切分。
                source_dataset = generator.simulate(n_patients=total_patients, seed=repeat_seed)
                train_dataset, test_dataset = source_dataset.split_counts(
                    [train_size, test_size],
                    seed=repeat_seed,
                )
                train_path = split_root / "train.pt"
                test_path = split_root / "test.pt"
                train_dataset.save(train_path)
                test_dataset.save(test_path)

                # 记录一下每个 run 的 summary
                train_summary = simulation_split_summary(
                    train_dataset,
                    clusters=n_clusters,
                    out=train_path,
                    seed=repeat_seed,
                )
                test_summary = simulation_split_summary(
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
        "run_dir": str(run_dir),
        "outputs": {
            "manifest": str(manifest_path),
            "summary": str(summary_path),
        },
        "runs": runs,
        "simulation": config.model_dump(mode="json"),
    }
    save_metrics_csv(manifest_path, runs)
    save_json(summary_path, summary)
    return summary


@hydra.main(config_path="../configs", config_name="simulate", version_base="1.3")
def main(raw_config: DictConfig) -> None:
    configure_tqdm_logging()
    config = SimulateApplicationConfig.model_validate(resolved_payload(raw_config))
    result = run(config)
    LOGGER.info(format_summary("simulate", result))


if __name__ == "__main__":
    main()

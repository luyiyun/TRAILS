from __future__ import annotations

import time
from itertools import combinations
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.metrics import adjusted_rand_score

from trails.artifacts import save_json
from trails.config import DataConfig, TrailsConfig, resolve_batch_size
from trails.data import ClinicalTimeSeriesDataset
from trails.estimator import (
    TrailsEstimator,
    selected_k_from_selection_metrics,
    selection_metrics_to_rows,
)
from trails_case.config import CaseApplicationConfig
from trails_case.mimic import SPLIT_SEED, prepare_mimic_datasets
from trails_case.outputs import resolve_input_path, resolve_output_path
from trails_case.selection import case_k_selection_candidates
from trails_simulate.config import resolved_payload

MODEL_SEEDS = (20260517, 20260518, 20260519)
MIN_CLUSTER_FRACTION = 0.05
MIN_STABILITY_ARI = 0.75


def _run_initial_selection(
    config: TrailsConfig,
    train: ClinicalTimeSeriesDataset,
    validation: ClinicalTimeSeriesDataset,
    candidates: tuple[int, ...],
    result_dir: Path,
) -> dict[str, object]:
    metric_rows: list[dict[str, float | int]] = []
    clusters: dict[tuple[int, int], np.ndarray] = {}
    seed_winners: dict[str, int] = {}
    for seed in MODEL_SEEDS:
        seed_config = config.model_copy(
            update={"seed": seed, "trainer": config.trainer.model_copy(update={"seed": seed})}
        )
        seed_dir = result_dir / f"seed-{seed}"
        selection = TrailsEstimator(seed_config).select_n_clusters(
            train,
            candidate_clusters=candidates,
            validation_data=validation,
            result_dir=seed_dir,
        )
        seed_winners[str(seed)] = selected_k_from_selection_metrics(selection)
        metric_rows.extend({"seed": seed, **row} for row in selection_metrics_to_rows(selection))
        # 标签只在内存中用于稳定性计算，不额外导出患者级验证集结果。
        for n_clusters in candidates:
            estimator = TrailsEstimator.load(
                seed_dir / f"k{n_clusters}" / "model.pt", device=config.trainer.device
            )
            clusters[(n_clusters, seed)] = estimator.predict(validation).numpy()
    stability_rows = [
        {
            "n_clusters": n_clusters,
            "seed_a": seed_a,
            "seed_b": seed_b,
            "ari": float(
                adjusted_rand_score(clusters[(n_clusters, seed_a)], clusters[(n_clusters, seed_b)])
            ),
        }
        for n_clusters in candidates
        for seed_a, seed_b in combinations(MODEL_SEEDS, 2)
    ]
    metrics = pd.DataFrame(metric_rows).sort_values(["n_clusters", "seed"])
    stability = pd.DataFrame(stability_rows).sort_values(["n_clusters", "seed_a", "seed_b"])
    k_rows: list[dict[str, float | int | bool]] = []
    for n_clusters in candidates:
        rows = metrics.loc[metrics["n_clusters"] == n_clusters]
        scores = rows["selection_score"].astype(float)
        mean_ari = float(stability.loc[stability["n_clusters"] == n_clusters, "ari"].mean())
        min_fraction = float(rows["cluster_min_fraction"].min())
        max_empty = int(rows["cluster_empty_count"].max())
        k_rows.append(
            {
                "n_clusters": n_clusters,
                "mean_selection_score": float(scores.mean()),
                "se_selection_score": float(scores.std(ddof=1) / np.sqrt(len(scores))),
                "mean_cindex": float(rows["cindex"].mean()),
                "min_cluster_fraction": min_fraction,
                "max_empty_clusters": max_empty,
                "mean_pairwise_ari": mean_ari,
                "passes_gate": max_empty == 0
                and min_fraction >= MIN_CLUSTER_FRACTION
                and mean_ari >= MIN_STABILITY_ARI,
            }
        )
    k_summary = pd.DataFrame(k_rows).sort_values("n_clusters")
    eligible = k_summary.loc[k_summary["passes_gate"]]
    preliminary_k: int | None = None
    if not eligible.empty:
        best = eligible.loc[eligible["mean_selection_score"].idxmax()]
        cutoff = float(best["mean_selection_score"] - best["se_selection_score"])
        preliminary_k = int(
            eligible.loc[eligible["mean_selection_score"] >= cutoff, "n_clusters"].min()
        )
    unanimous = set(seed_winners.values()) == {preliminary_k}
    summary: dict[str, object] = {
        "initial_model_seeds": list(MODEL_SEEDS),
        "seed_winners": seed_winners,
        "preliminary_selected_k": preliminary_k,
        "expansion_scope": "selected_k_only" if preliminary_k and unanimous else "all_candidates",
        "gate": {
            "min_cluster_fraction": MIN_CLUSTER_FRACTION,
            "min_mean_pairwise_ari": MIN_STABILITY_ARI,
            "require_no_empty_cluster": True,
        },
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(result_dir / "initial_selection_metrics.csv", index=False)
    stability.to_csv(result_dir / "initial_stability_pairs.csv", index=False)
    k_summary.to_csv(result_dir / "initial_k_summary.csv", index=False)
    save_json(result_dir / "initial_selection_summary.json", summary)
    return summary


def run(config: CaseApplicationConfig) -> dict[str, object]:
    run_dir = config.paths.dir
    run_dir.mkdir(parents=True, exist_ok=True)
    patients_csv = resolve_input_path(config.patients_csv)
    observations_csv = resolve_input_path(config.observations_csv)
    if missing := [str(path) for path in (patients_csv, observations_csv) if not path.is_file()]:
        raise FileNotFoundError(f"缺少 MIMIC K 选择输入：{missing}")
    if config.trainer.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("MIMIC K 选择要求可用 CUDA GPU")

    started = time.perf_counter()
    datasets, _ = prepare_mimic_datasets(
        patients_csv, observations_csv, config.feature_order, config.description
    )
    train, validation = datasets["train"], datasets["validation"]
    load_seconds = time.perf_counter() - started
    trails_config = TrailsConfig(
        data=DataConfig(n_features=train.n_features),
        model=config.model,
        trainer=config.trainer.model_copy(
            update={
                "batch_size": resolve_batch_size(len(train), config.trainer.batch_size),
                "valid_size": 0.0,
            }
        ),
        seed=config.trainer.seed,
    )
    result_dir = resolve_output_path(config.k_selection.result_dir, run_dir)
    started = time.perf_counter()
    selection = _run_initial_selection(
        trails_config, train, validation, case_k_selection_candidates(config), result_dir
    )
    summary: dict[str, object] = {
        "data": {
            "split_seed": SPLIT_SEED,
            "split_sizes": {name: len(data) for name, data in datasets.items()},
            "n_features": train.n_features,
        },
        "resources": {
            "device": config.trainer.device,
            "batch_size": trails_config.trainer.batch_size,
            "load_seconds": load_seconds,
            "training_seconds": time.perf_counter() - started,
        },
        "training": trails_config.model_dump(mode="json"),
        "k_selection": selection,
        "sealed_test_evaluated": False,
    }
    save_json(resolve_output_path(config.outputs.summary, run_dir), summary)
    return summary


@hydra.main(config_path="../configs", config_name="mimic_select_k", version_base="1.3")
def main(raw_config: DictConfig) -> None:
    run(CaseApplicationConfig.model_validate(resolved_payload(raw_config)))


if __name__ == "__main__":
    main()

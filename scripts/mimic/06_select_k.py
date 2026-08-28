from __future__ import annotations

import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from trails import (
    ClinicalTimeSeriesDataset,
    ClusterNumberSelector,
    DataConfig,
    TrailsConfig,
    resolve_batch_size,
)
from trails.artifacts import save_json
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
    result = ClusterNumberSelector(
        candidates=candidates,
        seeds=MODEL_SEEDS,
        split_seed=SPLIT_SEED,
        selection_rule="one_standard_error",
        require_non_empty=True,
        min_cluster_fraction=MIN_CLUSTER_FRACTION,
        min_mean_pairwise_ari=MIN_STABILITY_ARI,
        estimator_config=config,
    ).select(train, validation_data=validation)
    result.save(result_dir)

    preliminary_k = result.selected_k
    unanimous = preliminary_k is not None and set(result.seed_winners.values()) == {preliminary_k}
    summary: dict[str, object] = {
        "initial_model_seeds": list(MODEL_SEEDS),
        "seed_winners": result.seed_winners,
        "preliminary_selected_k": preliminary_k,
        "expansion_scope": (
            "selected_k_only" if preliminary_k is not None and unanimous else "all_candidates"
        ),
        "gate": {
            "min_cluster_fraction": MIN_CLUSTER_FRACTION,
            "min_mean_pairwise_ari": MIN_STABILITY_ARI,
            "require_no_empty_cluster": True,
        },
    }
    save_json(result_dir / "initial_decision.json", summary)
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


@hydra.main(config_path="../../configs", config_name="mimic_select_k", version_base="1.3")
def main(raw_config: DictConfig) -> None:
    run(CaseApplicationConfig.model_validate(resolved_payload(raw_config)))


if __name__ == "__main__":
    main()

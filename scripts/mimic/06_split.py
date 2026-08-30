from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import hydra
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from .data import prepare_mimic_datasets


def run(config: DictConfig) -> None:
    patients_csv = Path(str(config.paths.patients_csv)).resolve()
    observations_csv = Path(str(config.paths.observations_csv)).resolve()
    output_root = Path(str(config.paths.dir)).resolve()
    split_seeds = tuple(int(seed) for seed in config.split.seeds)
    feature_order = tuple(str(feature) for feature in config.feature_order)
    description = str(config.description)
    fractions = {
        "train": float(config.split.train_fraction),
        "validation": float(config.split.validation_fraction),
        "test": float(config.split.test_fraction),
    }
    if missing := [str(path) for path in (patients_csv, observations_csv) if not path.is_file()]:
        raise FileNotFoundError(f"缺少 MIMIC 特征输入：{missing}")
    if not split_seeds:
        raise ValueError("split.seeds 不能为空")
    if len(set(split_seeds)) != len(split_seeds):
        raise ValueError("split.seeds 不能包含重复值")
    if any(fraction <= 0 or fraction >= 1 for fraction in fractions.values()):
        raise ValueError("train、validation 和 test 比例必须位于 0 与 1 之间")
    if not math.isclose(sum(fractions.values()), 1.0):
        raise ValueError("train、validation 和 test 比例之和必须为 1")

    patients = pd.read_csv(patients_csv, dtype={"patient_id": str})
    required = {"patient_id", "event", "left_icu_before_48h"}
    if missing := sorted(required - set(patients.columns)):
        raise ValueError(f"patients.csv 缺少正式划分字段：{missing}")
    has_missing_ids = bool(patients["patient_id"].isna().to_numpy().any())
    has_duplicate_ids = bool(patients["patient_id"].duplicated().to_numpy().any())
    if has_missing_ids or has_duplicate_ids:
        raise ValueError("patients.csv 的 patient_id 必须完整且唯一")

    strata = patients["event"].astype(str) + ":" + patients["left_icu_before_48h"].astype(str)
    input_sha256 = hashlib.sha256(patients_csv.read_bytes()).hexdigest()
    for seed in split_seeds:
        first_split = train_test_split(
            patients,
            test_size=fractions["test"],
            random_state=seed,
            stratify=strata,
        )
        train_validation = cast(pd.DataFrame, first_split[0])
        test = cast(pd.DataFrame, first_split[1])
        train_validation_strata = (
            train_validation["event"].astype(str)
            + ":"
            + train_validation["left_icu_before_48h"].astype(str)
        )
        second_split = train_test_split(
            train_validation,
            test_size=fractions["validation"] / (fractions["train"] + fractions["validation"]),
            random_state=seed,
            stratify=train_validation_strata,
        )
        splits = {
            "train": cast(pd.DataFrame, second_split[0]),
            "validation": cast(pd.DataFrame, second_split[1]),
            "test": test,
        }

        all_ids = set(patients["patient_id"])
        split_ids = {name: set(frame["patient_id"]) for name, frame in splits.items()}
        if set().union(*split_ids.values()) != all_ids:
            raise RuntimeError("划分后的 patient_id 并集与输入队列不一致")
        if any(
            split_ids[left] & split_ids[right]
            for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
        ):
            raise RuntimeError("train、validation 和 test patient_id 存在重叠")

        seed_root = output_root / f"seed-{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        for name, frame in splits.items():
            frame.loc[:, ["patient_id"]].sort_values(by=["patient_id"]).to_csv(
                seed_root / f"{name}_ids.csv",
                index=False,
            )

        # 固定ID后立即拟合train预处理并保存tensor dataset，训练脚本不再重复解析长表。
        datasets, transformer = prepare_mimic_datasets(
            patients_csv,
            observations_csv,
            seed_root,
            seed,
            feature_order,
            description,
        )
        for name, dataset in datasets.items():
            dataset.save(seed_root / name / "dataset.pt")
        assert transformer.parameters_ is not None
        transformer.parameters_.to_csv(seed_root / "preprocessing_parameters.csv", index=False)

        manifest = {
            "dataset": "MIMIC-IV v3.1 primary early Sepsis-3 cohort",
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "input_patients_csv": str(patients_csv),
            "input_sha256": input_sha256,
            "split_seed": seed,
            "stratification": ["event", "left_icu_before_48h"],
            "fractions": fractions,
            "n_patients": len(patients),
            "split_counts": {name: len(frame) for name, frame in splits.items()},
            "id_columns": ["patient_id"],
            "feature_order": datasets["train"].feature_names,
            "preprocessing": "train-only 1/99% winsorization and standardization",
            "outputs": [
                *[f"{name}_ids.csv" for name in splits],
                *[f"{name}/dataset.pt" for name in splits],
                "preprocessing_parameters.csv",
            ],
        }
        (seed_root / "split_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"split seed {seed} 已保存至 {seed_root}: {manifest['split_counts']}")


@hydra.main(config_path="../../configs", config_name="mimic/split", version_base="1.3")
def main(config: DictConfig) -> None:
    run(config)


if __name__ == "__main__":
    main()

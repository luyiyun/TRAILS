from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.utils.preprocessing import Winsorizer
from trails import ClinicalTimeSeriesDataset

from .config import BASELINE_COVARIATE_COLUMNS


def prepare_mimic_datasets(
    patients_csv: Path,
    observations_csv: Path,
    split_dir: Path,
    split_seed: int,
    feature_order: tuple[str, ...],
    description: str,
) -> tuple[dict[str, ClinicalTimeSeriesDataset], pd.DataFrame]:
    """按冻结ID生成三套张量，用train拟合矩阵Pipeline并返回参数表。"""
    patients = pd.read_csv(patients_csv, dtype={"patient_id": str})
    observations = pd.read_csv(observations_csv, dtype={"patient_id": str})
    # 一、检查源表及特征名单；源观测必须有限，NaN仅由后续宽表透视产生。
    observation_columns = ["patient_id", "time", "feature", "value"]
    if missing := sorted(set(observation_columns) - set(observations.columns)):
        raise ValueError(f"observations.csv缺少字段：{missing}")
    if (
        observations[observation_columns].isna().to_numpy().any()
        or not np.isfinite(observations[["time", "value"]].to_numpy(dtype=float)).all()
    ):
        raise ValueError("源观测的ID、特征、时间和值必须完整，数值必须有限")
    resolved_feature_order = feature_order or tuple(
        str(feature) for feature in observations["feature"].drop_duplicates()
    )
    if not resolved_feature_order or len(set(resolved_feature_order)) != len(
        resolved_feature_order
    ):
        raise ValueError("特征名单须非空且不重复")
    # 与load_from_csv的use_features一致，仅转换请求的纵向输入。
    observations = observations.loc[observations["feature"].isin(resolved_feature_order)]
    required = {
        "patient_id",
        "survival_time",
        "event",
        "left_icu_before_48h",
        *BASELINE_COVARIATE_COLUMNS,
    }
    if missing := sorted(required - set(patients.columns)):
        raise ValueError(f"patients.csv 缺少正式划分字段：{missing}")
    if bool(patients["patient_id"].duplicated().to_numpy().any()):
        raise ValueError("patients.csv 的 patient_id 必须唯一")

    # 二、复核冻结ID并关联患者信息，保持原有患者顺序。
    id_frames: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "validation", "test"):
        ids_path = split_dir / f"{split_name}_ids.csv"
        if not ids_path.is_file():
            raise FileNotFoundError(f"缺少正式划分文件：{ids_path}")
        ids = pd.read_csv(ids_path, dtype={"patient_id": str})
        if list(ids.columns) != ["patient_id"]:
            raise ValueError(f"{ids_path} 必须只包含 patient_id 列")
        id_frames[split_name] = ids

    combined_ids = pd.concat(id_frames.values(), ignore_index=True)
    if bool(combined_ids["patient_id"].duplicated().to_numpy().any()):
        raise ValueError("train、validation 和 test patient_id 存在重复")
    if set(combined_ids["patient_id"]) != set(patients["patient_id"]):
        raise ValueError("正式划分 patient_id 并集与 patients.csv 不一致")
    patient_splits = {
        name: ids.merge(patients, on="patient_id", how="left", validate="one_to_one", sort=False)
        for name, ids in id_frames.items()
    }

    # 三、直接用多特征Pipeline；各split依次透视，避免同时常驻三套宽矩阵。
    winsorizer, scaler = Winsorizer(0.01, 0.99), StandardScaler()
    pipeline = Pipeline([("winsorize", winsorizer), ("scale", scaler)])
    parameters = pd.DataFrame()
    datasets: dict[str, ClinicalTimeSeriesDataset] = {}
    with TemporaryDirectory(prefix="trails-mimic-splits-") as temporary:
        temporary_root = Path(temporary)
        for split_name, split_patients in patient_splits.items():
            selected = observations.loc[
                observations["patient_id"].isin(split_patients["patient_id"])
            ].copy()
            matrix = selected.pivot(
                index=["patient_id", "time"], columns="feature", values="value"
            ).reindex(columns=list(resolved_feature_order))
            if split_name == "train":
                reasons = pd.Series("", index=matrix.columns)
                reasons.loc[matrix.count() == 0] = "no_train_observations"
                reasons.loc[matrix.min() == matrix.max()] = "constant_in_train"
                if (reasons != "").any():
                    raise ValueError(f"train存在退化指标：{reasons.loc[reasons != ''].to_dict()}")
                converted = np.asarray(pipeline.fit_transform(matrix), dtype=np.float64)
                constant = np.asarray(scaler.var_) <= 0
                if constant.any():
                    raise ValueError(f"训练集截尾后没有变异：{matrix.columns[constant].tolist()}")
                parameters = pd.DataFrame(
                    {
                        "feature": resolved_feature_order,
                        "lower": winsorizer.lower_,
                        "upper": winsorizer.upper_,
                        "center": scaler.mean_,
                        "scale": scaler.scale_,
                    }
                ).sort_values("feature", ignore_index=True)
            else:
                converted = np.asarray(pipeline.transform(matrix), dtype=np.float64)
            row_positions = matrix.index.get_indexer(
                pd.MultiIndex.from_frame(selected[["patient_id", "time"]])
            )
            column_positions = matrix.columns.get_indexer(selected["feature"])
            selected["value"] = converted[row_positions, column_positions]
            with np.errstate(over="ignore", invalid="ignore"):
                finite = np.isfinite(selected["value"].to_numpy(dtype=np.float32))
            if not finite.all():
                raise ValueError("预处理后的实际观测无法表示为有限float32张量")
            del matrix, converted
            patients_path = temporary_root / f"{split_name}_patients.csv"
            observations_path = temporary_root / f"{split_name}_observations.csv"
            split_patients.to_csv(patients_path, index=False)
            selected.to_csv(observations_path, index=False)
            datasets[split_name] = ClinicalTimeSeriesDataset.load_from_csv(
                patients_csv=patients_path,
                observations_csv=observations_path,
                use_features=resolved_feature_order,
                description=f"{description} ({split_name})",
                metadata={
                    "split_name": split_name,
                    "split_seed": split_seed,
                    "split_ids_csv": str(split_dir / f"{split_name}_ids.csv"),
                    "baseline_covariates": split_patients.loc[
                        :, ["patient_id", *BASELINE_COVARIATE_COLUMNS]
                    ].to_dict(orient="records"),
                },
            )
    return datasets, parameters


def run(config: DictConfig) -> None:
    patients_csv = Path(str(config.paths.patients_csv)).resolve()
    observations_csv = Path(str(config.paths.observations_csv)).resolve()
    output_root = Path(str(config.paths.dir)).resolve()
    strategy = str(config.split.strategy)
    split_seeds = tuple(int(seed) for seed in config.split.seeds)
    feature_order = tuple(str(feature) for feature in config.feature_order)
    description = str(config.description)
    if strategy not in {"random", "temporal"}:
        raise ValueError("split.strategy 必须是 random 或 temporal")
    if strategy == "random":
        fractions = {
            "train": float(config.split.random.train_fraction),
            "validation": float(config.split.random.validation_fraction),
            "test": float(config.split.random.test_fraction),
        }
        validation_fraction = fractions["validation"] / (
            fractions["train"] + fractions["validation"]
        )
        test_start_year = None
    else:
        fractions = None
        validation_fraction = float(config.split.temporal.validation_fraction)
        test_start_year = int(config.split.temporal.test_start_year)
    if missing := [str(path) for path in (patients_csv, observations_csv) if not path.is_file()]:
        raise FileNotFoundError(f"缺少 MIMIC 特征输入：{missing}")
    if not split_seeds:
        raise ValueError("split.seeds 不能为空")
    if len(set(split_seeds)) != len(split_seeds):
        raise ValueError("split.seeds 不能包含重复值")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation比例必须位于0与1之间")
    if fractions is not None:
        if any(fraction <= 0 or fraction >= 1 for fraction in fractions.values()):
            raise ValueError("train、validation 和 test 比例必须位于 0 与 1 之间")
        if not math.isclose(sum(fractions.values()), 1.0):
            raise ValueError("train、validation 和 test 比例之和必须为 1")

    patients = pd.read_csv(patients_csv, dtype={"patient_id": str})
    required = {"patient_id", "event", "left_icu_before_48h"}
    if strategy == "temporal":
        required.add("anchor_year_group")
    if missing := sorted(required - set(patients.columns)):
        raise ValueError(f"patients.csv 缺少正式划分字段：{missing}")
    has_missing_ids = bool(patients["patient_id"].isna().to_numpy().any())
    has_duplicate_ids = bool(patients["patient_id"].duplicated().to_numpy().any())
    if has_missing_ids or has_duplicate_ids:
        raise ValueError("patients.csv 的 patient_id 必须完整且唯一")

    strata = patients["event"].astype(str) + ":" + patients["left_icu_before_48h"].astype(str)
    temporal_groups: dict[str, int] | None = None
    development: pd.DataFrame | None = None
    temporal_test: pd.DataFrame | None = None
    if strategy == "temporal":
        extracted = (
            patients["anchor_year_group"]
            .astype("string")
            .str.extract(r"^\s*(\d{4})\s*-\s*\d{4}\s*$", expand=False)
        )
        if bool(extracted.isna().to_numpy().any()):
            invalid = sorted(
                patients.loc[extracted.isna(), "anchor_year_group"].astype(str).unique()
            )
            raise ValueError(f"anchor_year_group 格式无效：{invalid}")
        group_start_year = extracted.astype(int)
        assert test_start_year is not None
        development_frame = patients.loc[group_start_year < test_start_year]
        temporal_test_frame = patients.loc[group_start_year >= test_start_year]
        if development_frame.empty or temporal_test_frame.empty:
            raise ValueError("年代截点必须同时产生非空开发队列和测试集")
        development = development_frame
        temporal_test = temporal_test_frame
        temporal_groups = {
            str(group): int(count)
            for group, count in patients["anchor_year_group"].value_counts().sort_index().items()
        }
    input_sha256 = hashlib.sha256(patients_csv.read_bytes()).hexdigest()
    strategy_root = (
        output_root if strategy == "random" else output_root / f"temporal-{test_start_year}"
    )
    seed_roots = {seed: strategy_root / f"seed-{seed}" for seed in split_seeds}
    if existing := [str(path) for path in seed_roots.values() if path.exists()]:
        raise FileExistsError(f"拒绝覆盖既有冻结划分：{existing}")

    for seed in split_seeds:
        if strategy == "random":
            assert fractions is not None
            first_split = train_test_split(
                patients,
                test_size=fractions["test"],
                random_state=seed,
                stratify=strata,
            )
            train_validation = cast(pd.DataFrame, first_split[0])
            test = cast(pd.DataFrame, first_split[1])
        else:
            assert development is not None and temporal_test is not None
            train_validation = development
            test = temporal_test
        train_validation_strata = (
            train_validation["event"].astype(str)
            + ":"
            + train_validation["left_icu_before_48h"].astype(str)
        )
        second_split = train_test_split(
            train_validation,
            test_size=validation_fraction,
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

        seed_root = seed_roots[seed]
        seed_root.mkdir(parents=True, exist_ok=False)
        for name, frame in splits.items():
            frame.loc[:, ["patient_id"]].sort_values(by=["patient_id"]).to_csv(
                seed_root / f"{name}_ids.csv",
                index=False,
            )

        # 固定ID后立即拟合train预处理并保存tensor dataset，训练脚本不再重复解析长表。
        datasets, parameters = prepare_mimic_datasets(
            patients_csv,
            observations_csv,
            seed_root,
            seed,
            feature_order,
            description,
        )
        for name, dataset in datasets.items():
            dataset.metadata.update(
                {
                    "split_strategy": strategy,
                    "temporal_test_start_year": test_start_year,
                }
            )
            dataset.save(seed_root / name / "dataset.pt")
        parameters.to_csv(seed_root / "preprocessing_parameters.csv", index=False)

        manifest = {
            "dataset": "MIMIC-IV v3.1 primary early Sepsis-3 cohort",
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "input_patients_csv": str(patients_csv),
            "input_sha256": input_sha256,
            "split_strategy": strategy,
            "split_seed": seed,
            "train_test_rule": (
                "stratified random"
                if strategy == "random"
                else f"anchor_year_group start year >= {test_start_year}"
            ),
            "validation_rule": "development-cohort stratified random",
            "stratification": ["event", "left_icu_before_48h"],
            "fractions": fractions,
            "temporal": (
                {
                    "test_start_year": test_start_year,
                    "validation_fraction": validation_fraction,
                    "anchor_year_group_counts": temporal_groups,
                }
                if strategy == "temporal"
                else None
            ),
            "n_patients": len(patients),
            "split_counts": {name: len(frame) for name, frame in splits.items()},
            "realized_fractions": {
                name: len(frame) / len(patients) for name, frame in splits.items()
            },
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

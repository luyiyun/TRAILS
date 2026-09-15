from __future__ import annotations

import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from scripts.crc_yunnan.config import CRCSplitConfig
from scripts.utils.preprocessing import (
    AutoLog1pTransformer,
    RobustScalerWithStdFallback,
    save_scaler,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from trails import ClinicalTimeSeriesDataset

LOGGER = logging.getLogger(__name__)
SPLIT_NAMES = ("train", "validation", "test")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    path.chmod(0o600)


def _check_ids(splits: dict[str, pd.DataFrame], expected: set[str]) -> None:
    combined = cast(
        pd.DataFrame,
        pd.concat([frame[["patient_id"]] for frame in splits.values()], ignore_index=True),
    )
    if any(frame.empty for frame in splits.values()):
        raise ValueError("train、validation、test均须非空")
    if (
        combined["patient_id"].isna().to_numpy().any()
        or combined["patient_id"].duplicated().to_numpy().any()
    ):
        raise ValueError("三套划分的患者ID必须完整且互斥")
    if set(combined["patient_id"]) != expected:
        raise ValueError("三套划分的患者ID并集与目标队列不一致")


def _read_data(config: CRCSplitConfig) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """读取01的标准产物；保留基础纳排，不重复审计结局构造和附加列。"""
    patients = pd.read_csv(config.paths.patients_csv, dtype={"patient_id": "string"})
    observation_columns = ["patient_id", "time_days", "feature", "value"]
    observations = pd.read_csv(
        config.paths.observations_csv,
        usecols=observation_columns,
        dtype={
            "patient_id": "string",
            "feature": "string",
            "time_days": "float64",
            "value": "float64",
        },
    )
    dates = ["surgery_date", "last_followup_date"]
    patients[dates] = patients[dates].apply(pd.to_datetime)
    numeric = ["dfs_time_days", "os_time_days", "dfs_event", "os_event", "surgery_year"]
    if (
        patients[["patient_id", *dates]].isna().to_numpy().any()
        or not np.isfinite(patients[numeric].to_numpy(dtype=float)).all()
    ):
        raise ValueError("患者划分所需字段须完整且数值有限")
    if (
        observations.isna().to_numpy().any()
        or not np.isfinite(observations[["time_days", "value"]].to_numpy()).all()
    ):
        raise ValueError("实际纵向观测须完整且数值有限")
    excluded = (patients["last_followup_date"] < patients["surgery_date"]) & bool(
        config.cohort.exclude_followup_before_surgery
    )
    if config.cohort.max_outcome_days is not None:
        excluded |= (
            patients[["dfs_time_days", "os_time_days"]]
            .gt(float(config.cohort.max_outcome_days))
            .any(axis=1)
        )
    base = patients.loc[~excluded].sort_values("patient_id").copy()
    if base.empty:
        raise ValueError("基础纳排后没有患者")
    baseline_columns = [
        name
        for name in patients.columns
        if name not in {"patient_id", "recurrence", *dates, *numeric}
    ]
    LOGGER.info("源患者=%d，基础队列=%d", len(patients), len(base))
    return base, observations, baseline_columns


def _master_split(
    config: CRCSplitConfig,
    base: pd.DataFrame,
    bundle_paths: dict[tuple[str, int], Path],
) -> dict[tuple[str, int], dict[str, pd.DataFrame]]:
    """创建或复用患者ID，核对划分规则和基础队列，不审计文件哈希。"""
    assignment_root = Path(str(config.paths.assignment_root)).resolve()
    master_splits: dict[tuple[str, int], dict[str, pd.DataFrame]] = {}
    for (strategy, seed), bundle in bundle_paths.items():
        master_dir = assignment_root / bundle.parent.name / bundle.name
        manifest_path = master_dir / "assignment_manifest.json"
        # 只比较影响主划分的规则；旧manifest的哈希等额外字段不参与复用判断。
        rules = {
            "cohort": {
                "exclude_followup_before_surgery": bool(
                    config.cohort.exclude_followup_before_surgery
                ),
                "max_outcome_days": config.cohort.max_outcome_days,
            },
            "strategy": strategy,
            "seed": seed,
            "stratification": ["dfs_event", "os_event"],
            "random": {
                name: float(getattr(config.split.random, f"{name}_fraction"))
                for name in SPLIT_NAMES
            }
            if strategy == "random"
            else None,
            "temporal": config.split.temporal.model_dump(mode="json")
            if strategy == "temporal"
            else None,
        }
        reused = master_dir.exists()
        if reused:
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))["contract"]
            if any(saved[key] != value for key, value in rules.items()):
                raise ValueError(f"主划分规则不匹配，请使用新版本：{master_dir}")
            id_frames = {
                name: pd.read_csv(master_dir / f"{name}_ids.csv", dtype={"patient_id": "string"})
                for name in SPLIT_NAMES
            }
            _check_ids(id_frames, set(base["patient_id"]))
            splits = {
                name: ids[["patient_id"]].merge(base, on="patient_id", validate="one_to_one")
                for name, ids in id_frames.items()
            }
        else:
            if strategy == "random":
                development, test = train_test_split(
                    base,
                    test_size=float(config.split.random.test_fraction),
                    random_state=seed,
                    stratify=base["dfs_event"].astype(str) + ":" + base["os_event"].astype(str),  # type: ignore
                )
                development, test = cast(pd.DataFrame, development), cast(pd.DataFrame, test)
                validation_fraction = float(config.split.random.validation_fraction) / (
                    float(config.split.random.train_fraction)
                    + float(config.split.random.validation_fraction)
                )
            else:
                cutoff = int(config.split.temporal.test_start_year)
                development, test = (
                    base.loc[base["surgery_year"] < cutoff],
                    base.loc[base["surgery_year"] >= cutoff],
                )
                validation_fraction = float(config.split.temporal.validation_fraction)
            train, validation = train_test_split(
                development,
                test_size=validation_fraction,
                random_state=seed,
                stratify=development["dfs_event"].astype(str)  # type: ignore
                + ":"
                + development["os_event"].astype(str),
            )
            splits = {
                "train": cast(pd.DataFrame, train),
                "validation": cast(pd.DataFrame, validation),
                "test": test,
            }
            _check_ids(splits, set(base["patient_id"]))
        if strategy == "temporal" and set(splits["test"]["patient_id"]) != set(
            base.loc[
                base["surgery_year"] >= int(config.split.temporal.test_start_year), "patient_id"
            ]
        ):
            raise ValueError("temporal测试ID不符合手术年份切点")
        if not reused:
            master_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
            for name, frame in splits.items():
                path = master_dir / f"{name}_ids.csv"
                frame[["patient_id"]].sort_values("patient_id").to_csv(path, index=False)
                path.chmod(0o600)
            _write_json(manifest_path, {"contract": rules})
        master_splits[strategy, seed] = splits
    return master_splits


def _get_cohort_by_landmark(
    base: pd.DataFrame,
    observations: pd.DataFrame,
    feature_order: list[str],
    outcome: str,
    landmark_days: float,
    config: CRCSplitConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按landmark和完整面板筛选一次观察日期，不自动删特征或再次收缩队列。"""
    landmark_patients = base.loc[base[f"{outcome}_time_days"] > landmark_days].copy()
    landmark_patients["survival_time"] = landmark_patients[f"{outcome}_time_days"] - landmark_days
    landmark_patients["event"] = landmark_patients[f"{outcome}_event"]
    window_observations = observations.loc[
        observations["time_days"].between(0, landmark_days)
        & observations["feature"].isin(feature_order)
        & observations["patient_id"].isin(landmark_patients["patient_id"])
    ]
    date_counts = window_observations.groupby("patient_id")["time_days"].nunique()
    cohort = landmark_patients.loc[
        landmark_patients["patient_id"].isin(
            date_counts.index[date_counts >= int(config.cohort.min_observation_dates)]
        )
    ]
    cohort_observations = window_observations.loc[
        window_observations["patient_id"].isin(cohort["patient_id"])
    ]
    return cohort, cohort_observations


def _save_split(
    bundle: Path,
    name: str,
    patients: pd.DataFrame,
    observations: pd.DataFrame,
    feature_order: list[str],
    baseline_columns: list[str],
    metadata: dict[str, Any],
) -> None:
    """保存一套患者表、ID和张量。"""
    split_dir = bundle / name
    split_dir.mkdir(mode=0o700)
    patients[["patient_id"]].to_csv(bundle / f"{name}_ids.csv", index=False)
    patient_output = split_dir / "patients.csv"
    patients.drop(columns="split").to_csv(patient_output, index=False)
    selected = observations.loc[observations["patient_id"].isin(patients["patient_id"])]
    baseline = cast(pd.DataFrame, patients[["patient_id", *baseline_columns]].astype(object))
    baseline = baseline.where(pd.notna(baseline), None)
    with TemporaryDirectory(prefix="tensor-input-", dir=bundle) as temporary:
        observation_output = Path(temporary) / "observations.csv"
        selected.to_csv(observation_output, index=False)
        dataset = ClinicalTimeSeriesDataset.load_from_csv(
            patients_csv=patient_output,
            observations_csv=observation_output,
            time_col="time_days",
            use_features=feature_order,
            description=(
                f"CRC Yunnan {metadata['outcome'].upper()} "
                f"landmark {metadata['landmark_days']:g} days ({name})"
            ),
            metadata={
                **metadata,
                "split_name": name,
                "baseline_covariates": baseline.to_dict(orient="records"),
            },
        )
    if dataset.metadata["patient_ids"] != patients["patient_id"].tolist():
        raise RuntimeError("张量患者顺序与ID及评价表不一致")
    dataset.save(split_dir / "dataset.pt")


def run(config: CRCSplitConfig) -> None:
    # ==================================================================================
    # 一、解析配置、检查新输出目录，读取源表并完成基础纳排。
    # ==================================================================================
    feature_order = config.features.panels[config.features.panel]
    outcome, panel = str(config.outcome), str(config.features.panel)
    landmark_days = float(config.landmark_months) * 30.0
    output_root = Path(str(config.paths.dir)).resolve()
    bundle_paths = {
        (str(strategy), int(seed)): output_root
        / (
            "random"
            if strategy == "random"
            else f"temporal-{int(config.split.temporal.test_start_year)}"
        )
        / f"seed-{int(seed)}"
        for strategy in config.split.strategies
        for seed in config.split.seeds
    }
    if existing := [str(path) for path in bundle_paths.values() if path.exists()]:
        raise FileExistsError(f"拒绝覆盖既有冻结产物：{existing}")
    base, observations, baseline_columns = _read_data(config)

    # ==================================================================================
    # 二、创建或复用患者主划分；后续更换窗口、面板或预处理时保持归属不变。
    # ==================================================================================
    master_splits = _master_split(config, base, bundle_paths)

    # ==================================================================================
    # 三、按landmark和完整面板筛选一次观察日期，不自动删特征或再次收缩队列。
    # ==================================================================================
    cohort, cohort_observations = _get_cohort_by_landmark(
        base, observations, feature_order, outcome, landmark_days, config
    )
    if cohort.empty:
        raise ValueError("landmark与面板观测日期筛选后没有患者")

    for (strategy, seed), master in master_splits.items():
        splits = {
            name: cohort.loc[cohort["patient_id"].isin(master[name]["patient_id"])]
            .sort_values("patient_id")
            .assign(split=name)
            for name in SPLIT_NAMES
        }
        _check_ids(splits, set(cohort["patient_id"]))

        # ==============================================================================
        # 四、只用最终train原始观测选择log1p并拟合缩放；三套数据应用同一参数。
        # ==============================================================================
        train_observations = cohort_observations.loc[
            cohort_observations["patient_id"].isin(splits["train"]["patient_id"])
        ]
        train_matrix = train_observations.pivot(
            index=["patient_id", "time_days"], columns="feature", values="value"
        ).reindex(columns=feature_order)
        reasons = pd.Series("", index=train_matrix.columns)
        reasons.loc[train_matrix.count() == 0] = "no_train_observations"
        reasons.loc[train_matrix.min() == train_matrix.max()] = "constant_in_train"
        if (reasons != "").any():
            raise ValueError(f"固定面板在train存在退化指标：{reasons.loc[reasons != ''].to_dict()}")

        # 在train数据上应用特征预处理
        scalers = {
            "none": "passthrough",
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScalerWithStdFallback(),
        }
        log_transformer = (
            AutoLog1pTransformer(skew_threshold=float(config.preprocessing.skew_threshold))
            if config.preprocessing.log_transform == "auto-log1p"
            else "passthrough"
        )
        scaler = scalers[str(config.preprocessing.scaling)]
        pipeline = Pipeline([("log", log_transformer), ("scale", scaler)]).fit(train_matrix)
        del train_matrix

        transformed = cast(
            pd.DataFrame,
            cohort_observations[["patient_id", "time_days", "feature", "value"]].copy(),
        )
        for frame in splits.values():
            selected = cohort_observations.loc[
                cohort_observations["patient_id"].isin(frame["patient_id"])
            ]
            matrix_i = selected.pivot(
                index=["patient_id", "time_days"], columns="feature", values="value"
            ).reindex(columns=feature_order)
            converted = np.asarray(pipeline.transform(matrix_i), dtype=np.float64)
            row_positions = matrix_i.index.get_indexer(
                pd.MultiIndex.from_frame(selected[["patient_id", "time_days"]])
            )
            column_positions = matrix_i.columns.get_indexer(selected["feature"])  # type: ignore
            values = converted[row_positions, column_positions]

            # 检查是否存在infinity
            with np.errstate(over="ignore", invalid="ignore"):
                finite = np.isfinite(values.astype(np.float32))
            if not finite.all():
                features = selected.loc[~finite, "feature"].unique().tolist()
                raise ValueError(f"预处理结果无法表示为有限float32张量：{features}")

            transformed.loc[selected.index, "value"] = values
            del matrix_i, converted

        # ==============================================================================
        # 五、保存三套冻结数据及必要参数；患者数和事件数仅记录一次。
        # ==============================================================================
        bundle = bundle_paths[strategy, seed]
        bundle.mkdir(parents=True, exist_ok=False, mode=0o700)

        # 保存scaler的相关参数
        save_scaler(bundle / "preprocessing_parameters.csv", scaler, log_transformer, feature_order)

        OmegaConf.save(
            OmegaConf.create(config.model_dump(mode="json")), bundle / "resolved_config.yaml"
        )
        for name, frame in splits.items():
            _save_split(
                bundle,
                name,
                frame,
                transformed,
                feature_order,
                baseline_columns,
                metadata={
                    "source": "crc_yunnan_frozen_split",
                    "split_seed": seed,
                    "split_strategy": strategy,
                    "outcome": outcome,
                    "landmark_days": landmark_days,
                    "observation_time_unit": "days_since_surgery",
                    "survival_time_unit": "days_since_landmark",
                    "observations_csv": str(Path(str(config.paths.observations_csv)).resolve()),
                    "preprocessing_parameters_csv": str(bundle / "preprocessing_parameters.csv"),
                    "split_manifest": str(bundle / "split_manifest.json"),
                },
            )
        for path in bundle.rglob("*"):
            if path.is_file():
                path.chmod(0o600)
        counts = {
            name: {"patients": len(frame), "events": int(frame["event"].sum())}
            for name, frame in splits.items()
        }
        _write_json(
            bundle / "split_manifest.json",
            {
                "schema_version": 3,
                "dataset": "crc_yunnan",
                "assignment_manifest": str(
                    Path(str(config.paths.assignment_root)).resolve()
                    / bundle.parent.name
                    / bundle.name
                    / "assignment_manifest.json"
                ),
                "strategy": strategy,
                "seed": seed,
                "outcome": outcome,
                "landmark_days": landmark_days,
                "panel": panel,
                "feature_order": feature_order,
                "preprocessing": config.preprocessing.model_dump(mode="json"),
                "preprocessing_parameters": "preprocessing_parameters.csv",
                "split_counts": counts,
            },
        )
        LOGGER.info("已冻结 %s seed=%s：%s，特征数=%d", strategy, seed, counts, len(feature_order))

    LOGGER.info("全部冻结完成：%s", output_root)


@hydra.main(config_path="../../configs", config_name="crc_yunnan/split", version_base="1.3")
def main(raw_config: DictConfig) -> None:
    config = CRCSplitConfig.model_validate(OmegaConf.to_container(raw_config, resolve=True))
    run(config)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import logging
from typing import Any, cast

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from scripts.crc_yunnan.config import CRCPreprocConfig
from scripts.crc_yunnan.data import read_tables, write_json
from scripts.utils.preprocessing import (
    AutoLog1pTransformer,
    RobustScalerWithStdFallback,
    save_scaler,
)
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from trails import ClinicalTimeSeriesDataset

LOGGER = logging.getLogger(__name__)


def run(config: CRCPreprocConfig) -> None:
    # ===========================================================================
    # 一、读取03标准产物，仅检查本次输入的患者交叉和输出位置。
    # ===========================================================================
    output = config.paths.dir.resolve()
    directories = {"train": config.inputs.train_dir.resolve()}
    directories.update({path.name: path.resolve() for path in config.inputs.test_dirs})
    if output.exists():
        raise FileExistsError(f"拒绝覆盖既有预处理目录：{output}")
    if any(
        output == directory or directory in output.parents for directory in directories.values()
    ):
        raise ValueError("预处理产物不得写入原始输入目录")
    features = config.features.panels[config.features.panel]
    landmark = float(config.landmark_months) * 30
    time_column, event_column = f"{config.outcome}_time_days", f"{config.outcome}_event"
    frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    sources: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    for name, directory in directories.items():
        patients, observations = read_tables(
            directory / "patients.csv", directory / "observations.csv"
        )
        if overlap := seen_ids.intersection(patients["patient_id"]):
            raise ValueError(f"输入患者存在交叉：{name}，重复人数={len(overlap)}")
        seen_ids.update(patients["patient_id"])
        if missing := set(features) - set(observations.columns):
            raise ValueError(f"固定面板缺少特征列：{name}，{sorted(missing)}")
        sources[name] = json.loads(
            (directory / "dataset_manifest.json").read_text(encoding="utf-8")
        )

        # =======================================================================
        # 二、在各输入上执行同一窗口及面板规则，保留原始主划分归属。
        # =======================================================================
        patients = patients.loc[patients[time_column] > landmark].copy()
        patients["survival_time"] = patients[time_column] - landmark
        patients["event"] = patients[event_column]
        observations = observations.loc[
            observations["patient_id"].isin(patients["patient_id"])
            & observations["time_days"].between(0, landmark),
            ["patient_id", "time_days", *features],
        ].copy()
        # 面板之外的观测不构成该面板的有效访视；NaN不会增加日期计数。
        observations = observations.loc[observations[features].notna().any(axis=1)].copy()
        dates = observations.groupby("patient_id")["time_days"].nunique()
        eligible = dates.index[dates >= config.min_observation_dates]
        patients = patients.loc[patients["patient_id"].isin(eligible)].reset_index(drop=True)
        observations = observations.loc[observations["patient_id"].isin(eligible)].reset_index(
            drop=True
        )
        if patients.empty:
            raise ValueError(f"{name}经landmark和面板观测日期筛选后为空")
        frames[name] = patients, observations

    # ===========================================================================
    # 三、仅用最终训练观测拟合；每个非缺失观测等权，不补值、不截尾。
    # ===========================================================================
    train_matrix = frames["train"][1][features]
    reasons = pd.Series("", index=features)
    reasons.loc[train_matrix.count() == 0] = "no_train_observations"
    reasons.loc[train_matrix.min() == train_matrix.max()] = "constant_in_train"
    if reasons.ne("").any():
        raise ValueError(f"固定面板在train存在退化指标：{reasons.loc[reasons.ne('')].to_dict()}")
    scalers: dict[str, BaseEstimator | str] = {
        "none": "passthrough",
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScalerWithStdFallback(),
    }
    log_transformer = (
        AutoLog1pTransformer(skew_threshold=config.preprocessing.skew_threshold)
        if config.preprocessing.log_transform == "auto-log1p"
        else "passthrough"
    )
    scaler = scalers[config.preprocessing.scaling]
    pipeline = Pipeline([("log", log_transformer), ("scale", scaler)]).fit(train_matrix)
    for name, (_, observations) in frames.items():
        original_mask = observations[features].notna().to_numpy()
        converted = np.asarray(pipeline.transform(observations[features]), dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            finite = np.isfinite(converted.astype(np.float32))
        if (
            not np.array_equal(np.isnan(converted), ~original_mask)
            or not finite[original_mask].all()
        ):
            raise ValueError(f"{name}预处理结果改变缺失位置或不能表示为有限float32")
        observations[features] = converted

    # ===========================================================================
    # 四、通过数据集API读取中格式，保存数值表、参数和张量。
    # ===========================================================================
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    save_scaler(
        output / "preprocessing_parameters.csv",
        cast(BaseEstimator, scaler),
        log_transformer,
        features,
    )
    OmegaConf.save(config.model_dump(mode="json"), output / "resolved_config.yaml")
    counts: dict[str, dict[str, int]] = {}
    for name, (patients, observations) in frames.items():
        directory = output / name
        directory.mkdir(mode=0o700)
        patients.to_csv(directory / "patients.csv", index=False)
        patients[["patient_id"]].to_csv(directory / "ids.csv", index=False)
        observations.to_csv(directory / "observations.csv", index=False)
        excluded_columns = {
            "patient_id",
            "recurrence",
            "surgery_date",
            "last_followup_date",
            "surgery_year",
            "dfs_time_days",
            "os_time_days",
            "dfs_event",
            "os_event",
            "survival_time",
            "event",
        }
        baseline: pd.DataFrame = patients.loc[
            :, ["patient_id", *(column for column in patients if column not in excluded_columns)]
        ].astype(object)
        baseline = baseline.where(pd.notna(baseline), None)
        dataset = ClinicalTimeSeriesDataset.from_dataframes(
            patients=patients,
            observations=observations,
            time_col="time_days",
            use_features=features,
            cluster_label_col=None,
            description=f"CRC Yunnan {config.outcome.upper()} landmark {landmark:g} days ({name})",
            metadata={
                "source": "crc_yunnan_preproc",
                "split_name": name,
                "baseline_covariates": baseline.to_dict(orient="records"),
                "outcome": config.outcome,
                "landmark_days": landmark,
                "observation_time_unit": "days_since_surgery",
                "survival_time_unit": "days_since_landmark",
                "preproc_manifest": str(output / "preproc_manifest.json"),
                "preprocessing_parameters_csv": str(output / "preprocessing_parameters.csv"),
                "patients_csv": str(directory / "patients.csv"),
                "observations_csv": str(directory / "observations.csv"),
            },
        )
        dataset.save(directory / "dataset.pt")
        counts[name] = {
            "patients": len(patients),
            "events": int(patients["event"].to_numpy().sum()),
        }
    write_json(
        output / "preproc_manifest.json",
        {
            "schema_version": 2,
            "dataset": config.dataset,
            "outcome": config.outcome,
            "landmark_days": landmark,
            "panel": config.features.panel,
            "feature_order": features,
            "preprocessing": config.preprocessing.model_dump(mode="json"),
            "preprocessing_parameters": "preprocessing_parameters.csv",
            "datasets": list(frames),
            "split_counts": counts,
            "source_datasets": {name: str(path) for name, path in directories.items()},
            "assignment_manifests": sorted(
                {
                    source["assignment_manifest"]
                    for source in sources.values()
                    if source["assignment_manifest"] is not None
                }
            ),
            "training_scope": "full_cohort"
            if sources["train"]["split_name"] == "full"
            else "split",
        },
    )
    for path in output.rglob("*"):
        if path.is_file():
            path.chmod(0o600)
    LOGGER.info("预处理完成：%s，特征数=%d，%s", output, len(features), counts)


@hydra.main(config_path="../../configs", config_name="crc_yunnan/preproc", version_base="1.3")
def main(raw_config: DictConfig) -> None:
    config = CRCPreprocConfig.model_validate(OmegaConf.to_container(raw_config, resolve=True))
    run(config)


if __name__ == "__main__":
    main()

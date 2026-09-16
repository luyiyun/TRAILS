from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from scripts.crc_yunnan.config import CRCCohortConfig


def _read_csv(path: Path, patient_id_column: str, config: CRCCohortConfig) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={patient_id_column: "string"})
    export_index = config.columns.export_index
    if config.processing.drop_export_index and export_index in frame.columns:
        frame = frame.drop(columns=export_index)
    return frame


def run(config: CRCCohortConfig) -> None:
    # ==================================================================================
    # 一、读取原始患者表和纵向观测表。
    # ==================================================================================
    patients_csv = config.paths.patients_csv
    observations_csv = config.paths.observations_csv
    if missing := [str(path) for path in (patients_csv, observations_csv) if not path.is_file()]:
        raise FileNotFoundError(f"缺少云南结直肠癌原始输入：{missing}")

    columns = config.columns
    processing = config.processing
    derived_dir = config.paths.derived_dir
    derived_names = (
        "patients.csv",
        "observations.csv",
        "private_id_map.csv",
        "feature_metadata.csv",
    )
    if processing.refuse_overwrite and derived_dir.exists():
        raise FileExistsError(f"拒绝覆盖既有预处理目录：{derived_dir}")

    source_id = columns.patient_id
    patients = _read_csv(patients_csv, source_id, config)
    observations = _read_csv(observations_csv, source_id, config)

    # ==================================================================================
    # 二、校验两张源表的字段契约。
    # ==================================================================================
    required_patient_columns = {
        source_id,
        columns.surgery_date,
        columns.last_followup_date,
        columns.survival_time,
        columns.death,
        columns.recurrence_free_time,
        columns.recurrence,
        *columns.baseline,
    }
    required_observation_columns = {
        source_id,
        columns.feature,
        columns.feature_group,
        columns.measurement_date,
        columns.value,
    }
    if missing := sorted(required_patient_columns - set(patients.columns)):
        raise ValueError(f"{patients_csv.name} 缺少必需字段：{missing}")
    if missing := sorted(required_observation_columns - set(observations.columns)):
        raise ValueError(f"{observations_csv.name} 缺少必需字段：{missing}")

    # ==================================================================================
    # 三、清洗患者日期和生存结局，并建立私有研究ID映射。
    # ==================================================================================
    # 1. 住院号只用于两表关联；最终分析表仅保留随机排列的研究ID。
    source_ids = patients[source_id] = patients[source_id].astype("string").str.strip()
    observations[source_id] = observations[source_id].astype("string").str.strip()
    if bool(source_ids.isna().any() | source_ids.eq("").any()):
        raise ValueError("患者表住院号存在缺失，无法建立研究ID映射")
    if bool(source_ids.duplicated().any()):
        raise ValueError("患者表住院号必须唯一")

    rng = np.random.default_rng(processing.research_id_seed)
    random_numbers = rng.permutation(np.arange(1, len(patients) + 1))  # type: ignore
    private_id_map = pd.DataFrame(
        {source_id: source_ids, "patient_id": [f"CRC{number:06d}" for number in random_numbers]}
    )
    patients = patients.merge(private_id_map, on=source_id, validate="one_to_one")

    # 2. 仅按配置修正已核实的错误日期；其他无法解析的日期保留为缺失并计数。
    surgery_date = columns.surgery_date
    followup_date = columns.last_followup_date
    patient_date_corrections = (
        config.date_corrections.patients if processing.apply_date_corrections else {}
    )
    patients[surgery_date] = pd.to_datetime(
        patients[surgery_date].astype("string").str.strip().replace(patient_date_corrections),
        format="mixed",
        errors="coerce",
    )
    patients[followup_date] = pd.to_datetime(
        patients[followup_date].astype("string").str.strip(), format="mixed", errors="coerce"
    )

    # 3. DFS事件包括复发或死亡，但DFS时间直接使用数据提供的RFT并按30天/月换算。
    death: pd.Series = pd.to_numeric(patients[columns.death], errors="coerce")  # type: ignore[reportAssignmentType]
    recurrence: pd.Series = pd.to_numeric(patients[columns.recurrence], errors="coerce")  # type: ignore[reportAssignmentType]
    dfs_time: pd.Series = pd.to_numeric(patients[columns.recurrence_free_time], errors="coerce")  # type: ignore[reportAssignmentType]
    os_time: pd.Series = pd.to_numeric(patients[columns.survival_time], errors="coerce")  # type: ignore[reportAssignmentType]
    unit_days = processing.outcome_time_unit_days
    patients["recurrence"] = recurrence
    patients["dfs_event"] = ((recurrence == 1) | (death == 1)).astype("int8")
    patients["dfs_time_days"] = dfs_time * unit_days
    patients["os_event"] = death
    patients["os_time_days"] = os_time * unit_days
    patients["surgery_year"] = patients[surgery_date].dt.year.astype("Int64")

    valid_events = death.isin([0, 1]) & recurrence.isin([0, 1])
    valid_times = np.isfinite(dfs_time) & np.isfinite(os_time)
    if processing.exclude_negative_outcome_time:
        valid_times &= (dfs_time >= 0) & (os_time >= 0)
    valid_surgery = patients[surgery_date].notna()
    included = valid_events & valid_times
    if processing.exclude_invalid_surgery_date:
        included &= valid_surgery

    # 基础纳排在划分前完成；各规则人数可重叠，另报顺序新增排除人数。
    initial_included = included.copy()
    reversed_followup = (
        patients[followup_date] < patients[surgery_date]
    ) & processing.exclude_followup_before_surgery
    too_long: pd.Series = pd.Series(False, index=patients.index)
    if processing.max_outcome_days is not None:
        too_long = (  # type: ignore[reportAssignmentType]
            patients[["dfs_time_days", "os_time_days"]].gt(processing.max_outcome_days).any(axis=1)
        )
    included &= ~reversed_followup & ~too_long
    if not included.any():
        raise ValueError("基础纳排后没有患者")
    summary = {
        "原始患者数": len(patients),
        "排除人数（规则间可重叠）": {
            "结局事件编码无效": int((~valid_events).sum()),
            "结局时间无效": int((~valid_times).sum()),
            "手术日期无效": int((~valid_surgery).to_numpy().sum()),
            "随访早于手术": int(reversed_followup.sum()),
            "任一结局时间超过上限": int(too_long.to_numpy().sum()),
        },
        "顺序排除人数": {
            "原清洗规则": int((~initial_included).sum()),
            "随访早于手术": int((initial_included & reversed_followup).sum()),
            "任一结局时间超过上限": int((initial_included & ~reversed_followup & too_long).sum()),
        },
        "最终患者数": int(included.sum()),
    }

    patient_columns = [
        "patient_id",
        surgery_date,
        followup_date,
        *columns.baseline,
        "recurrence",
        "dfs_event",
        "dfs_time_days",
        "os_event",
        "os_time_days",
        "surgery_year",
    ]
    prepared_patients = patients.loc[included, patient_columns].rename(
        columns={surgery_date: "surgery_date", followup_date: "last_followup_date"}
    )

    # ==================================================================================
    # 四、将纵向记录转换为相对手术时间，并聚合同日同指标记录。
    # ==================================================================================
    measurement_date = columns.measurement_date
    observation_date_corrections = (
        config.date_corrections.observations if processing.apply_date_corrections else {}
    )
    observations[measurement_date] = pd.to_datetime(
        observations[measurement_date]
        .astype("string")
        .str.strip()
        .replace(observation_date_corrections),
        format="mixed",
        errors="coerce",
    )
    value = columns.value
    observations[value] = pd.to_numeric(observations[value], errors="coerce")

    observations = observations.merge(
        patients.loc[included, [source_id, "patient_id", surgery_date]],
        on=source_id,
        how="inner",
        validate="many_to_one",
    )
    observations["time_days"] = (
        observations[measurement_date] - observations[surgery_date]
    ).dt.days
    feature = columns.feature
    feature_group = columns.feature_group
    valid_observation = (
        observations[measurement_date].notna()
        & observations[feature].notna()
        & np.isfinite(observations[value])
        & observations["time_days"].notna()
    )
    observations = observations.loc[valid_observation]

    same_day_keys = ["patient_id", feature, "time_days"]
    prepared_observations: pd.DataFrame = (
        observations.groupby(same_day_keys, as_index=False, dropna=False)
        .agg(feature_group=(feature_group, "first"), value=(value, processing.same_day_aggregation))
        .rename(columns={feature: "feature"})
    )

    if prepared_observations.empty:
        raise ValueError("清洗后没有有效纵向观测")
    if prepared_observations["feature"].isin(["patient_id", "time_days"]).to_numpy().any():
        raise ValueError("特征名称不能占用patient_id或time_days列")
    # 分组沿用原EDA的众数口径；元数据独立保存，不混入数值矩阵。
    feature_metadata: pd.DataFrame = (
        prepared_observations.assign(
            feature_group=lambda frame: frame["feature_group"].fillna("未记录").astype(str)
        )
        .groupby("feature", as_index=False)["feature_group"]
        .agg(lambda values: values.mode().iloc[0])
        .sort_values(["feature_group", "feature"])  # type: ignore[reportCallIssue]
    )
    observation_count = len(prepared_observations)
    prepared_observations = (
        prepared_observations.pivot(
            index=["patient_id", "time_days"], columns="feature", values="value"
        )
        .reindex(columns=feature_metadata["feature"].tolist())
        .reset_index()
    )
    prepared_observations.columns.name = None
    summary.update({"有效观测值数": observation_count, "患者时间点数": len(prepared_observations)})

    # ==================================================================================
    # 五、写出清洗后的患者表、纵向观测表和私有ID映射。
    # ==================================================================================
    private_id_map["included"] = included.to_numpy(dtype=bool)
    prepared_patients = prepared_patients.sort_values("patient_id")
    prepared_observations = prepared_observations.sort_values(["patient_id", "time_days"])
    private_id_map = private_id_map.sort_values("patient_id")

    derived_dir.mkdir(parents=True, exist_ok=not processing.refuse_overwrite, mode=0o700)
    derived_paths = {name: derived_dir / name for name in derived_names}
    prepared_patients.to_csv(derived_paths["patients.csv"], index=False)
    prepared_observations.to_csv(derived_paths["observations.csv"], index=False)
    private_id_map.to_csv(derived_paths["private_id_map.csv"], index=False)
    feature_metadata.to_csv(derived_paths["feature_metadata.csv"], index=False)
    (derived_dir / "cohort_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    OmegaConf.save(config.model_dump(mode="json"), derived_dir / "resolved_config.yaml")
    for path in derived_dir.iterdir():
        path.chmod(0o600)
    print(
        f"基础队列完成：患者数={len(prepared_patients)}, "
        f"患者时间点数={len(prepared_observations)}, 有效观测数={observation_count}；"
        f"输出目录：{derived_dir}"
    )


@hydra.main(config_path="../../configs", config_name="crc_yunnan/cohort", version_base="1.3")
def main(raw_config: DictConfig) -> None:
    config = CRCCohortConfig.model_validate(OmegaConf.to_container(raw_config, resolve=True))
    run(config)


if __name__ == "__main__":
    main()

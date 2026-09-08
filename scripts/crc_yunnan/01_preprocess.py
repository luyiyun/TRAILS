from __future__ import annotations

from pathlib import Path
from typing import cast

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    return cast(pd.Series, frame[name])


def _read_csv(path: Path, patient_id_column: str, config: DictConfig) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={patient_id_column: "string"})
    export_index = str(config.columns.export_index)
    if bool(config.processing.drop_export_index) and export_index in frame.columns:
        frame = frame.drop(columns=export_index)
    return frame


def _parse_dates(series: pd.Series, corrections: dict[str, str]) -> pd.Series:
    values = series.astype("string").str.strip().replace(corrections)
    return cast(pd.Series, pd.to_datetime(values, format="mixed", errors="coerce"))


@hydra.main(config_path="../../configs", config_name="crc_yunnan/preprocess", version_base="1.3")
def main(config: DictConfig) -> None:
    # ==================================================================================
    # 一、读取原始患者表和纵向观测表。
    # ==================================================================================
    patients_csv = Path(str(config.paths.patients_csv)).resolve()
    observations_csv = Path(str(config.paths.observations_csv)).resolve()
    if missing := [str(path) for path in (patients_csv, observations_csv) if not path.is_file()]:
        raise FileNotFoundError(f"缺少云南结直肠癌原始输入：{missing}")

    columns = config.columns
    processing = config.processing
    derived_dir = Path(str(config.paths.derived_dir)).resolve()
    derived_names = ("patients.csv", "observations.csv", "private_id_map.csv")
    if bool(processing.refuse_overwrite) and derived_dir.exists():
        raise FileExistsError(f"拒绝覆盖既有预处理目录：{derived_dir}")

    source_id = str(columns.patient_id)
    patients = _read_csv(patients_csv, source_id, config)
    observations = _read_csv(observations_csv, source_id, config)

    # ==================================================================================
    # 二、校验两张源表的字段契约。
    # ==================================================================================
    required_patient_columns = {
        source_id,
        str(columns.surgery_date),
        str(columns.last_followup_date),
        str(columns.survival_time),
        str(columns.death),
        str(columns.recurrence_free_time),
        str(columns.recurrence),
        *(str(column) for column in columns.baseline),
    }
    required_observation_columns = {
        source_id,
        str(columns.feature),
        str(columns.feature_group),
        str(columns.measurement_date),
        str(columns.value),
    }
    if missing := sorted(required_patient_columns - set(patients.columns)):
        raise ValueError(f"{patients_csv.name} 缺少必需字段：{missing}")
    if missing := sorted(required_observation_columns - set(observations.columns)):
        raise ValueError(f"{observations_csv.name} 缺少必需字段：{missing}")

    # ==================================================================================
    # 三、清洗患者日期和生存结局，并建立私有研究ID映射。
    # ==================================================================================
    # 1. 住院号只用于两表关联；最终分析表仅保留随机排列的研究ID。
    patients[source_id] = _column(patients, source_id).astype("string").str.strip()
    observations[source_id] = _column(observations, source_id).astype("string").str.strip()
    source_ids = _column(patients, source_id)
    if bool(source_ids.isna().any() | source_ids.eq("").any()):
        raise ValueError("患者表住院号存在缺失，无法建立研究ID映射")
    if bool(source_ids.duplicated().any()):
        raise ValueError("患者表住院号必须唯一")

    rng = np.random.default_rng(int(processing.research_id_seed))
    random_numbers = rng.permutation(np.arange(1, len(patients) + 1))
    private_id_map = pd.DataFrame(
        {source_id: source_ids, "patient_id": [f"CRC{number:06d}" for number in random_numbers]}
    )
    patients = patients.merge(private_id_map, on=source_id, validate="one_to_one")

    # 2. 仅按配置修正已核实的错误日期；其他无法解析的日期保留为缺失并计数。
    surgery_date = str(columns.surgery_date)
    followup_date = str(columns.last_followup_date)
    patient_date_corrections = (
        {str(old): str(new) for old, new in config.date_corrections.patients.items()}
        if bool(processing.apply_date_corrections)
        else {}
    )
    patients[surgery_date] = _parse_dates(_column(patients, surgery_date), patient_date_corrections)
    patients[followup_date] = _parse_dates(_column(patients, followup_date), {})

    # 3. DFS事件包括复发或死亡，但DFS时间直接使用数据提供的RFT并按30天/月换算。
    death = cast(pd.Series, pd.to_numeric(_column(patients, str(columns.death)), errors="coerce"))
    recurrence = cast(
        pd.Series, pd.to_numeric(_column(patients, str(columns.recurrence)), errors="coerce")
    )
    dfs_time = cast(
        pd.Series,
        pd.to_numeric(_column(patients, str(columns.recurrence_free_time)), errors="coerce"),
    )
    os_time = cast(
        pd.Series,
        pd.to_numeric(_column(patients, str(columns.survival_time)), errors="coerce"),
    )
    unit_days = int(processing.outcome_time_unit_days)
    patients["recurrence"] = recurrence.astype("Int8")
    patients["dfs_event"] = ((recurrence == 1) | (death == 1)).astype("int8")
    patients["dfs_time_days"] = dfs_time * unit_days
    patients["os_event"] = death.astype("Int8")
    patients["os_time_days"] = os_time * unit_days
    patients["surgery_year"] = _column(patients, surgery_date).dt.year.astype("Int64")

    valid_events = death.isin([0, 1]) & recurrence.isin([0, 1])
    valid_times = dfs_time.notna() & os_time.notna()
    if bool(processing.exclude_negative_outcome_time):
        valid_times &= (dfs_time >= 0) & (os_time >= 0)
    valid_surgery = _column(patients, surgery_date).notna()
    included = valid_events & valid_times
    if bool(processing.exclude_invalid_surgery_date):
        included &= valid_surgery

    patient_columns = [
        "patient_id",
        surgery_date,
        followup_date,
        *(str(column) for column in columns.baseline),
        "recurrence",
        "dfs_event",
        "dfs_time_days",
        "os_event",
        "os_time_days",
        "surgery_year",
    ]
    prepared_patients = cast(pd.DataFrame, patients.loc[included, patient_columns]).rename(
        columns={surgery_date: "surgery_date", followup_date: "last_followup_date"}
    )

    # ==================================================================================
    # 四、将纵向记录转换为相对手术时间，并聚合同日同指标记录。
    # ==================================================================================
    measurement_date = str(columns.measurement_date)
    observation_date_corrections = (
        {str(old): str(new) for old, new in config.date_corrections.observations.items()}
        if bool(processing.apply_date_corrections)
        else {}
    )
    observations[measurement_date] = _parse_dates(
        _column(observations, measurement_date), observation_date_corrections
    )
    value = str(columns.value)
    observations[value] = pd.to_numeric(_column(observations, value), errors="coerce")

    observations = observations.merge(
        cast(pd.DataFrame, patients.loc[included, [source_id, "patient_id", surgery_date]]),
        on=source_id,
        how="inner",
        validate="many_to_one",
    )
    observations["time_days"] = (
        _column(observations, measurement_date) - _column(observations, surgery_date)
    ).dt.days
    feature = str(columns.feature)
    feature_group = str(columns.feature_group)
    valid_observation = (
        _column(observations, measurement_date).notna()
        & _column(observations, feature).notna()
        & _column(observations, value).notna()
    )
    observations = cast(pd.DataFrame, observations.loc[valid_observation])

    same_day_keys = ["patient_id", feature, "time_days"]
    aggregation = str(processing.same_day_aggregation)
    if aggregation not in {"median", "mean"}:
        raise ValueError("processing.same_day_aggregation 必须是 median 或 mean")
    prepared_observations = cast(
        pd.DataFrame,
        observations.groupby(same_day_keys, as_index=False, dropna=False).agg(
            feature_group=(feature_group, "first"), value=(value, aggregation)
        ),
    ).rename(columns={feature: "feature"})

    # ==================================================================================
    # 五、写出清洗后的患者表、纵向观测表和私有ID映射。
    # ==================================================================================
    private_id_map["included"] = included.to_numpy(dtype=bool)
    prepared_patients = prepared_patients.sort_values("patient_id")
    prepared_observations = prepared_observations.sort_values(
        ["patient_id", "time_days", "feature"]
    )
    private_id_map = private_id_map.sort_values("patient_id")

    derived_dir.mkdir(parents=True, exist_ok=not bool(processing.refuse_overwrite))
    derived_paths = {name: derived_dir / name for name in derived_names}
    prepared_patients.to_csv(derived_paths["patients.csv"], index=False)
    prepared_observations.to_csv(derived_paths["observations.csv"], index=False)
    private_id_map.to_csv(derived_paths["private_id_map.csv"], index=False)
    for path in derived_paths.values():
        path.chmod(0o600)
    print(
        f"预处理完成：patients={len(prepared_patients)}, "
        f"observations={len(prepared_observations)}；输出目录：{derived_dir}"
    )


if __name__ == "__main__":
    main()

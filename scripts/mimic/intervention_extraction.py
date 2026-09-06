"""从mimic-code生成的官方concept表汇总患者级器官支持与治疗措施。"""

from __future__ import annotations

from collections.abc import Collection

import duckdb
import numpy as np
import pandas as pd

INTERVENTION_BINARY_COLUMNS = (
    "invasive_ventilation_any",
    "noninvasive_ventilation_any",
    "hfnc_any",
    "vasopressor_any",
    "rrt_any",
    "crrt_any",
    "antibiotic_any",
)
INTERVENTION_CONTINUOUS_COLUMNS = (
    "observed_window_hours",
    "invasive_ventilation_hours",
    "invasive_ventilation_fraction",
    "noninvasive_ventilation_hours",
    "noninvasive_ventilation_fraction",
    "hfnc_hours",
    "hfnc_fraction",
    "vasopressor_hours",
    "vasopressor_fraction",
    "vasopressor_ned_peak",
    "vasopressor_ned_twa",
    "antibiotic_agent_count",
    "antibiotic_first_start_hours",
)
INTERVENTION_COLUMNS = INTERVENTION_BINARY_COLUMNS + INTERVENTION_CONTINUOUS_COLUMNS
INTERVENTION_TABLES = {
    "ventilation",
    "rrt",
    "crrt",
    "antibiotic",
    "norepinephrine_equivalent_dose",
}
INTERVENTION_SOURCES = {
    "observed_window_hours": "cohort: ICU intime to min(outtime, landmark_time)",
    "invasive_ventilation_any": "ventilation: InvasiveVent",
    "invasive_ventilation_hours": "ventilation: merged InvasiveVent intervals",
    "invasive_ventilation_fraction": "hours divided by observed window",
    "noninvasive_ventilation_any": "ventilation: NonInvasiveVent",
    "noninvasive_ventilation_hours": "ventilation: merged NonInvasiveVent intervals",
    "noninvasive_ventilation_fraction": "hours divided by observed window",
    "hfnc_any": "ventilation: HFNC",
    "hfnc_hours": "ventilation: merged HFNC intervals",
    "hfnc_fraction": "hours divided by observed window",
    "vasopressor_any": "norepinephrine_equivalent_dose: positive total NED interval",
    "vasopressor_hours": "norepinephrine_equivalent_dose: merged positive total NED intervals",
    "vasopressor_fraction": "hours divided by observed window",
    "vasopressor_ned_peak": "norepinephrine_equivalent_dose: total NED peak (mcg/kg/min)",
    "vasopressor_ned_twa": "norepinephrine_equivalent_dose: duration-weighted positive total NED",
    "rrt_any": "rrt: dialysis_active or active CRRT",
    "crrt_any": "rrt dialysis type or crrt system_active",
    "antibiotic_any": "antibiotic prescription",
    "antibiotic_agent_count": "distinct normalized antibiotic prescription labels",
    "antibiotic_first_start_hours": "first overlapping antibiotic prescription",
}

# 可靠区间先合并重叠记录；RRT/CRRT因来源混合而只保留存在性。
INTERVENTIONS_SQL = """
WITH cohort AS (
    SELECT
        subject_id,
        hadm_id,
        stay_id,
        intime,
        LEAST(outtime, landmark_time) AS window_end,
        DATE_DIFF('second', intime, LEAST(outtime, landmark_time)) / 3600.0
            AS observed_window_hours
    FROM trails_cohort_primary
), ventilation_clipped AS (
    SELECT
        c.stay_id,
        v.ventilation_status,
        GREATEST(v.starttime, c.intime) AS starttime,
        LEAST(v.endtime, c.window_end) AS endtime
    FROM cohort AS c
    INNER JOIN mimiciv_derived.ventilation AS v
        ON c.stay_id = v.stay_id
        AND v.starttime < c.window_end
        AND v.endtime > c.intime
    WHERE v.ventilation_status IN ('InvasiveVent', 'NonInvasiveVent', 'HFNC')
), ventilation_prepared AS (
    SELECT
        *,
        MAX(endtime) OVER (
            PARTITION BY stay_id, ventilation_status
            ORDER BY starttime, endtime
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS prior_endtime
    FROM ventilation_clipped
), ventilation_grouped AS (
    SELECT
        *,
        SUM(CAST(prior_endtime IS NULL OR starttime > prior_endtime AS INTEGER))
            OVER (
                PARTITION BY stay_id, ventilation_status
                ORDER BY starttime, endtime
            ) AS interval_group
    FROM ventilation_prepared
), ventilation_merged AS (
    SELECT
        stay_id,
        ventilation_status,
        MIN(starttime) AS starttime,
        MAX(endtime) AS endtime
    FROM ventilation_grouped
    GROUP BY stay_id, ventilation_status, interval_group
), ventilation AS (
    SELECT
        c.stay_id,
        COALESCE(SUM(CASE WHEN v.ventilation_status = 'InvasiveVent'
            THEN DATE_DIFF('second', v.starttime, v.endtime) ELSE 0 END), 0) / 3600.0
            AS invasive_ventilation_hours,
        COALESCE(SUM(CASE WHEN v.ventilation_status = 'NonInvasiveVent'
            THEN DATE_DIFF('second', v.starttime, v.endtime) ELSE 0 END), 0) / 3600.0
            AS noninvasive_ventilation_hours,
        COALESCE(SUM(CASE WHEN v.ventilation_status = 'HFNC'
            THEN DATE_DIFF('second', v.starttime, v.endtime) ELSE 0 END), 0) / 3600.0
            AS hfnc_hours
    FROM cohort AS c
    LEFT JOIN ventilation_merged AS v ON c.stay_id = v.stay_id
    GROUP BY c.stay_id
), vasopressor_clipped AS (
    -- 直接复用官方总NED，暴露、时长与剂量统一来自正剂量区间。
    SELECT
        c.stay_id,
        v.norepinephrine_equivalent_dose AS ned_rate,
        GREATEST(v.starttime, c.intime) AS starttime,
        LEAST(v.endtime, c.window_end) AS endtime
    FROM cohort AS c
    INNER JOIN mimiciv_derived.norepinephrine_equivalent_dose AS v
        ON c.stay_id = v.stay_id
        AND v.starttime < c.window_end
        AND v.endtime > c.intime
        AND v.endtime > v.starttime
        AND v.norepinephrine_equivalent_dose > 0
), vasopressor_prepared AS (
    SELECT
        *,
        MAX(endtime) OVER (
            PARTITION BY stay_id
            ORDER BY starttime, endtime
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS prior_endtime
    FROM vasopressor_clipped
), vasopressor_grouped AS (
    SELECT
        *,
        SUM(CAST(prior_endtime IS NULL OR starttime > prior_endtime AS INTEGER))
            OVER (PARTITION BY stay_id ORDER BY starttime, endtime) AS interval_group
    FROM vasopressor_prepared
), vasopressor_merged AS (
    SELECT stay_id, MIN(starttime) AS starttime, MAX(endtime) AS endtime
    FROM vasopressor_grouped
    GROUP BY stay_id, interval_group
), vasopressor_duration AS (
    SELECT
        stay_id,
        SUM(DATE_DIFF('second', starttime, endtime)) / 3600.0 AS hours
    FROM vasopressor_merged
    GROUP BY stay_id
), vasopressor_dose AS (
    SELECT
        stay_id,
        MAX(ned_rate) AS ned_peak,
        SUM(ned_rate * DATE_DIFF('second', starttime, endtime))
            / SUM(DATE_DIFF('second', starttime, endtime)) AS ned_twa
    FROM vasopressor_clipped
    GROUP BY stay_id
), vasopressor AS (
    SELECT
        c.stay_id,
        CAST(COALESCE(h.hours, 0) > 0 AS INTEGER) AS any_flag,
        COALESCE(h.hours, 0) AS hours,
        COALESCE(h.hours, 0) / c.observed_window_hours AS fraction,
        COALESCE(n.ned_peak, 0) AS ned_peak,
        COALESCE(n.ned_twa, 0) AS ned_twa
    FROM cohort AS c
    LEFT JOIN vasopressor_duration AS h ON c.stay_id = h.stay_id
    LEFT JOIN vasopressor_dose AS n ON c.stay_id = n.stay_id
), renal_replacement AS (
    SELECT
        c.stay_id,
        COALESCE(MAX(CAST(r.dialysis_active = 1 AS INTEGER)), 0) AS rrt,
        COALESCE(
            MAX(CAST(
                r.dialysis_active = 1
                AND UPPER(COALESCE(r.dialysis_type, ''))
                    IN ('CRRT', 'CVVH', 'CVVHD', 'CVVHDF', 'SCUF')
                AS INTEGER
            )),
            0
        ) AS crrt
    FROM cohort AS c
    LEFT JOIN mimiciv_derived.rrt AS r
        ON c.stay_id = r.stay_id
        AND r.charttime >= c.intime
        AND r.charttime <= c.window_end
    GROUP BY c.stay_id
), continuous_renal_replacement AS (
    SELECT
        c.stay_id,
        COALESCE(MAX(CAST(r.system_active = 1 AS INTEGER)), 0) AS crrt
    FROM cohort AS c
    LEFT JOIN mimiciv_derived.crrt AS r
        ON c.stay_id = r.stay_id
        AND r.charttime >= c.intime
        AND r.charttime <= c.window_end
    GROUP BY c.stay_id
), antibiotic AS (
    SELECT
        c.stay_id,
        CAST(COUNT(a.hadm_id) > 0 AS INTEGER) AS antibiotic,
        COUNT(DISTINCT LOWER(TRIM(a.antibiotic))) AS antibiotic_agent_count,
        CASE WHEN COUNT(a.hadm_id) > 0 THEN
            DATE_DIFF('second', c.intime, MIN(GREATEST(a.starttime, c.intime))) / 3600.0
        END AS antibiotic_first_start_hours
    FROM cohort AS c
    LEFT JOIN mimiciv_derived.antibiotic AS a
        ON c.subject_id = a.subject_id
        AND c.hadm_id = a.hadm_id
        AND a.starttime < c.window_end
        AND COALESCE(a.stoptime, a.starttime) > c.intime
    GROUP BY c.stay_id, c.intime
)
SELECT
    c.stay_id AS patient_id,
    c.observed_window_hours,
    CAST(v.invasive_ventilation_hours > 0 AS INTEGER) AS invasive_ventilation_any,
    v.invasive_ventilation_hours,
    v.invasive_ventilation_hours / c.observed_window_hours
        AS invasive_ventilation_fraction,
    CAST(v.noninvasive_ventilation_hours > 0 AS INTEGER)
        AS noninvasive_ventilation_any,
    v.noninvasive_ventilation_hours,
    v.noninvasive_ventilation_hours / c.observed_window_hours
        AS noninvasive_ventilation_fraction,
    CAST(v.hfnc_hours > 0 AS INTEGER) AS hfnc_any,
    v.hfnc_hours,
    v.hfnc_hours / c.observed_window_hours AS hfnc_fraction,
    p.any_flag AS vasopressor_any,
    p.hours AS vasopressor_hours,
    p.fraction AS vasopressor_fraction,
    p.ned_peak AS vasopressor_ned_peak,
    p.ned_twa AS vasopressor_ned_twa,
    GREATEST(r.rrt, x.crrt) AS rrt_any,
    GREATEST(r.crrt, x.crrt) AS crrt_any,
    a.antibiotic AS antibiotic_any,
    a.antibiotic_agent_count,
    a.antibiotic_first_start_hours
FROM cohort AS c
INNER JOIN ventilation AS v ON c.stay_id = v.stay_id
INNER JOIN vasopressor AS p ON c.stay_id = p.stay_id
INNER JOIN renal_replacement AS r ON c.stay_id = r.stay_id
INNER JOIN continuous_renal_replacement AS x ON c.stay_id = x.stay_id
INNER JOIN antibiotic AS a ON c.stay_id = a.stay_id
ORDER BY c.stay_id
"""


def extract_interventions(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """执行治疗事件聚合，并固定患者级输出列顺序。"""
    interventions = connection.execute(INTERVENTIONS_SQL).df()
    return interventions.loc[:, ["patient_id", *INTERVENTION_COLUMNS]]


def validate_interventions(
    interventions: pd.DataFrame,
    expected_patients: Collection[object],
) -> None:
    """校验治疗数据的一行一患者、取值范围和时间窗口约束。"""
    if interventions["patient_id"].duplicated().any():
        raise ValueError("治疗措施文件的patient_id必须唯一")
    if set(interventions["patient_id"]) != set(expected_patients):
        raise ValueError("治疗措施与建模患者集合不一致")

    flags = interventions.loc[:, INTERVENTION_BINARY_COLUMNS].to_numpy(dtype=np.int64)
    if not np.isin(flags, (0, 1)).all():
        raise ValueError("治疗措施标志必须为0或1")
    nonnegative_columns = [
        column
        for column in INTERVENTION_CONTINUOUS_COLUMNS
        if column != "antibiotic_first_start_hours"
    ]
    if (interventions.loc[:, nonnegative_columns] < 0).to_numpy().any():
        raise ValueError("治疗措施时长、比例和计数不能为负数")

    observed_hours = interventions["observed_window_hours"].to_numpy(dtype=float)
    if not (observed_hours > 0.0).all():
        raise ValueError("治疗措施观察窗口必须大于0小时")
    fractions = interventions.filter(regex=r"_fraction$")
    if not fractions.apply(lambda column: column.between(0.0, 1.0)).to_numpy().all():
        raise ValueError("治疗措施时间比例必须位于0到1之间")
    ned_peaks = interventions["vasopressor_ned_peak"].to_numpy(dtype=float)
    ned_twas = interventions["vasopressor_ned_twa"].to_numpy(dtype=float)
    if (ned_twas > ned_peaks + 1e-12).any():
        raise ValueError("血管活性药NED时间加权均值不能超过峰值")
    if not (np.isfinite(ned_peaks).all() and np.isfinite(ned_twas).all()):
        raise ValueError("官方总NED峰值和时间加权均值必须为有限数值")

    antibiotic_time = interventions["antibiotic_first_start_hours"]
    expected_missing = interventions["antibiotic_any"].eq(0)
    if not (antibiotic_time.isna().to_numpy() == expected_missing.to_numpy()).all():
        raise ValueError("首次抗菌药时间必须且只能在未治疗患者中缺失")
    antibiotic_hours = antibiotic_time.fillna(0.0).to_numpy(dtype=float)
    if (antibiotic_hours > observed_hours).any() or (antibiotic_hours < 0.0).any():
        raise ValueError("首次抗菌药时间超出患者实际观察窗口")

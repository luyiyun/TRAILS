"""从MIMIC事件表提取患者级器官支持与治疗措施。"""

from __future__ import annotations

from collections.abc import Collection

import duckdb
import numpy as np
import pandas as pd

VASOPRESSOR_DRUGS: dict[str, tuple[int, str, str]] = {
    "norepinephrine": (221906, "norepinephrine", "1.0"),
    "epinephrine": (221289, "epinephrine", "1.0"),
    "dopamine": (221662, "dopamine", "1.0 / 100.0"),
    "phenylephrine": (221749, "phenylephrine", "1.0 / 10.0"),
    "vasopressin": (222315, "vasopressin", "2.5 / 60.0"),
}
VASOPRESSOR_ITEM_IDS = tuple(specification[0] for specification in VASOPRESSOR_DRUGS.values())
VASOPRESSOR_METRICS = {
    "any": "any_flag",
    "hours": "hours",
    "fraction": "fraction",
    "ned_peak": "ned_peak",
    "ned_twa": "ned_twa",
}
INTERVENTION_BINARY_COLUMNS = (
    "invasive_ventilation_any",
    "noninvasive_ventilation_any",
    "hfnc_any",
    *(f"{drug}_any" for drug in VASOPRESSOR_DRUGS),
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
    *(
        f"{drug}_{metric}"
        for drug in VASOPRESSOR_DRUGS
        for metric in tuple(VASOPRESSOR_METRICS)[1:]
    ),
    "antibiotic_agent_count",
    "antibiotic_first_start_hours",
)
INTERVENTION_COLUMNS = INTERVENTION_BINARY_COLUMNS + INTERVENTION_CONTINUOUS_COLUMNS
INTERVENTION_TABLES = {
    "ventilation",
    "rrt",
    "crrt",
    "antibiotic",
    "vasoactive_agent",
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
    **{
        f"{drug}_{metric}": {
            "any": f"inputevents: documented {drug}",
            "hours": f"inputevents: merged positive-duration {drug} intervals",
            "fraction": "hours divided by observed window",
            "ned_peak": f"{drug} norepinephrine-equivalent peak (mcg/kg/min)",
            "ned_twa": f"duration-weighted {drug} NED during positive-dose intervals",
        }[metric]
        for drug in VASOPRESSOR_DRUGS
        for metric in VASOPRESSOR_METRICS
    },
    "rrt_any": "rrt: dialysis_active or active CRRT",
    "crrt_any": "rrt dialysis type or crrt system_active",
    "antibiotic_any": "antibiotic prescription",
    "antibiotic_agent_count": "distinct normalized antibiotic prescription labels",
    "antibiotic_first_start_hours": "first overlapping antibiotic prescription",
}

VASOPRESSOR_DRUG_CASE_SQL = (
    "CASE itemid "
    + " ".join(
        f"WHEN {item_id} THEN '{drug}'" for drug, (item_id, _, _) in VASOPRESSOR_DRUGS.items()
    )
    + " END"
)
VASOPRESSOR_DRUG_VALUES_SQL = ", ".join(f"('{drug}')" for drug in VASOPRESSOR_DRUGS)
VASOPRESSOR_DOSE_EVENTS_SQL = "\nUNION ALL\n".join(
    "SELECT stay_id, starttime, endtime, "
    f"'{drug}' AS drug, {column} * ({factor}) AS ned_rate "
    "FROM mimiciv_derived.vasoactive_agent "
    f"WHERE {column} IS NOT NULL"
    for drug, (_, column, factor) in VASOPRESSOR_DRUGS.items()
)
VASOPRESSOR_PIVOT_SQL = ",\n        ".join(
    f"MAX(CASE WHEN drug = '{drug}' THEN {value} END) AS {drug}_{metric}"
    for drug in VASOPRESSOR_DRUGS
    for metric, value in VASOPRESSOR_METRICS.items()
)
VASOPRESSOR_OUTPUT_SQL = ",\n    ".join(
    f"p.{drug}_{metric}" for drug in VASOPRESSOR_DRUGS for metric in VASOPRESSOR_METRICS
)

# 可靠区间先合并重叠记录；RRT/CRRT因来源混合而只保留存在性。
INTERVENTIONS_SQL = f"""
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
), vasopressor_events AS (
    SELECT
        stay_id,
        {VASOPRESSOR_DRUG_CASE_SQL} AS drug,
        starttime,
        endtime
    FROM mimiciv_icu.inputevents
    WHERE itemid IN ({", ".join(map(str, VASOPRESSOR_ITEM_IDS))})
        AND COALESCE(statusdescription, '') <> 'Rewritten'
        AND (COALESCE(amount, 0) > 0 OR COALESCE(rate, 0) > 0)
), vasopressor_presence AS (
    SELECT c.stay_id, v.drug, 1 AS any_flag
    FROM cohort AS c
    INNER JOIN vasopressor_events AS v
        ON c.stay_id = v.stay_id
        AND v.starttime < c.window_end
        AND COALESCE(v.endtime, v.starttime) > c.intime
    GROUP BY c.stay_id, v.drug
), vasopressor_clipped AS (
    SELECT
        c.stay_id,
        v.drug,
        GREATEST(v.starttime, c.intime) AS starttime,
        LEAST(v.endtime, c.window_end) AS endtime
    FROM cohort AS c
    INNER JOIN vasopressor_events AS v
        ON c.stay_id = v.stay_id
        AND v.starttime < c.window_end
        AND v.endtime > c.intime
        AND v.endtime > v.starttime
), vasopressor_prepared AS (
    SELECT
        *,
        MAX(endtime) OVER (
            PARTITION BY stay_id, drug
            ORDER BY starttime, endtime
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS prior_endtime
    FROM vasopressor_clipped
), vasopressor_grouped AS (
    SELECT
        *,
        SUM(CAST(prior_endtime IS NULL OR starttime > prior_endtime AS INTEGER))
            OVER (PARTITION BY stay_id, drug ORDER BY starttime, endtime) AS interval_group
    FROM vasopressor_prepared
), vasopressor_merged AS (
    SELECT stay_id, drug, MIN(starttime) AS starttime, MAX(endtime) AS endtime
    FROM vasopressor_grouped
    GROUP BY stay_id, drug, interval_group
), vasopressor_duration AS (
    SELECT
        stay_id,
        drug,
        SUM(DATE_DIFF('second', starttime, endtime)) / 3600.0 AS hours
    FROM vasopressor_merged
    GROUP BY stay_id, drug
), vasopressor_dose_events AS (
    {VASOPRESSOR_DOSE_EVENTS_SQL}
), vasopressor_dose AS (
    SELECT
        c.stay_id,
        n.drug,
        MAX(n.ned_rate) AS ned_peak,
        SUM(
            n.ned_rate
            * DATE_DIFF(
                'second',
                GREATEST(n.starttime, c.intime),
                LEAST(n.endtime, c.window_end)
            )
        ) / NULLIF(
            SUM(DATE_DIFF(
                'second',
                GREATEST(n.starttime, c.intime),
                LEAST(n.endtime, c.window_end)
            )),
            0
        ) AS ned_twa
    FROM cohort AS c
    INNER JOIN vasopressor_dose_events AS n
        ON c.stay_id = n.stay_id
        AND n.starttime < c.window_end
        AND n.endtime > c.intime
        AND n.endtime > n.starttime
        AND n.ned_rate > 0
    GROUP BY c.stay_id, n.drug
), vasopressor_long AS (
    SELECT
        c.stay_id,
        g.drug,
        COALESCE(p.any_flag, 0) AS any_flag,
        COALESCE(h.hours, 0) AS hours,
        COALESCE(h.hours, 0) / c.observed_window_hours AS fraction,
        CASE WHEN COALESCE(p.any_flag, 0) = 0 THEN 0 ELSE n.ned_peak END AS ned_peak,
        CASE WHEN COALESCE(p.any_flag, 0) = 0 THEN 0 ELSE n.ned_twa END AS ned_twa
    FROM cohort AS c
    CROSS JOIN (VALUES {VASOPRESSOR_DRUG_VALUES_SQL}) AS g(drug)
    LEFT JOIN vasopressor_presence AS p
        ON c.stay_id = p.stay_id AND g.drug = p.drug
    LEFT JOIN vasopressor_duration AS h
        ON c.stay_id = h.stay_id AND g.drug = h.drug
    LEFT JOIN vasopressor_dose AS n
        ON c.stay_id = n.stay_id AND g.drug = n.drug
), vasopressor AS (
    SELECT
        stay_id,
        {VASOPRESSOR_PIVOT_SQL}
    FROM vasopressor_long
    GROUP BY stay_id
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
    {VASOPRESSOR_OUTPUT_SQL},
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
    ned_peaks = interventions.loc[:, [f"{drug}_ned_peak" for drug in VASOPRESSOR_DRUGS]].to_numpy(
        dtype=float
    )
    ned_twas = interventions.loc[:, [f"{drug}_ned_twa" for drug in VASOPRESSOR_DRUGS]].to_numpy(
        dtype=float
    )
    drug_flags = interventions.loc[:, [f"{drug}_any" for drug in VASOPRESSOR_DRUGS]].to_numpy(
        dtype=np.int64
    )
    if (ned_twas > ned_peaks + 1e-12).any():
        raise ValueError("血管活性药NED时间加权均值不能超过峰值")
    if (np.isnan(ned_peaks) != np.isnan(ned_twas)).any():
        raise ValueError("血管活性药NED峰值和时间加权均值必须同时缺失")
    if (np.isnan(ned_peaks) & (drug_flags == 0)).any():
        raise ValueError("未使用血管活性药的患者NED应记为0而不是缺失")

    antibiotic_time = interventions["antibiotic_first_start_hours"]
    expected_missing = interventions["antibiotic_any"].eq(0)
    if not (antibiotic_time.isna().to_numpy() == expected_missing.to_numpy()).all():
        raise ValueError("首次抗菌药时间必须且只能在未治疗患者中缺失")
    antibiotic_hours = antibiotic_time.fillna(0.0).to_numpy(dtype=float)
    if (antibiotic_hours > observed_hours).any() or (antibiotic_hours < 0.0).any():
        raise ValueError("首次抗菌药时间超出患者实际观察窗口")

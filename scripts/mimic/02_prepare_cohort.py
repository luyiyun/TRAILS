from __future__ import annotations

from pathlib import Path

import duckdb

DATA_ROOT = Path("data/real/mimic-iv-3.1")
DATABASE = DATA_ROOT / "derived" / "mimiciv.duckdb"
OUTPUT_ROOT = DATA_ROOT / "derived"

LABELS_SQL = """
SELECT
    i.subject_id,
    i.hadm_id,
    i.stay_id,
    i.intime,
    i.outtime,
    CAST(s.stay_id IS NOT NULL AND COALESCE(s.sepsis3, FALSE) AS INTEGER) AS sepsis3,
    CAST(
        s.stay_id IS NOT NULL
        AND COALESCE(s.sepsis3, FALSE)
        AND GREATEST(s.suspected_infection_time, s.sofa_time)
            BETWEEN i.intime - INTERVAL '6 hours' AND i.intime + INTERVAL '24 hours'
        AS INTEGER
    ) AS early_sepsis3,
    s.suspected_infection_time,
    s.sofa_time,
    GREATEST(s.suspected_infection_time, s.sofa_time) AS sepsis_onset_time,
    s.sofa_score
FROM mimiciv_icu.icustays AS i
LEFT JOIN mimiciv_derived.sepsis3 AS s
    ON i.subject_id = s.subject_id AND i.stay_id = s.stay_id
"""

COHORT_SQL = """
WITH eligible AS (
    SELECT
        l.*,
        age.age,
        p.gender,
        ad.race,
        ad.deathtime,
        p.dod,
        ROW_NUMBER() OVER (
            PARTITION BY l.subject_id
            ORDER BY l.sepsis_onset_time, l.intime, l.stay_id
        ) AS early_stay_number
    FROM trails_sepsis3_labels AS l
    INNER JOIN mimiciv_derived.age AS age
        ON l.subject_id = age.subject_id AND l.hadm_id = age.hadm_id
    INNER JOIN mimiciv_hosp.admissions AS ad
        ON l.subject_id = ad.subject_id AND l.hadm_id = ad.hadm_id
    INNER JOIN mimiciv_hosp.patients AS p
        ON l.subject_id = p.subject_id
    WHERE l.early_sepsis3 = 1 AND age.age >= 18
), selected AS (
    SELECT *, intime + INTERVAL '48 hours' AS landmark_time
    FROM eligible
    WHERE early_stay_number = 1
), landmark AS (
    SELECT
        *,
        CASE
            WHEN deathtime IS NOT NULL THEN deathtime
            -- 院外死亡仅精确到日期，以次日零点作为该日期的右端点。
            WHEN dod IS NOT NULL THEN CAST(dod AS TIMESTAMP) + INTERVAL '1 day'
        END AS outcome_death_time,
        CASE
            WHEN deathtime IS NOT NULL THEN 'hospital_datetime'
            WHEN dod IS NOT NULL THEN 'date_right_endpoint'
            ELSE 'none'
        END AS death_time_source
    FROM selected
)
SELECT
    subject_id,
    hadm_id,
    stay_id,
    age,
    gender,
    race,
    intime,
    outtime,
    sepsis_onset_time,
    suspected_infection_time,
    sofa_time,
    sofa_score,
    landmark_time,
    deathtime AS hospital_deathtime,
    dod,
    outcome_death_time,
    death_time_source,
    DATE_DIFF('second', intime, outtime) / 3600.0 AS icu_los_hours,
    CASE
        WHEN outcome_death_time <= landmark_time + INTERVAL '28 days'
        THEN DATE_DIFF('second', landmark_time, outcome_death_time) / 86400.0
        ELSE 28.0
    END AS survival_time,
    CASE
        WHEN outcome_death_time <= landmark_time + INTERVAL '28 days' THEN 1
        ELSE 0
    END AS event
FROM landmark
WHERE outcome_death_time IS NULL OR outcome_death_time > landmark_time
"""


class MimicCohortExporter:
    def __init__(self) -> None:
        self.database = DATABASE.resolve()
        self.output_root = OUTPUT_ROOT.resolve()

    def run(self) -> None:
        if not self.database.is_file():
            raise FileNotFoundError(
                f"缺少 {self.database}；请先运行 scripts/mimic/01_build_sepsis.py"
            )
        self.output_root.mkdir(parents=True, exist_ok=True)
        connection = duckdb.connect(str(self.database))
        self._require_tables(connection)

        connection.execute("CREATE OR REPLACE TABLE trails_sepsis3_labels AS " + LABELS_SQL)
        connection.execute("CREATE OR REPLACE TABLE trails_cohort_primary AS " + COHORT_SQL)
        connection.execute(
            "CREATE OR REPLACE TABLE trails_cohort_icu_los_ge_48h AS "
            "SELECT * FROM trails_cohort_primary "
            "WHERE outtime >= landmark_time"
        )

        outputs = {
            "trails_sepsis3_labels": self.output_root / "sepsis3_labels.csv",
            "trails_cohort_primary": self.output_root / "cohort_primary.csv",
            "trails_cohort_icu_los_ge_48h": self.output_root / "cohort_icu_los_ge_48h.csv",
        }
        for table, path in outputs.items():
            escaped_path = str(path).replace("'", "''")
            connection.execute(f"COPY {table} TO '{escaped_path}' (HEADER, FORMAT CSV)")
            count = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            assert count is not None
            print(f"{path}: {count[0]} rows")
        connection.close()

    @staticmethod
    def _require_tables(connection: duckdb.DuckDBPyConnection) -> None:
        required = {
            "mimiciv_icu.icustays",
            "mimiciv_hosp.admissions",
            "mimiciv_hosp.patients",
            "mimiciv_derived.age",
            "mimiciv_derived.sepsis3",
        }
        available = {
            f"{schema}.{table}"
            for schema, table in connection.execute(
                "SELECT table_schema, table_name FROM information_schema.tables"
            ).fetchall()
        }
        missing = sorted(required - available)
        if missing:
            raise ValueError(f"DuckDB 缺少必要表：{missing}")


def main() -> None:
    MimicCohortExporter().run()


if __name__ == "__main__":
    main()

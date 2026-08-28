from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DATA_ROOT = Path("data/real/mimic-iv-3.1")
DATABASE = DATA_ROOT / "derived" / "mimiciv.duckdb"
OUTPUT_ROOT = DATA_ROOT / "derived" / "trails_case"
MIN_PATIENT_COVERAGE = 0.50

# 首版聚类输入只含生理与器官功能，治疗变量留给亚型解释，避免聚类主要反映治疗选择。
FEATURE_SOURCES: dict[str, tuple[str, tuple[str, ...]]] = {
    "vitalsign": (
        "stay_id",
        ("heart_rate", "sbp", "mbp", "resp_rate", "temperature", "spo2"),
    ),
    "gcs": ("stay_id", ("gcs",)),
    "complete_blood_count": ("hadm_id", ("wbc", "hemoglobin", "platelet")),
    "chemistry": (
        "hadm_id",
        ("aniongap", "bicarbonate", "bun", "creatinine", "glucose", "sodium", "potassium"),
    ),
    "enzyme": ("hadm_id", ("bilirubin_total",)),
    "coagulation": ("hadm_id", ("inr",)),
    "bg": ("hadm_id", ("lactate", "ph", "pao2fio2ratio")),
    "urine_output_rate": ("stay_id", ("uo_mlkghr_6hr",)),
}
FEATURE_ORDER = tuple(feature for _, features in FEATURE_SOURCES.values() for feature in features)


def main() -> None:
    database = DATABASE.resolve()
    output_root = OUTPUT_ROOT.resolve()
    if not database.is_file():
        raise FileNotFoundError(f"缺少 {database}；请先完成 MIMIC 队列构建")
    output_root.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(database), read_only=True)

    required = {"trails_cohort_primary"} | {f"mimiciv_derived.{table}" for table in FEATURE_SOURCES}
    available = {
        table if schema == "main" else f"{schema}.{table}"
        for schema, table in connection.execute(
            "SELECT table_schema, table_name FROM information_schema.tables"
        ).fetchall()
    }
    if missing := sorted(required - available):
        raise ValueError(f"DuckDB 缺少必要表：{missing}")

    patients = connection.execute(
        "SELECT stay_id AS patient_id, survival_time, event, "
        "CAST(outtime < landmark_time AS INTEGER) AS left_icu_before_48h "
        "FROM trails_cohort_primary ORDER BY stay_id"
    ).df()
    long_frames: list[pd.DataFrame] = []
    for table, (key, features) in FEATURE_SOURCES.items():
        subject_join = " AND c.subject_id = t.subject_id" if key == "hadm_id" else ""
        columns = ", ".join(f"t.{feature}" for feature in features)
        sql = f"""
            SELECT
                c.stay_id AS patient_id,
                DATE_DIFF('second', c.intime, t.charttime) / 3600.0 AS time,
                {columns}
            FROM trails_cohort_primary AS c
            INNER JOIN mimiciv_derived.{table} AS t
                ON c.{key} = t.{key}{subject_join}
                AND t.charttime >= c.intime
                AND t.charttime <= LEAST(c.outtime, c.landmark_time)
        """
        wide = connection.execute(sql).df()
        long_frames.append(
            wide.melt(
                id_vars=["patient_id", "time"],
                value_vars=list(features),
                var_name="feature",
                value_name="value",
            ).dropna(subset=["value"])
        )
    connection.close()

    observations = pd.concat(long_frames, ignore_index=True)
    observations["value"] = pd.to_numeric(observations["value"], errors="raise")
    if not np.isfinite(observations["value"].to_numpy(dtype=float)).all():
        raise ValueError("纵向观测包含非有限数值")

    # 同一变量在完全相同时间可能来自多个标本；取中位数后仍保留原始异步时间轴。
    observations = (
        observations.groupby(["patient_id", "time", "feature"], as_index=False)["value"]
        .median()
        .sort_values(["patient_id", "time", "feature"])  # pyright: ignore
    )
    if not observations["time"].between(0.0, 48.0).all():
        raise ValueError("观测时间超出 ICU 入科后 0–48 小时")
    observed_patients = set(observations["patient_id"].unique())
    expected_patients = set(patients["patient_id"].unique())
    if observed_patients != expected_patients:
        raise ValueError(f"有 {len(expected_patients - observed_patients)} 名患者没有任何观测")

    grouped = observations.groupby("feature", sort=False)["value"]
    feature_summary = grouped.agg(n_observations="size").reset_index()
    patient_counts = observations.groupby("feature")["patient_id"].nunique()
    quantiles = grouped.quantile(np.array([0.01, 0.5, 0.99])).unstack()
    quantiles.columns = ["q01", "median", "q99"]
    feature_summary = feature_summary.join(patient_counts, on="feature").join(
        quantiles, on="feature"
    )
    feature_summary = feature_summary.rename(columns={"patient_id": "n_patients"})
    feature_summary["coverage_percent"] = (
        100.0 * feature_summary["n_patients"] / len(patients)
    ).round(2)
    feature_summary["feature"] = pd.Categorical(
        feature_summary["feature"], categories=FEATURE_ORDER, ordered=True
    )
    feature_summary = feature_summary.sort_values("feature")
    if (feature_summary["coverage_percent"] < 100 * MIN_PATIENT_COVERAGE).any():
        raise ValueError("至少一个预设变量的患者覆盖率低于 50%")

    patients.to_csv(output_root / "patients.csv", index=False)
    observations.to_csv(output_root / "observations.csv", index=False)
    feature_summary.to_csv(output_root / "feature_summary.csv", index=False)
    summary = {
        "dataset": "MIMIC-IV v3.1 primary early Sepsis-3 cohort",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "n_patients": len(patients),
        "n_observations": len(observations),
        "feature_order": list(FEATURE_ORDER),
        "n_features": len(FEATURE_ORDER),
        "time_window": "ICU intime to min(outtime, intime + 48 hours)",
        "time_unit": "hours since ICU intime",
        "duplicate_rule": "median within identical patient-time-feature",
        "missing_rule": "no imputation; absence is represented by TRAILS mask",
        "value_scale": (
            "raw; train-only winsorization and standardization occur in scripts/mimic/07_case.py"
        ),
    }
    (output_root / "extraction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"TRAILS case 输入已保存至 {output_root}: "
        f"{len(patients)} patients, {len(observations)} observations, {len(FEATURE_ORDER)} features"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

DATA_ROOT = Path("data/real/mimic-iv-3.1")
DERIVED_ROOT = DATA_ROOT / "derived"
CASE_ROOT = DERIVED_ROOT / "trails_case"
OUTPUT_ROOT = DERIVED_ROOT / "model_readiness"
TIME_EDGES = (-1e-6, 6.0, 12.0, 24.0, 48.000001)
TIME_LABELS = ("0-6h", "6-12h", "12-24h", "24-48h")
TIME_STARTS = (0.0, 6.0, 12.0, 24.0)


def _quantile(values: Any, probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), probability))


def _sequence_row(group: str, frame: pd.DataFrame) -> dict[str, float | int | str]:
    return {
        "landmark_status": group,
        "n_patients": len(frame),
        "mortality_28d_percent": round(100.0 * float(frame["event"].mean()), 2),  # pyright: ignore
        "icu_los_hours_median": round(float(frame["icu_los_hours"].median()), 2),  # pyright: ignore
        "n_observations_median": _quantile(frame["n_observations"], 0.5),
        "n_observations_p99": _quantile(frame["n_observations"], 0.99),
        "n_observations_max": int(frame["n_observations"].max()),  # pyright: ignore
        "n_visits_median": _quantile(frame["n_visits"], 0.5),
        "n_visits_p90": _quantile(frame["n_visits"], 0.9),
        "n_visits_p99": _quantile(frame["n_visits"], 0.99),
        "n_visits_max": int(frame["n_visits"].max()),  # pyright: ignore
        "last_time_median": round(float(frame["last_time"].median()), 2),  # pyright: ignore
        "missing_fraction_median": round(float(frame["missing_fraction"].median()), 4),  # pyright: ignore
    }


def main() -> None:
    paths = {
        "cohort": DERIVED_ROOT / "cohort_primary.csv",
        "patients": CASE_ROOT / "patients.csv",
        "observations": CASE_ROOT / "observations.csv",
        "extraction": CASE_ROOT / "extraction_summary.json",
    }
    if missing := [str(path.resolve()) for path in paths.values() if not path.is_file()]:
        raise FileNotFoundError(f"缺少建模输入，请先完成队列和特征提取：{missing}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    extraction = json.loads(paths["extraction"].read_text(encoding="utf-8"))
    n_features = int(extraction["n_features"])
    cohort = pd.read_csv(paths["cohort"], usecols=["stay_id", "icu_los_hours", "event"])
    patients = pd.read_csv(paths["patients"], usecols=["patient_id"])
    observations = pd.read_csv(
        paths["observations"],
        usecols=["patient_id", "time", "feature"],
        dtype={"patient_id": "int64", "time": "float32", "feature": "category"},
    )
    cohort = cohort.rename(columns={"stay_id": "patient_id"})
    cohort["landmark_status"] = np.where(
        cohort["icu_los_hours"] >= 48.0,
        "In ICU at 48h",
        "Left ICU before 48h",
    )
    cohort["window_end"] = cohort["icu_los_hours"].clip(upper=48.0)
    if len(cohort) != len(patients) or set(cohort["patient_id"]) != set(patients["patient_id"]):
        raise ValueError("队列与建模患者文件不一致")
    # n_visits 是 loader 展开后的真实序列长度，即该患者全部变量时间戳的并集。
    patient_metrics = (
        observations.groupby("patient_id", observed=True)
        .agg(
            n_observations=("feature", "size"),
            n_visits=("time", "nunique"),
            last_time=("time", "max"),
        )
        .reset_index()
        .merge(cohort, on="patient_id", validate="one_to_one")
    )
    patient_metrics["missing_fraction"] = 1.0 - patient_metrics["n_observations"] / (
        patient_metrics["n_visits"] * n_features
    )
    if len(patient_metrics) != len(cohort):
        raise ValueError("至少一名队列患者缺少纵向观测")

    patient_groups = [
        ("All", patient_metrics),
        *list(patient_metrics.groupby("landmark_status", sort=True)),
    ]
    sequence_summary = pd.DataFrame(
        [_sequence_row(str(status), frame) for status, frame in patient_groups]
    )

    observations["landmark_status"] = observations["patient_id"].map(
        cohort.set_index("patient_id")["landmark_status"]  # pyright: ignore
    )
    observations["time_bin"] = pd.cut(
        observations["time"], bins=TIME_EDGES, labels=TIME_LABELS, right=False
    )
    if observations[["landmark_status", "time_bin"]].isna().any().any():  # pyright: ignore
        raise ValueError("存在无法映射队列状态或时间分箱的观测")

    observed = observations.drop_duplicates(["patient_id", "time_bin", "feature"])
    coverage = (
        observed.groupby(["landmark_status", "time_bin", "feature"], observed=True)
        .size()
        .rename("n_patients_observed")  # pyright: ignore
        .reset_index()
    )
    all_coverage = (
        observed.groupby(["time_bin", "feature"], observed=True)
        .size()
        .rename("n_patients_observed")  # pyright: ignore
        .reset_index()
        .assign(landmark_status="All")
    )
    coverage = pd.concat([all_coverage, coverage], ignore_index=True)

    counts = (
        observations.groupby(["landmark_status", "time_bin", "feature"], observed=True)
        .size()
        .rename("n_observations")  # pyright: ignore
        .reset_index()
    )
    all_counts = (
        observations.groupby(["time_bin", "feature"], observed=True)
        .size()
        .rename("n_observations")  # pyright: ignore
        .reset_index()
        .assign(landmark_status="All")
    )
    counts = pd.concat([all_counts, counts], ignore_index=True)

    denominator_rows: list[dict[str, int | str]] = []
    for status, frame in patient_groups:
        for time_bin, start in zip(TIME_LABELS, TIME_STARTS, strict=True):
            denominator_rows.append(
                {
                    "landmark_status": str(status),
                    "time_bin": time_bin,
                    "n_patients_group": len(frame),
                    "n_patients_window_available": int((frame["window_end"] >= start).sum()),
                }
            )
    denominators = pd.DataFrame(denominator_rows)
    time_coverage = coverage.merge(counts, on=["landmark_status", "time_bin", "feature"])
    time_coverage = time_coverage.merge(
        denominators, on=["landmark_status", "time_bin"], validate="many_to_one"
    )
    time_coverage["coverage_all_percent"] = (
        100.0 * time_coverage["n_patients_observed"] / time_coverage["n_patients_group"]
    ).round(2)
    time_coverage["coverage_available_percent"] = (
        100.0 * time_coverage["n_patients_observed"] / time_coverage["n_patients_window_available"]
    ).round(2)

    correlation_rows: list[dict[str, float | str]] = []
    for status, frame in patient_groups:
        if frame["window_end"].nunique() < 2:
            continue
        for metric in ("n_observations", "n_visits", "last_time"):
            coefficient, p_value = stats.spearmanr(frame["window_end"], frame[metric])
            correlation_rows.append(
                {
                    "landmark_status": str(status),
                    "metric": metric,
                    "spearman_r": float(coefficient),  # pyright: ignore
                    "p_value": float(p_value),  # pyright: ignore
                }
            )
    correlations = pd.DataFrame(correlation_rows)

    sequence_summary.to_csv(OUTPUT_ROOT / "sequence_summary.csv", index=False)
    time_coverage.to_csv(OUTPUT_ROOT / "time_bin_feature_coverage.csv", index=False)
    correlations.to_csv(OUTPUT_ROOT / "length_correlations.csv", index=False)
    payload = {
        "dataset": "MIMIC-IV v3.1 primary early Sepsis-3 cohort",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "n_patients": len(patient_metrics),
        "n_features": n_features,
        "n_observations": len(observations),
        "aligned_sequence_definition": "unique timestamps across all features per patient",
        "outputs": [
            "sequence_summary.csv",
            "time_bin_feature_coverage.csv",
            "length_correlations.csv",
        ],
    }
    (OUTPUT_ROOT / "readiness_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"建模就绪性聚合审计已保存至 {OUTPUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

DATA_ROOT = Path("data/real/mimic-iv-3.1")
DERIVED_ROOT = DATA_ROOT / "derived"
OUTPUT_ROOT = DERIVED_ROOT / "eda"


def _raw_csv(module: str, name: str) -> Path:
    for suffix in (".csv.gz", ".csv"):
        path = DATA_ROOT / module / f"{name}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"缺少 MIMIC-IV 原始表：{module}/{name}.csv[.gz]")


def _mean(series: Any) -> float:
    values = np.asarray(series, dtype=float)
    return float(stats.tmean(np.ma.masked_invalid(values))) if values.size else np.nan


def _median(series: Any) -> float:
    values = np.asarray(series, dtype=float)
    return float(np.nanmedian(values)) if values.size else np.nan


def _percent(numerator: Any, denominator: float) -> float:
    return round(float(np.divide(100.0 * numerator, denominator)), 2) if denominator else np.nan


def _stage(order: int, name: str, frame: pd.DataFrame) -> dict[str, int | str]:
    return dict(
        stage_order=order,
        stage=name,
        n_stays=len(frame),
        n_subjects=len(frame["subject_id"].drop_duplicates()),
    )


def _cohort_summary(name: str, frame: pd.DataFrame) -> dict[str, float | int | str]:
    left_icu = (frame["outtime"] < frame["landmark_time"]).sum()
    return {
        "cohort": name,
        "n": len(frame),
        "age_mean": round(_mean(frame["age"]), 2),
        "age_median": round(_median(frame["age"]), 2),
        "female_percent": _percent((frame["gender"] == "F").sum(), len(frame)),
        "sofa_median": round(_median(frame["sofa_score"]), 2),
        "icu_los_hours_median": round(_median(frame["icu_los_hours"]), 2),
        "mortality_28d_percent": _percent(frame["event"].sum(), len(frame)),
        "left_icu_before_48h_percent": _percent(left_icu, len(frame)),
        "sofa_missing_percent": _percent(frame["sofa_score"].isna().sum(), len(frame)),
    }


def main() -> None:
    paths = {
        "labels": DERIVED_ROOT / "sepsis3_labels.csv",
        "primary": DERIVED_ROOT / "cohort_primary.csv",
        "sensitivity": DERIVED_ROOT / "cohort_icu_los_ge_48h.csv",
    }
    if missing := [str(path.resolve()) for path in paths.values() if not path.is_file()]:
        raise FileNotFoundError(f"缺少衍生 CSV，请先运行前两个 MIMIC 脚本：{missing}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(paths["labels"], parse_dates=["intime", "sepsis_onset_time"])
    primary = pd.read_csv(paths["primary"], parse_dates=["outtime", "landmark_time"])
    sensitivity = pd.read_csv(paths["sensitivity"], parse_dates=["outtime", "landmark_time"])
    patients = pd.read_csv(
        _raw_csv("hosp", "patients"), usecols=["subject_id", "anchor_age", "anchor_year"]
    )
    admissions = pd.read_csv(
        _raw_csv("hosp", "admissions"),
        usecols=["subject_id", "hadm_id", "admittime"],
        parse_dates=["admittime"],
    )

    # 标签文件不含年龄；依照 MIMIC-IV 年龄定义恢复每次住院的年龄。
    adult_early = (
        labels.loc[labels["early_sepsis3"] == 1]
        .merge(admissions, on=["subject_id", "hadm_id"], validate="many_to_one")
        .merge(patients, on="subject_id", validate="many_to_one")
    )
    adult_early["age"] = (
        adult_early["anchor_age"] + adult_early["admittime"].dt.year - adult_early["anchor_year"]
    )
    adult_first = (
        adult_early.loc[adult_early["age"] >= 18]
        .sort_values(["subject_id", "sepsis_onset_time", "intime", "stay_id"])
        .drop_duplicates("subject_id", keep="first")
    )
    sepsis = labels.loc[labels["sepsis3"] == 1]
    early = labels.loc[labels["early_sepsis3"] == 1]
    flow = pd.DataFrame(
        [
            _stage(1, "All ICU stays", labels),
            _stage(2, "Sepsis-3 stays", sepsis),
            _stage(3, "Early Sepsis-3 stays", early),
            _stage(4, "Adult first early stay", adult_first),
            _stage(5, "Alive at 48h landmark", primary),
            _stage(6, "ICU LOS >= 48h", sensitivity),
        ]
    )

    prevalence = pd.DataFrame(
        data=[
            [1, "Sepsis-3 among all ICU stays", len(sepsis), len(labels)],
            [2, "Early Sepsis-3 among all ICU stays", len(early), len(labels)],
            [3, "Early Sepsis-3 among Sepsis-3 stays", len(early), len(sepsis)],
        ],
        columns=["metric_order", "metric", "numerator", "denominator"],
    )
    prevalence["percent"] = np.round(100.0 * prevalence["numerator"] / prevalence["denominator"], 2)
    cohort = pd.DataFrame(
        [_cohort_summary("Primary", primary), _cohort_summary("ICU LOS >= 48h", sensitivity)]
    )

    landmark = primary.assign(
        landmark_status=np.where(
            primary["outtime"] < primary["landmark_time"],
            "Left ICU before 48h",
            "In ICU at 48h",
        )
    )
    landmark_summary = (
        landmark.groupby("landmark_status", sort=True)
        .agg(
            n=("subject_id", "size"),
            age_mean=("age", lambda values: round(_mean(values), 2)),
            sofa_median=("sofa_score", lambda values: round(_median(values), 2)),
            mortality_28d_percent=("event", lambda values: round(100 * _mean(values), 2)),
        )
        .reset_index()
    )

    races = pd.concat(
        [primary.assign(cohort="Primary"), sensitivity.assign(cohort="ICU LOS >= 48h")]
    )
    races["race"] = races["race"].fillna("MISSING")
    race = races.groupby(["cohort", "race"], as_index=False).size()
    race.columns = ["cohort", "race", "n"]
    race["percent"] = (100.0 * race["n"] / race.groupby("cohort")["n"].transform("sum")).round(2)
    race = race.sort_values(["cohort", "n"], ascending=[True, False])  # pyright: ignore

    frames = {
        "cohort_flow": flow,
        "label_prevalence": prevalence,
        "cohort_summary": cohort,
        "landmark_status": landmark_summary,
        "race_distribution": race,
    }
    for name, frame in frames.items():
        frame.to_csv(OUTPUT_ROOT / f"{name}.csv", index=False)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.barh(flow["stage"], flow["n_stays"], color="#4477AA")
    axis.invert_yaxis()
    axis.set(xlabel="Number of ICU stays", title="MIMIC-IV Sepsis cohort flow")
    for index, value in enumerate(flow["n_stays"]):
        axis.text(value, index, f" {int(value):,}", va="center")
    figure.tight_layout()
    figure.savefig(OUTPUT_ROOT / "cohort_flow.png", dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    axes[0].bar(cohort["cohort"], cohort["n"], color="#4477AA")
    axes[0].set_ylabel("Patients")
    axes[1].bar(cohort["cohort"], cohort["mortality_28d_percent"], color="#CC6677")
    axes[1].set_ylabel("28-day mortality (%)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    figure.savefig(OUTPUT_ROOT / "cohort_comparison.png", dpi=200)
    plt.close(figure)

    payload = {
        "dataset": "MIMIC-IV v3.1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "tables": {name: frame.to_dict(orient="records") for name, frame in frames.items()},
    }
    (OUTPUT_ROOT / "eda_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"聚合 EDA 已保存至 {OUTPUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()

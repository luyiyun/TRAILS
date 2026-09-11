from __future__ import annotations

# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false
from pathlib import Path
from typing import Any, cast

import hydra
import matplotlib
import numpy as np
import pandas as pd
from omegaconf import DictConfig

matplotlib.use("Agg")
import seaborn as sns  # noqa: E402
from lifelines import KaplanMeierFitter  # noqa: E402
from lifelines.plotting import add_at_risk_counts  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


def _percent(numerator: int | float, denominator: int | float) -> float:
    return 100.0 * numerator / denominator if denominator else np.nan


def _category(series: pd.Series) -> pd.Series:
    return cast(
        pd.Series,
        series.fillna("MISSING").astype("string").str.strip().replace("", "MISSING"),
    )


def _save_png(figure: Figure, output_dir: Path, name: str, dpi: int) -> None:
    figure.savefig(output_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _add_group_separators(axes: Any, feature_groups: pd.Series) -> None:
    axis_list = np.atleast_1d(axes).ravel()
    groups = feature_groups.to_numpy()
    changes = np.flatnonzero(groups[1:] != groups[:-1]) + 1
    for axis in axis_list:
        for position in changes:
            axis.axhline(position - 0.5, color="white", linewidth=1.2)


@hydra.main(config_path="../../configs", config_name="crc_yunnan/eda", version_base="1.3")
def main(config: DictConfig) -> None:

    # 参数配置解析
    landmark_days = sorted(int(value) for value in config.trajectory.landmark_days)
    time_bin_days = int(config.trajectory.time_bin_days)
    annual_segment_days = int(config.trajectory.annual_segment_days)
    max_landmark = max(landmark_days)
    dpi = int(config.plot.dpi)
    categorical_baseline = [str(name) for name in config.columns.categorical_baseline]

    # 设置绘图风格和字体，确保中文显示正常。
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    font_family = next(
        (str(name) for name in config.plot.font_families if str(name) in available_fonts),
        "DejaVu Sans",
    )
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "font.family": font_family,
            "axes.unicode_minus": False,
            "figure.dpi": int(config.plot.dpi),
        }
    )

    # ==================================================================================
    # 一、读取预处理后的患者表与纵向观测表，固定本次描述性EDA的数据边界。
    # ==================================================================================
    patient_path = Path(str(config.paths.patients_csv)).resolve()
    observation_path = Path(str(config.paths.observations_csv)).resolve()
    missing_paths = [str(path) for path in (patient_path, observation_path) if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"缺少CRC云南EDA输入：{missing_paths}")

    output_dir = Path(str(config.paths.output_dir)).resolve()
    if bool(config.quality.refuse_overwrite) and output_dir.exists():
        raise FileExistsError(f"拒绝覆盖既有EDA目录：{output_dir}")

    numeric_baseline = [str(name) for name in config.columns.numeric_baseline]
    # 在读取时明确数值类型；非法数值沿用pandas默认行为，直接抛出异常。
    patients = pd.read_csv(
        patient_path,
        dtype={
            "patient_id": "string",
            "dfs_time_days": "float64",
            "dfs_event": "float64",
            "os_time_days": "float64",
            "os_event": "float64",
            **dict.fromkeys(numeric_baseline, "float64"),
        },
        parse_dates=["surgery_date", "last_followup_date"],
    )
    observations = pd.read_csv(
        observation_path,
        dtype={
            "patient_id": "string",
            "feature": "string",
            "feature_group": "string",
            "time_days": "float32",
            "value": "float32",
        },
    )

    # 保证EDA输入表格包含必要字段，避免后续分析中出现KeyError。
    required_patient_columns = {
        "patient_id",
        "surgery_date",
        "last_followup_date",
        "dfs_time_days",
        "dfs_event",
        "os_time_days",
        "os_event",
        *[str(name) for name in config.columns.numeric_baseline],
        *[str(name) for name in config.columns.categorical_baseline],
    }
    required_observation_columns = {
        "patient_id",
        "feature",
        "feature_group",
        "time_days",
        "value",
    }
    missing_patient_columns = sorted(required_patient_columns - set(patients.columns))
    missing_observation_columns = sorted(required_observation_columns - set(observations.columns))
    if missing_patient_columns or missing_observation_columns:
        raise ValueError(
            "EDA输入缺少必要字段："
            f"patients={missing_patient_columns}, observations={missing_observation_columns}"
        )

    # 确保输出目录存在，若拒绝覆盖则在已存在时抛出异常。
    output_dir.mkdir(parents=True, exist_ok=not bool(config.quality.refuse_overwrite))

    # 计算日历随访天数和手术年份，便于后续分析。
    patients["calendar_followup_days"] = (
        patients["last_followup_date"] - patients["surgery_date"]
    ).dt.days
    patients["surgery_year"] = patients["surgery_date"].dt.year.astype("Int64")

    # 计算患者ID和观测ID的集合，并找出有观测记录的患者ID。
    patient_ids = set(patients["patient_id"].dropna())
    observation_ids = set(observations["patient_id"].dropna())
    observed_patient_ids = patient_ids & observation_ids

    # 根据观测表中的特征和特征组信息，生成特征元数据表格，并按特征组和特征名称排序。
    feature_metadata = (
        observations[["feature", "feature_group"]]
        .dropna(subset=["feature"])
        .assign(feature_group=lambda frame: _category(frame["feature_group"]))
        .groupby("feature", observed=True)["feature_group"]
        .agg(lambda values: values.mode().iloc[0] if not values.mode().empty else "MISSING")
        .rename("feature_group")
        .reset_index()
        .sort_values(["feature_group", "feature"], kind="stable")
        .reset_index(drop=True)
    )
    ordered_features = feature_metadata["feature"].astype(str).tolist()
    ordered_groups = feature_metadata.set_index("feature")["feature_group"].reindex(
        ordered_features
    )

    # ==================================================================================
    # 二、用三张小型表格概括队列边界、数据质量事实及基线临床特征。
    # ==================================================================================
    cohort_rows: list[dict[str, Any]] = []

    def add_cohort_row(
        section: str,
        metric: str,
        value: Any,
        denominator: int | None = None,
        unit: str | None = None,
    ) -> None:
        numeric_value = float(value) if isinstance(value, (int, float, np.number)) else np.nan
        cohort_rows.append(
            {
                "section": section,
                "metric": metric,
                "value": value,
                "denominator": denominator,
                "percent": (
                    _percent(numeric_value, denominator)
                    if denominator is not None and np.isfinite(numeric_value)
                    else np.nan
                ),
                "unit": unit,
            }
        )

    add_cohort_row("size", "patients", len(patients), unit="patients")
    add_cohort_row("size", "observations", len(observations), unit="records")
    add_cohort_row("size", "features", observations["feature"].nunique(), unit="features")
    add_cohort_row("size", "feature_groups", observations["feature_group"].nunique(), unit="groups")
    add_cohort_row(
        "observation_linkage",
        "patients_with_observations",
        len(observed_patient_ids),
        len(patients),
        "patients",
    )
    add_cohort_row(
        "observation_linkage",
        "patients_without_observations",
        len(patient_ids - observation_ids),
        len(patients),
        "patients",
    )
    for metric, series in (
        ("surgery_date", patients["surgery_date"]),
        ("last_followup_date", patients["last_followup_date"]),
    ):
        add_cohort_row("calendar_range", f"{metric}_minimum", series.min(), unit="date")
        add_cohort_row("calendar_range", f"{metric}_maximum", series.max(), unit="date")
    add_cohort_row(
        "observation_time",
        "relative_time_minimum",
        observations["time_days"].min(),
        unit="days_from_surgery",
    )
    add_cohort_row(
        "observation_time",
        "relative_time_maximum",
        observations["time_days"].max(),
        unit="days_from_surgery",
    )
    pd.DataFrame(cohort_rows).to_csv(output_dir / "cohort_summary.csv", index=False)

    quality_rows: list[dict[str, Any]] = []

    def add_quality_row(
        table: str,
        check: str,
        n_affected: int,
        denominator: int,
        definition: str,
    ) -> None:
        quality_rows.append(
            {
                "table": table,
                "check": check,
                "n_affected": int(n_affected),
                "denominator": int(denominator),
                "percent": _percent(n_affected, denominator),
                "definition": definition,
            }
        )

    patient_quality_checks = [
        ("missing_patient_id", patients["patient_id"].isna(), "patient_id is missing"),
        (
            "duplicate_patient_id_rows",
            patients["patient_id"].duplicated(keep=False),
            "rows belonging to a patient_id appearing more than once",
        ),
        ("exact_duplicate_rows", patients.duplicated(), "rows exactly duplicated"),
        ("missing_surgery_date", patients["surgery_date"].isna(), "surgery_date is missing"),
        (
            "missing_last_followup_date",
            patients["last_followup_date"].isna(),
            "last_followup_date is missing",
        ),
        (
            "followup_before_surgery",
            patients["last_followup_date"] < patients["surgery_date"],
            "last_followup_date is earlier than surgery_date",
        ),
        (
            "invalid_dfs_event_code",
            patients["dfs_event"].notna() & ~patients["dfs_event"].isin([0, 1]),
            "dfs_event is not 0 or 1",
        ),
        (
            "invalid_os_event_code",
            patients["os_event"].notna() & ~patients["os_event"].isin([0, 1]),
            "os_event is not 0 or 1",
        ),
        ("missing_dfs_time", patients["dfs_time_days"].isna(), "dfs_time_days is missing"),
        ("missing_os_time", patients["os_time_days"].isna(), "os_time_days is missing"),
        ("negative_dfs_time", patients["dfs_time_days"] < 0, "dfs_time_days is negative"),
        ("negative_os_time", patients["os_time_days"] < 0, "os_time_days is negative"),
        (
            "dfs_time_exceeds_os_time",
            patients["dfs_time_days"] > patients["os_time_days"],
            "dfs_time_days is greater than os_time_days",
        ),
    ]
    for check, affected, definition in patient_quality_checks:
        add_quality_row("patients", check, int(affected.sum()), len(patients), definition)
    add_quality_row(
        "patients",
        "patient_without_observation",
        len(patient_ids - observation_ids),
        len(patient_ids),
        "patient_id has no linked longitudinal observation",
    )

    for column in ("patient_id", "feature", "feature_group", "time_days", "value"):
        add_quality_row(
            "observations",
            f"missing_{column}",
            int(observations[column].isna().sum()),
            len(observations),
            f"{column} is missing",
        )
    add_quality_row(
        "observations",
        "exact_duplicate_rows",
        int(observations.duplicated().sum()),
        len(observations),
        "rows exactly duplicated after preprocessing",
    )
    observation_keys = ["patient_id", "time_days", "feature"]
    key_sizes = observations.groupby(observation_keys, dropna=False, observed=True).size()
    add_quality_row(
        "observations",
        "duplicate_patient_time_feature_groups",
        int((key_sizes > 1).sum()),
        len(key_sizes),
        "patient_id, time_days, and feature identify more than one row",
    )
    value_counts = observations.groupby(observation_keys, dropna=False, observed=True)[
        "value"
    ].nunique(dropna=False)
    add_quality_row(
        "observations",
        "conflicting_value_groups",
        int((value_counts > 1).sum()),
        len(value_counts),
        "the same patient_id, time_days, and feature has multiple values",
    )
    group_counts = observations.groupby(observation_keys, dropna=False, observed=True)[
        "feature_group"
    ].nunique(dropna=False)
    add_quality_row(
        "observations",
        "conflicting_feature_group_groups",
        int((group_counts > 1).sum()),
        len(group_counts),
        "the same patient_id, time_days, and feature has multiple feature groups",
    )
    add_quality_row(
        "observations",
        "observation_patient_not_in_patient_table",
        len(observation_ids - patient_ids),
        len(observation_ids),
        "observed patient_id is absent from the patient table",
    )
    pd.DataFrame(quality_rows).to_csv(output_dir / "data_quality.csv", index=False)

    baseline_rows: list[dict[str, Any]] = []
    for variable in numeric_baseline:
        values = patients[variable]
        observed = values.dropna()
        baseline_rows.append(
            {
                "variable": variable,
                "type": "numeric",
                "level": pd.NA,
                "summary": (
                    f"{observed.median():.3g} [{observed.quantile(0.25):.3g}, "
                    f"{observed.quantile(0.75):.3g}]; {observed.mean():.3g} "
                    f"({observed.std():.3g}); {observed.min():.3g}–{observed.max():.3g}"
                    if not observed.empty
                    else "NA"
                ),
                "n_available": int(observed.size),
                "n_missing": int(values.isna().sum()),
                "missing_percent": _percent(values.isna().sum(), len(values)),
            }
        )
    for variable in categorical_baseline:
        values = _category(patients[variable])
        missing_count = int((values == "MISSING").sum())
        for level, count in values.value_counts(dropna=False).items():
            baseline_rows.append(
                {
                    "variable": variable,
                    "type": "categorical",
                    "level": str(level),
                    "summary": f"{int(count)} ({_percent(int(count), len(values)):.1f}%)",
                    "n_available": len(values) - missing_count,
                    "n_missing": missing_count,
                    "missing_percent": _percent(missing_count, len(values)),
                }
            )
    pd.DataFrame(baseline_rows).to_csv(output_dir / "baseline_characteristics.csv", index=False)

    # ==================================================================================
    # 三、描述DFS、OS、日历随访的一致性，以及三个landmark可用人群。
    # ==================================================================================
    outcome_rows: list[dict[str, Any]] = []

    def add_outcome_row(
        section: str,
        name: str,
        values: pd.Series,
        events: pd.Series | None = None,
    ) -> None:
        observed = values.dropna()
        event_values = events.dropna() if events is not None else None
        outcome_rows.append(
            {
                "section": section,
                "outcome_or_group": name,
                "n": int(observed.size),
                "events": int((event_values == 1).sum()) if event_values is not None else np.nan,
                "event_percent": (
                    _percent((event_values == 1).sum(), event_values.size)
                    if event_values is not None
                    else np.nan
                ),
                "minimum_days": observed.min(),
                "q25_days": observed.quantile(0.25),
                "median_days": observed.median(),
                "q75_days": observed.quantile(0.75),
                "maximum_days": observed.max(),
                "n_negative": int((observed < 0).sum()),
                "n_zero": int((observed == 0).sum()),
                "n_positive": int((observed > 0).sum()),
            }
        )

    add_outcome_row("outcome_time", "DFS", patients["dfs_time_days"], patients["dfs_event"])
    add_outcome_row("outcome_time", "OS", patients["os_time_days"], patients["os_event"])
    add_outcome_row("calendar_followup", "all", patients["calendar_followup_days"])
    os_calendar_delta = patients["os_time_days"] - patients["calendar_followup_days"]
    for event_code, label in ((0, "censored"), (1, "death")):
        add_outcome_row(
            "os_minus_calendar_followup",
            label,
            os_calendar_delta.loc[patients["os_event"] == event_code],
        )
    add_outcome_row("dfs_minus_os", "all", patients["dfs_time_days"] - patients["os_time_days"])
    pd.DataFrame(outcome_rows).to_csv(output_dir / "outcome_followup_summary.csv", index=False)

    landmark_rows: list[dict[str, Any]] = []
    eligible_by_landmark: dict[int, pd.DataFrame] = {}
    window_observations: dict[int, pd.DataFrame] = {}
    for landmark in landmark_days:
        eligible = patients.loc[patients["dfs_time_days"] > landmark].copy()
        eligible_by_landmark[landmark] = eligible
        eligible_ids = set(eligible["patient_id"].dropna())
        window = observations.loc[
            observations["patient_id"].isin(eligible_ids)
            & observations["time_days"].between(0, landmark, inclusive="both")
        ].copy()
        window_observations[landmark] = window

        per_patient = (
            window.groupby("patient_id", observed=True)
            .agg(
                n_observations=("value", "size"),
                n_dates=("time_days", "nunique"),
                n_features=("feature", "nunique"),
            )
            .reindex(eligible["patient_id"])
            .fillna(0)
        )
        n_bins = (
            window.assign(time_bin=np.floor(window["time_days"] / time_bin_days).astype("Int64"))
            .groupby("patient_id", observed=True)["time_bin"]
            .nunique()
            .reindex(eligible["patient_id"])
            .fillna(0)
        )
        annual_segments = [
            (
                window["time_days"]
                .between(start, min(start + annual_segment_days, landmark), inclusive="left")
                .groupby(window["patient_id"], observed=True)
                .any()
                .reindex(eligible["patient_id"])
                .fillna(False)
            )
            for start in range(0, landmark, annual_segment_days)
        ]
        all_segments = pd.concat(annual_segments, axis=1).all(axis=1)
        remaining_dfs = eligible["dfs_time_days"] - landmark
        remaining_os = eligible["os_time_days"] - landmark
        landmark_rows.append(
            {
                "landmark_days": landmark,
                "landmark_months_30day": landmark / 30,
                "n_total": len(patients),
                "n_dfs_eligible": len(eligible),
                "dfs_eligible_percent": _percent(len(eligible), len(patients)),
                "n_dfs_events_after_landmark": int((eligible["dfs_event"] == 1).sum()),
                "dfs_events_after_landmark_percent": _percent(
                    (eligible["dfs_event"] == 1).sum(), len(eligible)
                ),
                "remaining_dfs_q25_days": remaining_dfs.quantile(0.25),
                "remaining_dfs_median_days": remaining_dfs.median(),
                "remaining_dfs_q75_days": remaining_dfs.quantile(0.75),
                "n_os_eligible": int((patients["os_time_days"] > landmark).sum()),
                "n_os_events_after_landmark": int(
                    ((patients["os_time_days"] > landmark) & (patients["os_event"] == 1)).sum()
                ),
                "n_with_observation": int((per_patient["n_observations"] > 0).sum()),
                "with_observation_percent": _percent(
                    (per_patient["n_observations"] > 0).sum(), len(eligible)
                ),
                "n_with_two_dates": int((per_patient["n_dates"] >= 2).sum()),
                "with_two_dates_percent": _percent(
                    (per_patient["n_dates"] >= 2).sum(), len(eligible)
                ),
                "n_with_two_time_bins": int((n_bins >= 2).sum()),
                "with_two_time_bins_percent": _percent((n_bins >= 2).sum(), len(eligible)),
                "n_with_every_annual_segment": int(all_segments.sum()),
                "with_every_annual_segment_percent": _percent(all_segments.sum(), len(eligible)),
                "median_observations_all_eligible": per_patient["n_observations"].median(),
                "median_dates_all_eligible": per_patient["n_dates"].median(),
                "median_features_all_eligible": per_patient["n_features"].median(),
                "median_time_bins_all_eligible": n_bins.median(),
                "remaining_os_median_days": remaining_os.median(),
            }
        )
    pd.DataFrame(landmark_rows).to_csv(output_dir / "landmark_summary.csv", index=False)

    # ==================================================================================
    # 四、用一幅双面板KM图展示完整队列的DFS和OS，不输出重复的生存估计表。
    # ==================================================================================
    calendar_max = patients.loc[
        patients["calendar_followup_days"] >= 0, "calendar_followup_days"
    ].max()
    if pd.isna(calendar_max) or calendar_max <= 0:
        calendar_max = max(
            patients["dfs_time_days"].quantile(0.99),
            patients["os_time_days"].quantile(0.99),
        )
    display_limit_years = float(calendar_max) / 365.25
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.7))
    for axis, outcome, event, label, title, color in (
        (axes[0], "dfs_time_days", "dfs_event", "DFS", "Disease-free survival", "#0072B2"),
        (axes[1], "os_time_days", "os_event", "OS", "Overall survival", "#D55E00"),
    ):
        valid = patients[outcome].notna() & patients[event].isin([0, 1])
        kmf = KaplanMeierFitter(label=label)
        kmf.fit(
            patients.loc[valid, outcome] / 365.25,
            event_observed=patients.loc[valid, event],
        )
        kmf.plot_survival_function(ax=axis, ci_show=True, color=color, censor_styles=None)
        axis.set_xlim(0, display_limit_years)
        axis.set_ylim(0, 1.03)
        axis.set_xlabel("Years since surgery")
        axis.set_ylabel("Survival probability")
        axis.set_title(f"{title} (n={int(valid.sum()):,})")
        if axis.legend_ is not None:
            axis.legend_.remove()
        add_at_risk_counts(kmf, ax=axis, rows_to_show=["At risk"], ypos=-0.45)
    figure.suptitle(
        "CRC Yunnan survival experience\n"
        "Display range follows the non-negative calendar follow-up span; "
        "recorded maxima are in the outcome table",
        y=1.05,
    )
    figure.subplots_adjust(bottom=0.27, wspace=0.25)
    _save_png(figure, output_dir, "survival_curves", dpi)

    # ==================================================================================
    # 五、展示患者观测密度与相对手术时间分布，观察长期监测数据的实际形态。
    # ==================================================================================
    post_window = observations.loc[observations["time_days"].between(0, max_landmark)]
    patient_observation_counts = (
        post_window.groupby("patient_id", observed=True)
        .agg(
            observations=("value", "size"),
            dates=("time_days", "nunique"),
            features=("feature", "nunique"),
        )
        .reindex(patients["patient_id"])
        .fillna(0)
    )
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    for axis, metric, title, color in (
        (axes[0, 0], "observations", "Observation records per patient", "#0072B2"),
        (axes[0, 1], "dates", "Observed dates per patient", "#009E73"),
        (axes[1, 0], "features", "Observed features per patient", "#CC79A7"),
    ):
        sns.ecdfplot(patient_observation_counts[metric], ax=axis, color=color, linewidth=2)
        if patient_observation_counts[metric].max() > 0:
            axis.set_xscale("symlog", linthresh=1)
        axis.set_xlabel(f"{metric.capitalize()} within 0–{max_landmark} days")
        axis.set_ylabel("Cumulative proportion")
        axis.set_title(title)

    density_bins = np.arange(-360, max_landmark + time_bin_days, time_bin_days)
    interval_index = pd.IntervalIndex.from_breaks(density_bins, closed="left")
    density = (
        observations.loc[observations["time_days"].between(-360, max_landmark)]
        .assign(
            time_bin=lambda frame: pd.cut(
                frame["time_days"], bins=density_bins, right=False, include_lowest=True
            )
        )
        .groupby("time_bin", observed=False)["patient_id"]
        .nunique()
        .reindex(interval_index, fill_value=0)
    )
    bin_centers_months = np.array([interval.mid for interval in density.index]) / 30.0
    axes[1, 1].plot(bin_centers_months, density.to_numpy(), marker="o", color="#E69F00")
    axes[1, 1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_xlabel("Months from surgery (30-day months)")
    axes[1, 1].set_ylabel("Patients with ≥1 observation")
    axes[1, 1].set_title("Patient observation density by 90-day interval")
    figure.suptitle("Longitudinal observation patterns", y=1.01)
    figure.tight_layout()
    _save_png(figure, output_dir, "observation_patterns", dpi)

    # ==================================================================================
    # 六、比较完整队列与三个DFS-landmark队列的基线构成变化。
    # ==================================================================================
    landmark_labels = [
        f"{days // 30}m\n(n={len(eligible_by_landmark[days]):,})" for days in landmark_days
    ]
    numeric_shift = pd.DataFrame(index=numeric_baseline, columns=landmark_labels, dtype=float)
    for variable in numeric_baseline:
        overall = patients[variable]
        overall_iqr = overall.quantile(0.75) - overall.quantile(0.25)
        for landmark, label in zip(landmark_days, landmark_labels, strict=True):
            cohort_median = eligible_by_landmark[landmark][variable].median()
            numeric_shift.loc[variable, label] = (
                (cohort_median - overall.median()) / overall_iqr
                if pd.notna(overall_iqr) and overall_iqr != 0
                else np.nan
            )

    categorical_shift_rows: list[pd.Series] = []
    categorical_shift_names: list[str] = []
    for variable in categorical_baseline:
        overall_values = _category(patients[variable])
        for level in sorted(overall_values.unique().tolist()):
            shifts: dict[str, float] = {}
            overall_percent = _percent((overall_values == level).sum(), len(overall_values))
            for landmark, label in zip(landmark_days, landmark_labels, strict=True):
                cohort_values = _category(eligible_by_landmark[landmark][variable])
                shifts[label] = (
                    _percent((cohort_values == level).sum(), len(cohort_values)) - overall_percent
                )
            categorical_shift_rows.append(pd.Series(shifts))
            categorical_shift_names.append(f"{variable} = {level}")
    categorical_shift = pd.DataFrame(categorical_shift_rows, index=categorical_shift_names)

    figure_height = max(11.0, 5.0 + 0.32 * len(categorical_shift))
    figure = plt.figure(figsize=(11, figure_height))
    grid = figure.add_gridspec(
        2,
        1,
        height_ratios=[max(2, len(numeric_shift)), max(4, len(categorical_shift))],
        hspace=0.35,
    )
    numeric_axis = figure.add_subplot(grid[0])
    categorical_axis = figure.add_subplot(grid[1])
    sns.heatmap(
        numeric_shift,
        ax=numeric_axis,
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Change in median / full-cohort IQR"},
    )
    numeric_axis.set_title("Numeric baseline shift relative to the full cohort")
    numeric_axis.set_xlabel("")
    numeric_axis.set_ylabel("")
    sns.heatmap(
        categorical_shift,
        ax=categorical_axis,
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".1f",
        cbar_kws={"label": "Percentage-point change from full cohort"},
    )
    categorical_axis.set_title("Categorical baseline shift relative to the full cohort")
    categorical_axis.set_xlabel("DFS-landmark eligible cohort")
    categorical_axis.set_ylabel("")
    figure.suptitle("Population composition after landmark eligibility", y=0.995)
    _save_png(figure, output_dir, "landmark_population_shift", dpi)

    # ==================================================================================
    # 七、用一幅全特征矩阵展示三个窗口及逐90天分箱的患者覆盖情况。
    # ==================================================================================
    coverage_matrix = pd.DataFrame(index=ordered_features, dtype=float)
    for landmark, label in zip(landmark_days, landmark_labels, strict=True):
        eligible = eligible_by_landmark[landmark]
        window = window_observations[landmark]
        denominator = len(eligible)
        patient_feature = window.groupby("feature", observed=True)["patient_id"].nunique()
        patient_feature_dates = (
            window.groupby(["patient_id", "feature"], observed=True)["time_days"]
            .nunique()
            .ge(2)
            .groupby("feature", observed=True)
            .sum()
        )
        short_label = label.split("\n")[0]
        coverage_matrix[f"{short_label} observed"] = (
            patient_feature.reindex(ordered_features, fill_value=0) / denominator * 100
        )
        coverage_matrix[f"{short_label} ≥2 dates"] = (
            patient_feature_dates.reindex(ordered_features, fill_value=0) / denominator * 100
        )

    for start in range(0, max_landmark, time_bin_days):
        end = min(start + time_bin_days, max_landmark)
        bin_eligible = patients.loc[patients["dfs_time_days"] > end]
        bin_window = observations.loc[
            observations["patient_id"].isin(set(bin_eligible["patient_id"].dropna()))
            & observations["time_days"].between(start, end, inclusive="left")
        ]
        patient_feature = bin_window.groupby("feature", observed=True)["patient_id"].nunique()
        coverage_matrix[f"{start // 30}–{end // 30}m"] = (
            patient_feature.reindex(ordered_features, fill_value=0) / len(bin_eligible) * 100
        )

    figure, axis = plt.subplots(figsize=(18, max(13, 0.29 * len(coverage_matrix))))
    sns.heatmap(
        coverage_matrix,
        ax=axis,
        cmap="viridis",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Eligible patients observed (%)"},
        yticklabels=True,
    )
    _add_group_separators(axis, ordered_groups)
    axis.set_title(
        "Feature availability across nested landmarks and 90-day intervals\n"
        "Each interval denominator is patients remaining DFS-event-free through its end"
    )
    axis.set_xlabel("Observation window")
    axis.set_ylabel("Feature (ordered by feature group)")
    axis.tick_params(axis="x", rotation=45)
    axis.tick_params(axis="y", rotation=0)
    _save_png(figure, output_dir, "feature_coverage_matrix", dpi)

    # ==================================================================================
    # 八、展示全部指标的观测量、零值/负值比例及无单位依赖的尾部分布形态。
    # ==================================================================================
    feature_grouped = observations.groupby("feature", observed=True)["value"]
    value_patterns = feature_grouped.agg(
        n_observations="size", minimum="min", median="median", maximum="max"
    ).reindex(ordered_features)
    quantiles = (
        feature_grouped.quantile([0.01, 0.25, 0.75, 0.99]).unstack().reindex(ordered_features)
    )
    value_patterns["zero_percent"] = (
        observations["value"]
        .eq(0)
        .groupby(observations["feature"], observed=True)
        .mean()
        .reindex(ordered_features)
        .mul(100)
    )
    value_patterns["negative_percent"] = (
        observations["value"]
        .lt(0)
        .groupby(observations["feature"], observed=True)
        .mean()
        .reindex(ordered_features)
        .mul(100)
    )
    iqr = quantiles[0.75] - quantiles[0.25]
    value_patterns["central_tail_span"] = ((quantiles[0.99] - quantiles[0.01]) / iqr).where(iqr > 0)
    upper_gap = (value_patterns["maximum"] - quantiles[0.99]) / iqr
    lower_gap = (quantiles[0.01] - value_patterns["minimum"]) / iqr
    value_patterns["outer_extreme_gap"] = (
        pd.concat([upper_gap, lower_gap], axis=1).max(axis=1).where(iqr > 0)
    )
    value_patterns = value_patterns.replace([np.inf, -np.inf], np.nan)

    y_positions = np.arange(len(value_patterns))
    figure, axes = plt.subplots(
        1, 4, figsize=(19, max(13, 0.29 * len(value_patterns))), sharey=True
    )
    axes[0].scatter(value_patterns["n_observations"], y_positions, s=15, color="#0072B2")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Observation records (log scale)")
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(ordered_features, fontsize=7)
    axes[0].invert_yaxis()
    axes[1].scatter(
        value_patterns["zero_percent"], y_positions, s=14, label="Zero", color="#009E73"
    )
    axes[1].scatter(
        value_patterns["negative_percent"],
        y_positions,
        s=14,
        label="Negative",
        color="#D55E00",
    )
    axes[1].set_xlabel("Observation values (%)")
    axes[1].legend(loc="lower right")
    axes[2].scatter(
        np.log1p(value_patterns["central_tail_span"].clip(lower=0)),
        y_positions,
        s=15,
        color="#CC79A7",
    )
    axes[2].set_xlabel("log1p((Q99−Q01) / IQR)")
    axes[3].scatter(
        np.log1p(value_patterns["outer_extreme_gap"].clip(lower=0)),
        y_positions,
        s=15,
        color="#E69F00",
    )
    axes[3].set_xlabel("log1p(max outer gap / IQR)")
    for axis in axes:
        axis.set_ylim(len(value_patterns) - 0.5, -0.5)
        axis.grid(axis="y", visible=False)
    _add_group_separators(axes, ordered_groups)
    figure.suptitle(
        "Feature value patterns without value removal or winsorization\n"
        "Scale-free tail ratios support visual review when measurement units are unavailable",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    _save_png(figure, output_dir, "feature_value_patterns", dpi)

    # ==================================================================================
    # 九、展示手术年代的队列规模、随访、结局比例及全指标覆盖变化。
    # ==================================================================================
    years = sorted(int(year) for year in patients["surgery_year"].dropna().unique())
    year_index = pd.Index(years, name="surgery_year")
    year_counts = patients.groupby("surgery_year", observed=True).size().reindex(year_index)
    year_followup = (
        patients.groupby("surgery_year", observed=True)["calendar_followup_days"]
        .median()
        .reindex(year_index)
        / 365.25
    )
    year_dfs_event = (
        patients.groupby("surgery_year", observed=True)["dfs_event"]
        .mean()
        .reindex(year_index)
        .mul(100)
    )
    year_os_event = (
        patients.groupby("surgery_year", observed=True)["os_event"]
        .mean()
        .reindex(year_index)
        .mul(100)
    )

    observation_year = observations.loc[
        observations["time_days"].between(0, max_landmark), ["patient_id", "feature"]
    ].merge(
        patients[["patient_id", "surgery_year"]],
        on="patient_id",
        how="inner",
        validate="many_to_one",
    )
    feature_year_counts = (
        observation_year.groupby(["feature", "surgery_year"], observed=True)["patient_id"]
        .nunique()
        .unstack(fill_value=0)
    )
    feature_year_coverage = (
        feature_year_counts.reindex(index=ordered_features, columns=years, fill_value=0)
        .div(year_counts, axis=1)
        .mul(100)
    )

    figure = plt.figure(figsize=(19, max(15, 7 + 0.27 * len(ordered_features))))
    grid = figure.add_gridspec(2, 3, height_ratios=[1, 4], hspace=0.32, wspace=0.28)
    count_axis = figure.add_subplot(grid[0, 0])
    followup_axis = figure.add_subplot(grid[0, 1])
    event_axis = figure.add_subplot(grid[0, 2])
    coverage_axis = figure.add_subplot(grid[1, :])
    count_axis.bar(years, year_counts, color="#0072B2")
    count_axis.set_title("Patients by surgery year")
    count_axis.set_ylabel("Patients")
    followup_axis.plot(years, year_followup, marker="o", color="#009E73")
    followup_axis.set_title("Median calendar follow-up")
    followup_axis.set_ylabel("Years")
    event_axis.plot(years, year_dfs_event, marker="o", label="DFS event", color="#D55E00")
    event_axis.plot(years, year_os_event, marker="o", label="Death", color="#CC79A7")
    event_axis.set_title("Crude recorded event proportions")
    event_axis.set_ylabel("Patients (%)")
    event_axis.legend()
    for axis in (count_axis, followup_axis, event_axis):
        axis.tick_params(axis="x", rotation=45)
    sns.heatmap(
        feature_year_coverage,
        ax=coverage_axis,
        cmap="viridis",
        vmin=0,
        vmax=100,
        cbar_kws={"label": f"Patients observed within 0–{max_landmark} days (%)"},
        yticklabels=True,
    )
    _add_group_separators(coverage_axis, ordered_groups)
    coverage_axis.set_title("Feature coverage by surgery year")
    coverage_axis.set_xlabel("Surgery year")
    coverage_axis.set_ylabel("Feature (ordered by feature group)")
    coverage_axis.tick_params(axis="y", rotation=0)
    figure.suptitle("Calendar-time patterns", y=0.995)
    _save_png(figure, output_dir, "calendar_time_patterns", dpi)

    # ==================================================================================
    # 十、枚举全部可用手术年份作为时间切分点，仅展示样本、事件和随访权衡。
    # ==================================================================================
    cutoff_rows: list[dict[str, Any]] = []
    for landmark in landmark_days:
        eligible = eligible_by_landmark[landmark]
        for cutoff in years[1:]:
            development = eligible.loc[eligible["surgery_year"] < cutoff]
            test = eligible.loc[eligible["surgery_year"] >= cutoff]
            cutoff_rows.append(
                {
                    "landmark_days": landmark,
                    "cutoff_year": cutoff,
                    "development_patients": len(development),
                    "test_patients": len(test),
                    "development_dfs_events": int((development["dfs_event"] == 1).sum()),
                    "test_dfs_events": int((test["dfs_event"] == 1).sum()),
                    "test_remaining_dfs_median_years": (
                        (test["dfs_time_days"] - landmark).median() / 365.25
                    ),
                }
            )
    cutoff_frame = pd.DataFrame(cutoff_rows)
    figure, axes = plt.subplots(
        len(landmark_days),
        3,
        figsize=(18, 4.1 * len(landmark_days)),
        squeeze=False,
        sharex=True,
    )
    for row, landmark in enumerate(landmark_days):
        data = cutoff_frame.loc[cutoff_frame["landmark_days"] == landmark]
        axes[row, 0].plot(
            data["cutoff_year"],
            data["development_patients"],
            marker="o",
            label="Development",
        )
        axes[row, 0].plot(
            data["cutoff_year"], data["test_patients"], marker="o", label="Temporal test"
        )
        axes[row, 0].set_ylabel(f"{landmark // 30}m landmark\nPatients")
        axes[row, 1].plot(
            data["cutoff_year"],
            data["development_dfs_events"],
            marker="o",
            label="Development",
        )
        axes[row, 1].plot(
            data["cutoff_year"],
            data["test_dfs_events"],
            marker="o",
            label="Temporal test",
        )
        axes[row, 1].set_ylabel("Subsequent DFS events")
        axes[row, 2].plot(
            data["cutoff_year"],
            data["test_remaining_dfs_median_years"],
            marker="o",
            color="#009E73",
        )
        axes[row, 2].set_ylabel("Test median remaining DFS (years)")
        for axis in axes[row]:
            axis.grid(alpha=0.3)
            axis.tick_params(axis="x", rotation=45)
    axes[0, 0].set_title("Eligible sample allocation")
    axes[0, 1].set_title("Subsequent event allocation")
    axes[0, 2].set_title("Temporal-test follow-up")
    axes[0, 0].legend()
    axes[0, 1].legend()
    for axis in axes[-1]:
        axis.set_xlabel("First surgery year assigned to temporal test")
    figure.suptitle(
        "Temporal cutoff trade-offs across all observed surgery years\n"
        "The figure is descriptive and does not select or reject a cutoff",
        y=1.01,
    )
    figure.tight_layout()
    _save_png(figure, output_dir, "temporal_cutoff_tradeoff", dpi)

    print(
        f"CRC云南EDA完成：{len(patients):,}名患者，{len(observations):,}条观测，"
        f"结果保存至 {output_dir}"
    )


if __name__ == "__main__":
    main()

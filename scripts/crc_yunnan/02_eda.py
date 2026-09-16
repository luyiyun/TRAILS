from __future__ import annotations

# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false
import json
from pathlib import Path
from typing import Any

import hydra
import matplotlib
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from scripts.crc_yunnan.config import CRCEDAConfig

matplotlib.use("Agg")
import seaborn as sns  # noqa: E402
from lifelines import KaplanMeierFitter  # noqa: E402
from lifelines.plotting import add_at_risk_counts  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib import pyplot as plt
from matplotlib.figure import Figure  # noqa: E402

BASELINE_LABELS = {
    "Preoperative_CEA": "术前CEA",
    "Preoperative_CA242": "术前CA242",
    "Age": "年龄",
    "Sex": "性别",
    "Primary_site": "原发部位",
    "Surgical_approach": "手术方式",
    "Tumor_differentiation": "肿瘤分化",
    "AJCC_8th_ed_Stage": "AJCC第8版分期",
    "Lymph_node_yield": "淋巴结检出数",
    "Mucinous_colloid_type": "黏液胶样类型",
    "Lymphovascular_invasion": "淋巴血管侵犯",
    "Perineural_invasion": "神经侵犯",
    "Adjuvant_chemotherapy": "辅助化疗",
}


def _percent(numerator: int | float, denominator: int | float) -> float:
    return 100 * numerator / denominator if denominator else np.nan


def _category(values: pd.Series) -> pd.Series:
    return values.fillna("缺失").astype("string").str.strip().replace("", "缺失")


def _distribution(values: pd.Series) -> dict[str, int | float]:
    """同类数值的描述统计用于JSON，不把不同含义的指标拼成通用表。"""
    values = values.dropna()
    return {
        "有效数": int(values.size),
        "最小值": values.min(),
        "下四分位数": values.quantile(0.25),
        "中位数": values.median(),
        "上四分位数": values.quantile(0.75),
        "最大值": values.max(),
        "负值数": int(values.lt(0).sum()),
        "零值数": int(values.eq(0).sum()),
        "正值数": int(values.gt(0).sum()),
    }


def _patient_counts(window: pd.DataFrame, ids: pd.Series, features: list[str]) -> pd.DataFrame:
    """将非缺失单元格计为观测，无观测的合格患者仍保留在分母中。"""
    observed = window[features].notna()
    return (
        pd.DataFrame(
            {
                "有效观测数": observed.sum(axis=1).groupby(window["patient_id"]).sum(),
                "观测日期数": window.groupby("patient_id")["time_days"].nunique(),
                "观测特征数": observed.groupby(window["patient_id"]).any().sum(axis=1),
            }
        )
        .reindex(ids)
        .fillna(0)
    )


def _coverage(window: pd.DataFrame, features: list[str], denominator: int) -> pd.DataFrame:
    observed_dates = window[features].notna().groupby(window["patient_id"]).sum()
    counts = pd.DataFrame(
        {
            "至少一次观测人数": observed_dates.ge(1).sum(),
            "至少两次观测人数": observed_dates.ge(2).sum(),
        }
    ).reindex(features, fill_value=0)
    counts["合格患者数"] = denominator
    counts["至少一次覆盖率（%）"] = counts["至少一次观测人数"] / (denominator or np.nan) * 100
    counts["至少两次覆盖率（%）"] = counts["至少两次观测人数"] / (denominator or np.nan) * 100
    return counts.rename_axis("特征")


def _save_png(figure: Figure, output: Path, name: str, dpi: int) -> None:
    figure.savefig(output / f"{name}.png", dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _plot_survival_curves(patients: pd.DataFrame, output: Path, dpi: int) -> None:
    """绘制基础队列的DFS和OS生存曲线及风险人数。"""
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    calendar_max = float(patients["calendar_followup_days"].clip(lower=0).max())
    if pd.isna(calendar_max) or calendar_max <= 0:
        calendar_max = max(
            patients["dfs_time_days"].quantile(0.99), patients["os_time_days"].quantile(0.99)
        )
    for axis, outcome, title in zip(
        axes, ("dfs", "os"), ("无病生存（DFS）", "总生存（OS）"), strict=True
    ):
        valid = patients[f"{outcome}_time_days"].ge(0) & patients[f"{outcome}_event"].isin([0, 1])
        if valid.any():
            km = KaplanMeierFitter(label=outcome.upper()).fit(
                patients.loc[valid, f"{outcome}_time_days"] / 365.25,
                event_observed=patients.loc[valid, f"{outcome}_event"],
            )
            km.plot_survival_function(ax=axis, ci_show=True)
            axis.set_xlim(0, max(float(calendar_max) / 365.25, 0.01))
            add_at_risk_counts(km, ax=axis, rows_to_show=["At risk"], ypos=-0.45)
            risk_axis = figure.axes[-1]
            ticks = risk_axis.get_xticks()
            risk_axis.set_xticks(
                ticks,
                [
                    label.get_text().replace("At risk", "风险人数")
                    for label in risk_axis.get_xticklabels()
                ],
            )
        axis.set(title=title, xlabel="手术后时间（年）", ylabel="生存概率", ylim=(0, 1.03))
    figure.suptitle("云南结直肠癌基础队列生存曲线（阴影为95%置信区间）")
    figure.subplots_adjust(bottom=0.27, wspace=0.25)
    _save_png(figure, output, "survival_curves", dpi)


def _plot_observation_patterns(
    counts: pd.DataFrame,
    density: pd.Series,
    max_landmark: int,
    bin_days: int,
    output: Path,
    dpi: int,
) -> None:
    """绘制患者观测分布与相对手术时间的观测密度。"""
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    for axis, column in zip(axes.flat, counts.columns, strict=False):
        sns.ecdfplot(counts[column], ax=axis)
        if counts[column].max() > 0:
            axis.set_xscale("symlog", linthresh=1)
        axis.set(
            xlabel=f"0–{max_landmark}天{column}", ylabel="累计比例", title=f"每位患者的{column}"
        )
    axes[1, 1].plot(
        [(interval.left + interval.right) / 60 for interval in density.index],
        density.to_numpy(),
        marker="o",
    )
    axes[1, 1].axvline(0, color="black", linestyle="--")
    axes[1, 1].set(
        xlabel="相对手术时间（月，每月30天）",
        ylabel="有观测患者数",
        title=f"每{bin_days}天的观测人数",
    )
    figure.tight_layout()
    _save_png(figure, output, "observation_patterns", dpi)


def _plot_landmark_population_shift(
    numeric_shift: pd.DataFrame,
    categorical_shift: pd.DataFrame,
    outcome_label: str,
    output: Path,
    dpi: int,
) -> None:
    """用已计算的基线差异展示landmark人群变化。"""
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11, max(10, 5 + 0.32 * len(categorical_shift))),
        gridspec_kw={"height_ratios": [max(5, len(numeric_shift)), max(4, len(categorical_shift))]},
    )
    for axis, frame, title, unit in (
        (axes[0], numeric_shift, "连续基线相对完整队列的变化", "中位数变化 / 完整队列IQR"),
        (axes[1], categorical_shift, "分类基线相对完整队列的变化", "比例变化（百分点）"),
    ):
        if not frame.empty:
            limit = float(np.nanmax(np.abs(frame.to_numpy(dtype=float)), initial=0)) or 1.0
            sns.heatmap(
                frame,
                ax=axis,
                cmap="vlag",
                center=0,
                vmin=-limit,
                vmax=limit,
                annot=True,
                fmt=".2f",
                cbar_kws={"label": unit},
            )
        axis.set(title=title, xlabel=f"{outcome_label}-landmark合格队列", ylabel="")
    figure.tight_layout()
    _save_png(figure, output, "landmark_population_shift", dpi)


def _plot_feature_coverage_matrix(
    coverage_matrix: pd.DataFrame, bin_days: int, output: Path, dpi: int
) -> None:
    """绘制各landmark及时间箱的特征覆盖热图。"""
    figure, axis = plt.subplots(figsize=(18, max(5, 0.29 * len(coverage_matrix))))
    sns.heatmap(
        coverage_matrix,
        ax=axis,
        cmap="viridis",
        vmin=0,
        vmax=100,
        yticklabels=True,
        cbar_kws={"label": "合格患者覆盖率（%）"},
    )
    axis.set(
        title=f"各landmark及{bin_days}天时间箱的特征覆盖\n分母与具体人数见Excel覆盖表",
        xlabel="观测窗口",
        ylabel="特征（按特征组排列）",
    )
    axis.tick_params(axis="x", rotation=45)
    axis.tick_params(axis="y", rotation=0)
    _save_png(figure, output, "feature_coverage_matrix", dpi)


def _plot_feature_value_patterns(value_patterns: pd.DataFrame, output: Path, dpi: int) -> None:
    """绘制特征观测数、零负值比例和尾部分布。"""
    features: list[str] = value_patterns.index.tolist()
    figure, axes = plt.subplots(1, 4, figsize=(19, max(5, 0.29 * len(features))), sharey=True)
    y = np.arange(len(features))
    axes[0].scatter(value_patterns["有效观测数"], y)
    if value_patterns["有效观测数"].gt(0).to_numpy().any():
        axes[0].set_xscale("log")
    axes[0].set_xlabel("有效观测数（对数刻度）")
    axes[0].set_yticks(y, features)
    for column in ("零值比例（%）", "负值比例（%）"):
        axes[1].scatter(value_patterns[column], y, label=column)
    axes[1].legend()
    axes[1].set_xlabel("实际观测值的比例（%）")
    for axis, column in zip(axes[2:], ("中部尾跨度比", "外侧极端间距比"), strict=True):
        axis.scatter(np.log1p(value_patterns[column].clip(lower=0)), y)
        axis.set_xlabel(f"log1p（{column}）")
    for axis in axes:
        axis.set_ylim(len(features) - 0.5, -0.5)
    figure.suptitle(
        "特征数值分布：保留原值，不删除或截尾\n"
        "中部尾跨度比=(Q99−Q01)/IQR；外侧极端间距比=max(最大值−Q99, Q01−最小值)/IQR"
    )
    figure.tight_layout()
    _save_png(figure, output, "feature_value_patterns", dpi)


def _plot_calendar_time_patterns(
    year_summary: pd.DataFrame,
    year_coverage: pd.DataFrame,
    max_landmark: int,
    output: Path,
    dpi: int,
) -> None:
    """绘制年度队列概况及按手术年份的特征覆盖。"""
    figure = plt.figure(figsize=(18, max(9, 5 + 0.27 * len(year_coverage))))
    grid = figure.add_gridspec(2, 3, height_ratios=[1, 3], hspace=0.4)
    for index, columns, title in (
        (0, ["患者数"], "手术年份与人数"),
        (1, ["日历随访中位数（年）"], "日历随访"),
        (2, ["DFS事件比例（%）", "OS事件比例（%）"], "粗事件比例"),
    ):
        axis = figure.add_subplot(grid[0, index])
        for column in columns:
            axis.plot(year_summary.index, year_summary[column], marker="o", label=column)
        axis.set(title=title, xlabel="手术年份")
        axis.legend()
    axis = figure.add_subplot(grid[1, :])
    sns.heatmap(
        year_coverage,
        ax=axis,
        vmin=0,
        vmax=100,
        cmap="viridis",
        yticklabels=True,
        cbar_kws={"label": f"0–{max_landmark}天有观测患者比例（%）"},
    )
    axis.set(xlabel="手术年份", ylabel="特征", title="按手术年份的特征覆盖（分母为当年全部患者）")
    axis.tick_params(axis="y", rotation=0)
    _save_png(figure, output, "calendar_time_patterns", dpi)


def _plot_temporal_cutoff_tradeoff(
    cutoff_frame: pd.DataFrame,
    landmark_days: list[int],
    outcome_label: str,
    output: Path,
    dpi: int,
) -> None:
    """展示各landmark下时间切点的样本、事件和随访权衡。"""
    figure, axes = plt.subplots(
        len(landmark_days), 3, figsize=(18, 4 * len(landmark_days)), squeeze=False
    )
    for row, landmark in enumerate(landmark_days):
        data = cutoff_frame.loc[cutoff_frame["landmark（天）"] == landmark]
        for axis, columns in zip(
            axes[row],
            (
                ["开发集人数", "测试集人数"],
                [f"开发集{outcome_label}事件数", f"测试集{outcome_label}事件数"],
                [f"测试集剩余{outcome_label}中位数（年）"],
            ),
            strict=True,
        ):
            for column in columns:
                axis.plot(data["时间测试集起始年份"], data[column], marker="o", label=column)
            axis.set(xlabel="时间测试集起始手术年份", title=f"{landmark / 30:g}个月landmark")
            axis.legend()
    figure.suptitle("时间切点的样本、事件和随访权衡：仅作描述，不自动选择切点")
    figure.tight_layout()
    _save_png(figure, output, "temporal_cutoff_tradeoff", dpi)


def run(config: CRCEDAConfig) -> None:
    ##################################################
    # 一、读取01基础队列及中格式矩阵；确定中文展示和统计分母。
    ##################################################
    output = config.paths.output_dir
    if config.quality.refuse_overwrite and output.exists():
        raise FileExistsError(f"拒绝覆盖既有EDA目录：{output}")
    # 直接消费01的标准产物；读取时恢复类型，不重复校验其结构和键约束。
    patients = pd.read_csv(config.paths.patients_csv, dtype={"patient_id": "string"})
    metadata = pd.read_csv(config.paths.feature_metadata_csv, dtype="string").sort_values(
        ["feature_group", "feature"]
    )
    features = metadata["feature"].tolist()
    observations = pd.read_csv(
        config.paths.observations_csv,
        dtype={"patient_id": "string", **dict.fromkeys(["time_days", *features], "float64")},
    )
    patients[["surgery_date", "last_followup_date"]] = patients[
        ["surgery_date", "last_followup_date"]
    ].apply(pd.to_datetime)
    patients["calendar_followup_days"] = (
        patients["last_followup_date"] - patients["surgery_date"]
    ).dt.days
    patients["surgery_year"] = patients["surgery_date"].dt.year.astype("Int64")
    matrix = observations[features]
    outcome_label = config.outcome.upper()
    time_column, event_column = f"{config.outcome}_time_days", f"{config.outcome}_event"
    landmark_days = config.trajectory.landmark_days
    bin_days, annual_days = (
        config.trajectory.time_bin_days,
        config.trajectory.annual_segment_days,
    )
    max_landmark = max(landmark_days)
    numeric = config.columns.numeric_baseline
    categorical = config.columns.categorical_baseline
    patients[numeric] = patients[numeric].astype(float)
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in config.plot.font_families if name in available_fonts), None)
    if font is None:
        raise ValueError("未找到配置中的中文绘图字体，请安装中文字体或设置plot.font_families")
    sns.set_theme(style="whitegrid", context="notebook", font=font)
    plt.rcParams.update({"font.family": [font, "DejaVu Sans"], "axes.unicode_minus": False})
    dpi = config.plot.dpi
    output.mkdir(parents=True, exist_ok=not config.quality.refuse_overwrite)
    tables: dict[str, pd.DataFrame] = {}
    summary: dict[str, Any] = {
        "统计口径": {
            "人群": "01完成基础纳排后的队列",
            "有效观测数": "中格式特征列中的非缺失单元格数",
            "患者时间点数": "中格式观测表的行数",
            "landmark": f"{outcome_label}时间严格大于landmark；观测窗口包括0和landmark两端",
            "分箱覆盖率": f"分母为{outcome_label}时间严格大于分箱右端的患者；窗口左闭右开",
            "分类编码": "保留源数据类别编码，不推定未提供的编码含义",
        },
        "队列": {
            "患者数": len(patients),
            "有效观测数": int(matrix.count().sum()),
            "患者时间点数": len(observations),
            "特征数": len(features),
            "特征组数": int(metadata["feature_group"].nunique()),
            "有观测患者数": int(observations["patient_id"].nunique()),
            "无观测患者数": int((~patients["patient_id"].isin(observations["patient_id"])).sum()),
            "最早手术日期": patients["surgery_date"].min(),
            "最晚手术日期": patients["surgery_date"].max(),
            "最早末次随访日期": patients["last_followup_date"].min(),
            "最晚末次随访日期": patients["last_followup_date"].max(),
            "观测相对手术最早天数": observations["time_days"].min(),
            "观测相对手术最晚天数": observations["time_days"].max(),
        },
        "数据质量": {
            "患者表分母": len(patients),
            "观测时间点分母": len(observations),
            "重复患者ID数": int(patients["patient_id"].duplicated().sum()),
            "重复患者时间点数": int(observations.duplicated(["patient_id", "time_days"]).sum()),
            "手术日期缺失人数": int(patients["surgery_date"].isna().sum()),
            "末次随访日期缺失人数": int(patients["last_followup_date"].isna().sum()),
            "随访早于手术人数": int(patients["calendar_followup_days"].lt(0).sum()),
            "DFS时间超过OS人数": int(patients["dfs_time_days"].gt(patients["os_time_days"]).sum()),
        },
        "结局与随访": {},
    }
    for outcome, label in (("dfs", "无病生存DFS"), ("os", "总生存OS")):
        times, events = patients[f"{outcome}_time_days"], patients[f"{outcome}_event"]
        summary["结局与随访"][label] = {
            "时间（天）": _distribution(times),
            "事件数": int(events.eq(1).sum()),
            "事件有效人数": int(events.notna().sum()),
            "事件比例（%）": _percent(events.eq(1).sum(), events.notna().sum()),
        }
        summary["数据质量"][f"{label}时间缺失人数"] = int(times.isna().sum())
        summary["数据质量"][f"{label}时间为负人数"] = int(times.lt(0).sum())
        summary["数据质量"][f"{label}事件编码无效人数"] = int((~events.isin([0, 1])).sum())
    summary["结局与随访"]["日历随访（天）"] = _distribution(patients["calendar_followup_days"])
    summary["结局与随访"]["DFS减OS（天）"] = _distribution(
        patients["dfs_time_days"] - patients["os_time_days"]
    )
    for code, label in ((0, "删失"), (1, "死亡")):
        delta = patients["os_time_days"] - patients["calendar_followup_days"]
        summary["结局与随访"][f"{label}患者OS减日历随访（天）"] = _distribution(
            delta.loc[patients["os_event"].eq(code)]
        )

    ##################################################
    # 二、基线、特征及landmark按各自自然维度形成表格。
    ##################################################
    baseline_rows: list[dict[str, Any]] = []
    for variable in numeric:
        values = patients[variable]
        baseline_rows.append(
            {
                "变量": BASELINE_LABELS.get(variable, variable),
                "类型": "连续",
                "类别": "",
                "描述": (
                    f"{values.median():.3g} [{values.quantile(0.25):.3g}, "
                    f"{values.quantile(0.75):.3g}]; {values.mean():.3g} ({values.std():.3g}); "
                    f"{values.min():.3g}–{values.max():.3g}"
                )
                if values.notna().to_numpy().any()
                else "缺失",
                "有效人数": int(values.count()),
                "缺失人数": int(values.isna().sum()),
                "缺失率（%）": _percent(values.isna().sum(), len(values)),
            }
        )
    for variable in categorical:
        values = _category(patients[variable])
        missing = int(values.eq("缺失").sum())
        for level, count in values.value_counts().items():
            baseline_rows.append(
                {
                    "变量": BASELINE_LABELS.get(variable, variable),
                    "类型": "分类",
                    "类别": level,
                    "描述": f"{count} ({_percent(count, len(values)):.1f}%)",
                    "有效人数": len(values) - missing,
                    "缺失人数": missing,
                    "缺失率（%）": _percent(missing, len(values)),
                }
            )
    tables["基线特征"] = pd.DataFrame(baseline_rows)
    summary["统计口径"]["连续基线描述"] = (
        "中位数[下四分位数,上四分位数]；均值(标准差)；最小值–最大值"
    )
    quantiles = matrix.quantile([0.01, 0.25, 0.75, 0.99])
    iqr = quantiles.loc[0.75] - quantiles.loc[0.25]
    value_patterns = (
        pd.DataFrame(
            {
                "有效观测数": matrix.count(),
                "时间点分母": len(matrix),
                "缺失时间点数": matrix.isna().sum(),
                "缺失比例（%）": matrix.isna().mean() * 100,
                "最小值": matrix.min(),
                "中位数": matrix.median(),
                "最大值": matrix.max(),
                "下四分位数": quantiles.loc[0.25],
                "上四分位数": quantiles.loc[0.75],
                "第1百分位数": quantiles.loc[0.01],
                "第99百分位数": quantiles.loc[0.99],
                "零值比例（%）": matrix.eq(0).sum() / matrix.count().replace(0, np.nan) * 100,
                "负值比例（%）": matrix.lt(0).sum() / matrix.count().replace(0, np.nan) * 100,
                "中部尾跨度比": ((quantiles.loc[0.99] - quantiles.loc[0.01]) / iqr).where(iqr > 0),
                "外侧极端间距比": pd.concat(
                    [
                        (matrix.max() - quantiles.loc[0.99]) / iqr,
                        (quantiles.loc[0.01] - matrix.min()) / iqr,
                    ],
                    axis=1,
                )
                .max(axis=1)
                .where(iqr > 0),
            }
        )
        .reindex(features)
        .rename_axis("特征")
    )
    value_patterns.insert(
        0, "特征组", metadata.set_index("feature")["feature_group"].reindex(features)
    )
    tables["特征分布"] = value_patterns.reset_index()
    eligible_by_landmark: dict[int, pd.DataFrame] = {}
    landmark_rows: list[dict[str, Any]] = []
    coverage_rows: list[pd.DataFrame] = []
    coverage_matrix = pd.DataFrame(index=features)
    for landmark in landmark_days:
        eligible = patients.loc[patients[time_column] > landmark]
        eligible_by_landmark[landmark] = eligible
        window = observations.loc[
            observations["patient_id"].isin(eligible["patient_id"])
            & observations["time_days"].between(0, landmark)
        ]
        counts = _patient_counts(window, eligible["patient_id"], features)
        bins = (
            (window["time_days"] // bin_days)
            .groupby(window["patient_id"])
            .nunique()
            .reindex(eligible["patient_id"])
            .fillna(0)
        )
        segments = pd.DataFrame(
            {
                start: window.loc[
                    window["time_days"].between(
                        start, min(start + annual_days, landmark), inclusive="left"
                    )
                ]
                .groupby("patient_id")
                .size()
                .reindex(eligible["patient_id"], fill_value=0)
                .gt(0)
                for start in range(0, landmark, annual_days)
            }
        )
        remaining = eligible[time_column] - landmark
        landmark_rows.append(
            {
                "landmark（天）": landmark,
                "landmark（月，每月30天）": landmark / 30,
                "完整队列人数": len(patients),
                f"{outcome_label}合格比例（%）": _percent(len(eligible), len(patients)),
                f"后续{outcome_label}事件比例（%）": _percent(
                    eligible[event_column].eq(1).sum(), len(eligible)
                ),
                f"剩余{outcome_label}下四分位数（天）": remaining.quantile(0.25),
                f"剩余{outcome_label}中位数（天）": remaining.median(),
                f"剩余{outcome_label}上四分位数（天）": remaining.quantile(0.75),
                **{
                    f"{outcome.upper()}合格人数": int(
                        patients[f"{outcome}_time_days"].gt(landmark).sum()
                    )
                    for outcome in ("dfs", "os")
                },
                **{
                    f"后续{outcome.upper()}事件数": int(
                        (
                            patients[f"{outcome}_time_days"].gt(landmark)
                            & patients[f"{outcome}_event"].eq(1)
                        ).sum()
                    )
                    for outcome in ("dfs", "os")
                },
                "有观测人数": int(counts["有效观测数"].gt(0).sum()),
                "有观测比例（%）": _percent(counts["有效观测数"].gt(0).sum(), len(eligible)),
                "至少两个日期人数": int(counts["观测日期数"].ge(2).sum()),
                "至少两个日期比例（%）": _percent(counts["观测日期数"].ge(2).sum(), len(eligible)),
                "至少两个时间箱人数": int(bins.ge(2).sum()),
                "至少两个时间箱比例（%）": _percent(bins.ge(2).sum(), len(eligible)),
                "每个年度段均有观测人数": int(segments.all(axis=1).sum()),
                "每个年度段均有观测比例（%）": _percent(segments.all(axis=1).sum(), len(eligible)),
                **{f"{column}中位数（全体合格患者）": counts[column].median() for column in counts},
            }
        )
        coverage = _coverage(window, features, len(eligible))
        coverage_rows.append(coverage.reset_index().assign(**{"landmark（天）": landmark}))
        for key, label in (
            ("至少一次覆盖率（%）", "有观测"),
            ("至少两次覆盖率（%）", "至少两日期"),
        ):
            coverage_matrix[f"{landmark / 30:g}月 {label}"] = coverage[key]
    tables["landmark概况"] = pd.DataFrame(landmark_rows)
    tables["窗口特征覆盖"] = pd.concat(coverage_rows, ignore_index=True)
    bin_rows: list[pd.DataFrame] = []
    for start in range(0, max_landmark, bin_days):
        end = min(start + bin_days, max_landmark)
        eligible = patients.loc[patients[time_column] > end]
        window = observations.loc[
            observations["patient_id"].isin(eligible["patient_id"])
            & observations["time_days"].between(start, end, inclusive="left")
        ]
        coverage = _coverage(window, features, len(eligible))
        bin_rows.append(
            coverage.reset_index().assign(**{"分箱起点（天）": start, "分箱终点（天）": end})
        )
        coverage_matrix[f"{start / 30:g}–{end / 30:g}月"] = coverage["至少一次覆盖率（%）"]
    tables["分箱特征覆盖"] = pd.concat(bin_rows, ignore_index=True)

    ##################################################
    # 三、统计患者观测模式和相对手术时间的观测密度。
    ##################################################

    post = observations.loc[observations["time_days"].between(0, max_landmark)]
    counts = _patient_counts(post, patients["patient_id"], features)
    summary["观测模式"] = {column: _distribution(counts[column]) for column in counts}
    density_edges = np.arange(-360, max_landmark + bin_days, bin_days)
    density = (
        observations.loc[observations["time_days"].between(-360, max_landmark)]
        .assign(时间箱=lambda frame: pd.cut(frame["time_days"], bins=density_edges, right=False))
        .groupby("时间箱", observed=False)["patient_id"]
        .nunique()
    )
    tables["观测时间密度"] = pd.DataFrame(
        {
            "分箱起点（天）": [interval.left for interval in density.index],
            "分箱终点（天）": [interval.right for interval in density.index],
            "有观测患者数": density.to_numpy(),
        }
    )

    ##################################################
    # 四、比较各landmark合格人群与完整队列的基线特征。
    ##################################################
    numeric_shift = pd.DataFrame(index=[BASELINE_LABELS.get(v, v) for v in numeric], dtype=float)
    categorical_shift = pd.DataFrame(dtype=float)
    for landmark, eligible in eligible_by_landmark.items():
        label = f"{landmark / 30:g}个月"
        for variable in numeric:
            full = patients[variable]
            scale = full.quantile(0.75) - full.quantile(0.25)
            numeric_shift.loc[BASELINE_LABELS.get(variable, variable), label] = (
                (eligible[variable].median() - full.median()) / scale if scale > 0 else np.nan
            )
        for variable in categorical:
            full_values, eligible_values = (
                _category(patients[variable]),
                _category(eligible[variable]),
            )
            for level in sorted(full_values.unique()):
                name = f"{BASELINE_LABELS.get(variable, variable)} = {level}"
                categorical_shift.loc[name, label] = _percent(
                    eligible_values.eq(level).sum(), len(eligible)
                ) - _percent(full_values.eq(level).sum(), len(patients))
    tables["连续基线变化"] = numeric_shift.rename_axis("变量").reset_index()
    tables["分类基线变化"] = categorical_shift.rename_axis("变量及类别").reset_index()

    ##################################################
    # 五、汇总年度变化和时间切点权衡，仅作描述性比较。
    ##################################################
    year_summary = (
        patients.groupby("surgery_year")
        .agg(
            患者数=("patient_id", "size"),
            日历随访中位数天=("calendar_followup_days", "median"),
            DFS事件比例=("dfs_event", "mean"),
            OS事件比例=("os_event", "mean"),
        )
        .rename_axis("手术年份")
    )
    year_summary["日历随访中位数（年）"] = year_summary.pop("日历随访中位数天") / 365.25
    for outcome in ("DFS", "OS"):
        year_summary[f"{outcome}事件比例（%）"] = year_summary.pop(f"{outcome}事件比例") * 100
    yearly_observed = post[features].notna().groupby(post["patient_id"]).any()
    yearly_observed = yearly_observed.join(patients.set_index("patient_id")["surgery_year"])
    year_coverage = (
        yearly_observed.groupby("surgery_year")[features]
        .sum()
        .T.reindex(index=features, columns=year_summary.index, fill_value=0)
        .div(year_summary["患者数"], axis=1)
        * 100
    )
    tables["年度概况"] = year_summary.reset_index()
    tables["年度特征覆盖"] = (
        year_coverage.rename_axis("特征")
        .reset_index()
        .rename(columns={year: f"{year}年覆盖率（%）" for year in year_coverage.columns})
    )
    cutoff_rows: list[dict[str, Any]] = []
    years = year_summary.index.tolist()
    for landmark, eligible in eligible_by_landmark.items():
        for cutoff in years[1:]:
            development, test = (
                eligible.loc[eligible["surgery_year"] < cutoff],
                eligible.loc[eligible["surgery_year"] >= cutoff],
            )
            cutoff_rows.append(
                {
                    "landmark（天）": landmark,
                    "时间测试集起始年份": cutoff,
                    "开发集人数": len(development),
                    "测试集人数": len(test),
                    f"开发集{outcome_label}事件数": int(development[event_column].eq(1).sum()),
                    f"测试集{outcome_label}事件数": int(test[event_column].eq(1).sum()),
                    f"测试集剩余{outcome_label}中位数（年）": (
                        test[time_column] - landmark
                    ).median()
                    / 365.25,
                }
            )
    cutoff_frame = pd.DataFrame(
        cutoff_rows,
        columns=[
            "landmark（天）",
            "时间测试集起始年份",
            "开发集人数",
            "测试集人数",
            f"开发集{outcome_label}事件数",
            f"测试集{outcome_label}事件数",
            f"测试集剩余{outcome_label}中位数（年）",
        ],
    )
    tables["时间切点比较"] = cutoff_frame

    ##################################################
    # 六、调用各图的绘制函数，复用上述统计结果。
    ##################################################
    _plot_survival_curves(patients, output, dpi)
    _plot_observation_patterns(counts, density, max_landmark, bin_days, output, dpi)
    _plot_landmark_population_shift(numeric_shift, categorical_shift, outcome_label, output, dpi)
    _plot_feature_coverage_matrix(coverage_matrix, bin_days, output, dpi)
    _plot_feature_value_patterns(value_patterns, output, dpi)
    _plot_calendar_time_patterns(year_summary, year_coverage, max_landmark, output, dpi)
    _plot_temporal_cutoff_tradeoff(cutoff_frame, landmark_days, outcome_label, output, dpi)

    ##################################################
    # 七、保存中文JSON和包含全部汇总表的Excel工作簿。
    ##################################################
    # pandas负责将嵌套统计中的NaN/NaT转为null，日期转为ISO字符串。
    payload = json.loads(pd.Series(summary).to_json(force_ascii=False, date_format="iso"))
    (output / "eda_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with pd.ExcelWriter(output / "eda_tables.xlsx", engine="openpyxl") as writer:
        for sheet, frame in tables.items():
            frame.to_excel(writer, sheet_name=sheet, index=False, inf_rep="不可计算")
            worksheet = writer.sheets[sheet]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column in worksheet.columns:
                letter = column[0].column_letter
                worksheet.column_dimensions[letter].width = min(
                    48, max(14, max(len(str(cell.value or "")) for cell in column) + 2)
                )
    print(
        f"CRC云南EDA完成：{len(patients):,}名患者，{int(matrix.count().sum()):,}个有效观测值；{output}"
    )


@hydra.main(config_path="../../configs", config_name="crc_yunnan/eda", version_base="1.3")
def main(raw_config: DictConfig) -> None:
    config = CRCEDAConfig.model_validate(OmegaConf.to_container(raw_config, resolve=True))
    run(config)


if __name__ == "__main__":
    main()

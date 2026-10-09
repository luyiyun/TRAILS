"""正文六面板图：统计由06提供，UMAP坐标不写入文件。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from .config import CRCFigureConfig
from .evaluation_plots import CLUSTER_MARKERS, CLUSTER_STYLES, english_label

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.gridspec import SubplotSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter  # noqa: E402


def _header(fig: Figure, cell: SubplotSpec, letter: str, title: str) -> None:
    box = cell.get_position(fig)
    offset = 6 / (fig.get_figheight() * 25.4)
    fig.text(box.x0, box.y1 + offset, letter, fontsize=8, weight="bold")
    fig.text(box.x0 + 0.025, box.y1 + offset, title, fontsize=6.5)


def _clean(axis: Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(length=2, width=0.5, pad=2, labelsize=6)
    axis.spines[["left", "bottom"]].set_linewidth(0.5)


def _setting(
    fig: Figure,
    cell: SubplotSpec,
    tables: dict[str, pd.DataFrame],
    frames: dict[str, pd.DataFrame],
    outcome: str,
    landmark: float,
    config: CRCFigureConfig,
) -> None:
    _header(fig, cell, "a", "Clinical setting and sparsity")
    grid = cell.subgridspec(2, 1, height_ratios=[1, 1], hspace=0.35)
    timeline = fig.add_subplot(grid[0])
    timeline.axis("off")
    months = landmark / 30
    timeline.set(xlim=(0, months * 1.65), ylim=(0, 1))
    timeline.text(
        0, 0.94, f"Postoperative CRC · {outcome.upper()} landmark: day {landmark:g}", fontsize=6
    )
    timeline.text(
        0,
        0.72,
        "   ".join(f"{english_label(name)} {len(frame):,}" for name, frame in frames.items()),
        fontsize=5.5,
    )
    timeline.plot([0, months], [0.42, 0.42], color="#52606A", lw=0.8)
    timeline.annotate(
        "",
        xy=(months * 1.6, 0.42),
        xytext=(months, 0.42),
        arrowprops={"arrowstyle": "->", "lw": 0.8},
    )
    timeline.text(0, 0.16, "Surgery", fontsize=5.5)
    timeline.text(months, 0.16, "Landmark", fontsize=5.5, ha="center")
    timeline.text(months / 2, 0.51, f"0–{months:g} months: model input", fontsize=5.5, ha="center")
    timeline.text(months * 1.4, -0.05, f"Subsequent {outcome.upper()}", fontsize=5.5, ha="center")
    timeline.text(
        0,
        -0.18,
        f"Density and coverage: {english_label(config.figure.profile_dataset)}",
        fontsize=5.5,
    )
    coverage = tables["Domain_coverage"]
    profile = config.figure.profile_dataset
    matrix = coverage.loc[coverage["dataset"].eq(profile)].pivot(
        index="feature_group", columns="bin", values="mean_feature_coverage"
    )
    priority = [
        "Tumor_marker",
        "Blood_routine_examination",
        "Liver_and_kidney_function",
        "Coagulation_function",
    ]
    matrix = matrix.reindex(
        [name for name in priority if name in matrix.index]
        + [name for name in matrix.index if name not in priority]
    )
    axis = fig.add_subplot(grid[1])
    edges = np.minimum(np.arange(len(matrix.columns) + 1) * config.trajectory_bin_days / 30, months)
    image = axis.pcolormesh(
        edges,
        np.arange(len(matrix) + 1) - 0.5,
        matrix.to_numpy(),
        vmin=0,
        vmax=1,
        cmap="cividis",
        rasterized=False,
    )
    axis.set(xlim=(0, months), ylim=(len(matrix) - 0.5, -0.5))
    short = {
        "Tumor_marker": "Tumor",
        "Blood_routine_examination": "Blood",
        "Liver_and_kidney_function": "Biochem.",
        "Coagulation_function": "Coag.",
    }
    axis.set_yticks(
        range(len(matrix)),
        [short.get(str(name), english_label(str(name))) for name in matrix.index],
    )
    axis.set_xticks([0, months / 2, months])
    axis.set_xlabel("Months after surgery", fontsize=6)
    axis.tick_params(length=0, labelsize=6)
    cbar = fig.colorbar(
        image, cax=axis.inset_axes((0.12, -0.48, 0.52, 0.08)), orientation="horizontal"
    )
    cbar.set_ticks([0, 0.5, 1], labels=["0", "50", "100%"])
    if cbar.solids is not None:
        cbar.solids.set_rasterized(False)
        cbar.solids.set_edgecolor("face")
    cbar.ax.tick_params(length=1, labelsize=5.5, pad=1)
    axis.text(0.7, -0.5, "Coverage", transform=axis.transAxes, fontsize=5.5)
    density = (
        tables["Density"]
        .loc[tables["Density"]["dataset"].eq(profile) & tables["Density"]["cluster"].eq(-1)]
        .iloc[0]
    )
    axis.text(
        -0.05,
        -0.85,
        (
            f"Dates/patient: {density['visits_median']:.0f} "
            f"(IQR {density['visits_q25']:.0f}–{density['visits_q75']:.0f})\n"
            f"Patient median gap: {density['median_gap_days_median']:.0f} d "
            f"(IQR {density['median_gap_days_q25']:.0f}–{density['median_gap_days_q75']:.1f})"
        ),
        transform=axis.transAxes,
        fontsize=5.5,
        va="top",
    )


def _atlas(
    fig: Figure,
    cell: SubplotSpec,
    profiles: pd.DataFrame,
    k: int,
    colors: dict[int, Any],
    config: CRCFigureConfig,
) -> None:
    _header(
        fig,
        cell,
        "b",
        f"Multivariate phenotype atlas ({english_label(config.figure.profile_dataset)})",
    )
    specs, phases = config.figure.atlas, config.figure.phase_days
    n_phases = len(phases) - 1
    selected = profiles.loc[profiles["dataset"].eq(config.figure.profile_dataset)]
    order = pd.MultiIndex.from_product([range(k), range(n_phases)], names=["cluster", "phase"])
    names = [spec.feature for spec in specs]
    values = selected.pivot(
        index="feature", columns=["cluster", "phase"], values="scaled_median"
    ).reindex(index=names, columns=order)
    counts = selected.pivot(index="feature", columns=["cluster", "phase"], values="n").reindex(
        index=names, columns=order
    )
    masked = np.ma.masked_where(
        counts.to_numpy() < config.figure.min_observed, values.to_numpy(dtype=float)
    )
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#D8D8D8")
    axis = fig.add_subplot(cell)
    image = axis.pcolormesh(
        np.arange(k * n_phases + 1) - 0.5,
        np.arange(len(names) + 1) - 0.5,
        masked,
        cmap=cmap,
        norm=TwoSlopeNorm(0, -2, 2),
        rasterized=False,
    )
    labels = [
        f"{start / 30:g}–{end / 30:g}" for start, end in zip(phases[:-1], phases[1:], strict=True)
    ]
    axis.set(
        xlim=(-0.5, k * n_phases - 0.5),
        ylim=(len(names) - 0.5, -0.5),
        yticks=range(len(names)),
        yticklabels=[spec.label for spec in specs],
        xticks=range(k * n_phases),
        xticklabels=labels * k,
    )
    axis.tick_params(length=0, labelsize=6, labelrotation=0)
    axis.tick_params(axis="x", labelrotation=60 if k <= 5 else 90)
    for cluster in range(k):
        axis.text(
            (cluster + 0.5) / k,
            1.015,
            f"C{cluster}",
            color=colors[cluster],
            transform=axis.transAxes,
            ha="center",
            fontsize=6.5,
            weight="bold",
        )
        if cluster:
            axis.axvline(cluster * n_phases - 0.5, color="white", lw=1)
    axis.set_xlabel("Months after surgery within each phenotype", fontsize=6)
    axis_height = axis.get_position().height * fig.get_figheight() * 25.4
    bar_depth = 14 / axis_height
    cbar = fig.colorbar(
        image,
        cax=axis.inset_axes((0.25, -bar_depth, 0.4, 1.5 / axis_height)),
        orientation="horizontal",
        extend="both",
    )
    cbar.ax.tick_params(labelsize=5.5, length=1, pad=1)
    if cbar.solids is not None:
        cbar.solids.set_rasterized(False)
    axis.text(
        0.72,
        -bar_depth - 0.025,
        f"Grey: n < {config.figure.min_observed}",
        fontsize=5.5,
        transform=axis.transAxes,
    )
    for spine in axis.spines.values():
        spine.set_visible(False)


def _trajectories(
    fig: Figure,
    cell: SubplotSpec,
    table: pd.DataFrame,
    landmark: float,
    colors: dict[int, Any],
    config: CRCFigureConfig,
) -> None:
    _header(fig, cell, "c", "Representative longitudinal profiles")
    specs = config.figure.trajectories
    rows = math.ceil(len(specs) / 3)
    grid = cell.subgridspec(
        rows + 1, 3, height_ratios=[0.1, *([1] * rows)], hspace=0.58, wspace=0.75
    )
    for index, spec in enumerate(specs):
        axis = fig.add_subplot(grid[index // 3 + 1, index % 3])
        selected = table.loc[
            table["dataset"].eq(config.figure.profile_dataset) & table["特征"].eq(spec.feature)
        ]
        for cluster, color in colors.items():
            values = selected.loc[selected["簇"].eq(cluster)].sort_values("时间箱")
            x = values["中点月份"].to_numpy(dtype=float)
            median = values["中位数"].to_numpy(dtype=float)
            lower, upper = (
                values["下四分位数"].to_numpy(dtype=float),
                values["上四分位数"].to_numpy(dtype=float),
            )
            valid = values["观测人数"].to_numpy() >= config.figure.min_observed
            if spec.scale == "log" and np.any(
                np.concatenate([median, lower, upper])[
                    np.isfinite(np.concatenate([median, lower, upper]))
                ]
                <= 0
            ):
                raise ValueError(f"{spec.feature}包含非正数，不能按配置绘制log轴")
            marker = CLUSTER_MARKERS[cluster % len(CLUSTER_MARKERS)]
            axis.plot(
                x,
                np.where(valid, median, np.nan),
                color=color,
                marker=marker,
                markersize=2,
                lw=0.7,
                linestyle=CLUSTER_STYLES[cluster % 4],
            )
            axis.fill_between(x, lower, upper, where=valid, color=color, alpha=0.08)
            axis.scatter(x[~valid], median[~valid], color="#AAAAAA", s=6, marker=marker)
        unit = spec.unit.replace("10^9", "10⁹")
        suffix = "; log" if spec.scale == "log" else ""
        axis.set_title(f"{spec.label} ({unit}{suffix})", fontsize=6, pad=3)
        axis.set_yscale(spec.scale)
        if spec.scale == "log":
            axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=5))
            axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            axis.yaxis.set_minor_formatter(NullFormatter())
        axis.set(xlim=(0, landmark / 30), xticks=[0, landmark / 60, landmark / 30])
        if index // 3 == rows - 1:
            axis.set_xlabel("Months after surgery", fontsize=5.5, labelpad=1)
        _clean(axis)
    box = cell.get_position(fig)
    handles = [
        Line2D(
            [],
            [],
            color=color,
            marker=CLUSTER_MARKERS[cluster % len(CLUSTER_MARKERS)],
            linestyle=CLUSTER_STYLES[cluster % 4],
            label=f"C{cluster}",
            markersize=3,
            lw=0.7,
        )
        for cluster, color in colors.items()
    ]
    legend_axis = fig.add_subplot(grid[0, :])
    legend_axis.axis("off")
    n_columns = min(len(colors), 8)
    handles = [
        handles[index]
        for column in range(n_columns)
        for index in range(column, len(handles), n_columns)
    ]
    legend_axis.legend(
        handles=handles,
        loc="center",
        ncol=n_columns,
        frameon=False,
        fontsize=5.5,
        borderaxespad=0,
        handlelength=1,
        columnspacing=0.6,
    )
    fig.text(
        box.x0,
        box.y0 - 0.042,
        (
            f"{english_label(config.figure.profile_dataset)} · median and IQR · "
            f"grey: n < {config.figure.min_observed}; disconnected\n"
            "Observed patients per bin: Source Data, Trajectories"
        ),
        fontsize=5.5,
        va="top",
    )


def _umap(
    fig: Figure,
    cell: SubplotSpec,
    projections: dict[str, np.ndarray],
    frames: dict[str, pd.DataFrame],
    colors: dict[int, Any],
    config: CRCFigureConfig,
) -> None:
    _header(fig, cell, "d", "Latent representation z")
    names = config.figure.display_datasets
    grid = cell.subgridspec(len(names), 1, hspace=0.65)
    all_points = np.concatenate([projections[name] for name in names]) if projections else None
    for row, name in enumerate(names):
        axis = fig.add_subplot(grid[row])
        if all_points is None:
            axis.text(
                0.5,
                0.5,
                "UMAP unavailable:\ntraining size <= n_neighbors",
                ha="center",
                va="center",
                fontsize=6,
                transform=axis.transAxes,
            )
            axis.axis("off")
            continue
        points = projections[name]
        for cluster, color in colors.items():
            selected = frames[name]["pred_cluster"].eq(cluster).to_numpy()
            axis.scatter(
                points[selected, 0],
                points[selected, 1],
                s=4,
                color=color,
                marker=CLUSTER_MARKERS[cluster % len(CLUSTER_MARKERS)],
                alpha=0.7,
                linewidths=0,
                rasterized=False,
            )
        lower, upper = all_points.min(axis=0), all_points.max(axis=0)
        pad = np.maximum((upper - lower) * 0.04, 0.01)
        axis.set(
            xlim=(lower[0] - pad[0], upper[0] + pad[0]), ylim=(lower[1] - pad[1], upper[1] + pad[1])
        )
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"{english_label(name)}, n = {len(frames[name]):,}", fontsize=6, pad=3)
        axis.set_ylabel("UMAP 2", fontsize=6, labelpad=1)
        if row == len(names) - 1:
            axis.set_xlabel("UMAP 1", fontsize=6, labelpad=1)
        _clean(axis)


def _survival(
    fig: Figure,
    cell: SubplotSpec,
    models: dict[str, dict[int, KaplanMeierFitter]],
    summaries: dict[str, dict[str, Any]],
    risk: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    outcome: str,
    colors: dict[int, Any],
    config: CRCFigureConfig,
) -> None:
    _header(fig, cell, "e", "Survival by phenotype")
    names, ticks = config.figure.display_datasets, config.figure.risk_months
    grid = cell.subgridspec(
        2, len(names), height_ratios=[36, 2.5 * (len(colors) + 1)], hspace=0.35, wspace=0.4
    )
    for column, name in enumerate(names):
        axis = fig.add_subplot(grid[0, column])
        for cluster, km in models[name].items():
            table = km.survival_function_.join(km.confidence_interval_)
            table.columns = ["survival", "lower", "upper"]
            table = table.loc[table.index <= config.figure.display_months]
            # 不延长至实际随访范围之外；只裁切显示，不改变KM或log-rank。
            last = min(config.figure.display_months, float(km.timeline[-1]))
            if table.index[-1] < last:
                table.loc[last] = table.iloc[-1]
            x = table.index.to_numpy(dtype=float)
            axis.step(
                x,
                table["survival"],
                where="post",
                color=colors[cluster],
                lw=0.8,
                linestyle=CLUSTER_STYLES[cluster % 4],
            )
            axis.fill_between(
                x,
                table["lower"].to_numpy(),
                table["upper"].to_numpy(),
                step="post",
                color=colors[cluster],
                alpha=0.08,
            )
            markers = np.asarray([month for month in ticks if 0 < month <= last])
            axis.plot(
                markers,
                km.predict(markers),
                marker=CLUSTER_MARKERS[cluster % len(CLUSTER_MARKERS)],
                markersize=2,
                linestyle="none",
                color=colors[cluster],
            )
        axis.set(
            xlim=(0, config.figure.display_months), ylim=(0, 1), xticks=ticks, yticks=[0, 0.5, 1]
        )
        axis.set_title(f"{english_label(name)}, n = {len(frames[name]):,}", fontsize=6, pad=3)
        axis.set_xlabel("Months after landmark", fontsize=5.5, labelpad=1)
        if column == 0:
            axis.set_ylabel(f"{outcome.upper()} probability", fontsize=6, labelpad=1)
        p_value = summaries[name]["总体log-rank_p值"]
        label = f"Log-rank P = {p_value:.2g}" if p_value is not None else "Log-rank unavailable"
        axis.text(0.03, 0.04, label, transform=axis.transAxes, fontsize=5.5)
        _clean(axis)
        at_risk = fig.add_subplot(grid[1, column])
        at_risk.axis("off")
        at_risk.set(xlim=(0, config.figure.display_months), ylim=(-0.5, len(colors) + 0.8))
        at_risk.text(-0.15, 1, "At risk", transform=at_risk.transAxes, fontsize=5.5)
        for cluster, color in colors.items():
            y = len(colors) - 1 - cluster
            at_risk.text(
                -0.15,
                (y + 0.5) / (len(colors) + 1.3),
                f"C{cluster}",
                transform=at_risk.transAxes,
                fontsize=5.5,
                color=color,
            )
            for month in ticks:
                count = risk.loc[
                    risk["dataset"].eq(name)
                    & risk["cluster"].eq(cluster)
                    & risk["month"].eq(month),
                    "at_risk",
                ].iloc[0]
                at_risk.text(month, y, str(int(count)), fontsize=5.5, ha="center", va="center")


def _cox(
    fig: Figure,
    cell: SubplotSpec,
    table: pd.DataFrame,
    summary: dict[str, Any],
    n: int,
    colors: dict[int, Any],
    config: CRCFigureConfig,
) -> None:
    _header(fig, cell, "f", "Adjusted survival associations")
    grid = cell.subgridspec(2, 1, height_ratios=[4, 1.2], hspace=0.35)
    axis = fig.add_subplot(grid[0])
    axis.set_title(f"{english_label(config.figure.cox_dataset)}, n = {n:,}", fontsize=6, pad=3)
    axis.set(
        yticks=range(len(colors)),
        yticklabels=[f"C{k}" for k in colors],
        ylim=(len(colors) - 0.5, -0.5),
    )
    for cluster, color in colors.items():
        values = table.loc[table["簇"].eq(cluster)]
        if values.empty:
            axis.text(
                1.04, cluster, "Not estimable", transform=axis.get_yaxis_transform(), fontsize=5.5
            )
            continue
        row = values.iloc[0]
        hr, lower, upper = float(row["HR"]), float(row["95%CI下限"]), float(row["95%CI上限"])
        if not np.isfinite([hr, lower, upper]).all() or min(hr, lower, upper) <= 0:
            axis.text(
                1.04, cluster, "Not estimable", transform=axis.get_yaxis_transform(), fontsize=5.5
            )
            continue
        axis.errorbar(
            hr,
            cluster,
            xerr=[[hr - lower], [upper - hr]],
            fmt=CLUSTER_MARKERS[cluster % len(CLUSTER_MARKERS)],
            color=color,
            markersize=3,
            lw=0.8,
            capsize=1.5,
        )
        printed = [
            f"{value:.2g}" if value < 0.01 or value >= 1000 else f"{value:.2f}"
            for value in (hr, lower, upper)
        ]
        label = (
            "1.00 (reference)"
            if cluster == summary["参考簇"]
            else f"{printed[0]} ({printed[1]}–{printed[2]})"
        )
        axis.text(
            1.04, cluster, label, transform=axis.get_yaxis_transform(), fontsize=5.5, va="center"
        )
    axis.axvline(1, color="#AAAAAA", linestyle="--", lw=0.7)
    axis.set_xscale("log")
    if table.empty:
        axis.set_xlim(0.5, 2)
        axis.xaxis.set_major_locator(FixedLocator([0.5, 1, 2]))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.xaxis.set_minor_formatter(NullFormatter())
    axis.set_xlabel("Hazard ratio", fontsize=6, labelpad=1)
    _clean(axis)
    # 右侧留出完整CI文本，避免截断极端HR。
    box = axis.get_position()
    axis.set_position((box.x0, box.y0, box.width * 0.42, box.height))
    note = fig.add_subplot(grid[1])
    note.axis("off")
    note.text(
        0,
        1,
        "Adjusted for age, sex, AJCC stage\nand primary site. All available follow-up.",
        fontsize=5.5,
        va="top",
    )
    if summary.get("不可计算原因"):
        note.text(0, 0.35, "Not estimable; see Source Data, Cox_details.", fontsize=5.5)


def plot_main_figure(
    tables: dict[str, pd.DataFrame],
    frames: dict[str, pd.DataFrame],
    projections: dict[str, np.ndarray],
    km_models: dict[str, dict[int, KaplanMeierFitter]],
    km_summaries: dict[str, dict[str, Any]],
    cox_summary: dict[str, Any],
    outcome: str,
    landmark: float,
    colors: dict[int, Any],
    config: CRCFigureConfig,
    stem: Path,
) -> tuple[float, float]:
    """保持180×215mm基准；新增指标、簇或展示集时扩展高度，不缩小字体。"""
    k = len(colors)
    extra_a = max(0, len(config.figure.atlas) - 13) * 3 + (7 if k > 5 else 0)
    extra_c = (
        max(0, math.ceil(len(config.figure.trajectories) / 3) - 2) * 30
        + max(0, math.ceil(k / 8) - 1) * 6
    )
    extra_c = max(extra_c, max(0, len(config.figure.display_datasets) - 2) * 32)
    extra_e = max(0, k - 5) * 3
    width = max(config.figure.width_mm, 180 + max(0, len(config.figure.display_datasets) - 2) * 48)
    height = max(config.figure.height_mm, 215 + extra_a + extra_c + extra_e)
    fig = plt.figure(figsize=(width / 25.4, height / 25.4), facecolor="white")
    plt.rcParams.update({"font.size": 6, "svg.fonttype": "none", "pdf.fonttype": 42})
    first = fig.add_gridspec(
        1,
        2,
        left=18 / width,
        right=1 - 5 / width,
        top=1 - 18 / height,
        bottom=1 - (58 + extra_a) / height,
        width_ratios=[42, 90],
        wspace=0.7,
    )
    second = fig.add_gridspec(
        1,
        2,
        left=10 / width,
        right=1 - 5 / width,
        top=1 - (87 + extra_a) / height,
        bottom=1 - (134 + extra_a + extra_c) / height,
        width_ratios=[100, 50],
        wspace=0.28,
    )
    third = fig.add_gridspec(
        1,
        2,
        left=11 / width,
        right=1 - 5 / width,
        top=1 - (159 + extra_a + extra_c) / height,
        bottom=5 / height,
        width_ratios=[100, 50],
        wspace=0.38,
    )
    _setting(fig, first[0], tables, frames, outcome, landmark, config)
    _atlas(fig, first[1], tables["Atlas"], k, colors, config)
    _trajectories(fig, second[0], tables["Trajectories"], landmark, colors, config)
    _umap(fig, second[1], projections, frames, colors, config)
    _survival(
        fig, third[0], km_models, km_summaries, tables["At_risk"], frames, outcome, colors, config
    )
    cox = tables["Cox"].loc[tables["Cox"]["dataset"].eq(config.figure.cox_dataset)]
    _cox(fig, third[1], cox, cox_summary, len(frames[config.figure.cox_dataset]), colors, config)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=config.plot.dpi, facecolor="white")
    plt.close(fig)
    return width, height

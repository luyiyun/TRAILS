"""06单图绘制函数；统计结果由主流程提供，统一中文标签和簇颜色。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts

from .config import BASELINE_LABELS
from .evaluation import SPLIT_LABELS

matplotlib.use("Agg")
import seaborn as sns  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


def save_figure(figure: Figure, stem: Path, dpi: int) -> None:
    """每张图同时提供屏幕PNG和可排版PDF。"""
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_cluster_sizes(table: pd.DataFrame, colors: dict[int, Any], stem: Path, dpi: int) -> None:
    datasets = table["数据集"].unique()
    figure, axes = plt.subplots(1, len(datasets), figsize=(4.5 * len(datasets), 4.4), squeeze=False)
    for axis, name in zip(axes.flat, datasets, strict=True):
        subset = table.loc[table["数据集"].eq(name)]
        bars = axis.bar(
            subset["簇"].astype(str),
            subset["占比（%）"],
            color=[colors[int(k)] for k in subset["簇"]],
        )
        axis.bar_label(
            bars,
            labels=[
                f"{n}人\n{p:.1f}%" for n, p in zip(subset["人数"], subset["占比（%）"], strict=True)
            ],
            padding=3,
        )
        axis.set(title=name, xlabel="簇", ylabel="患者占比（%）", ylim=(0, 110))
    figure.suptitle("最终模型的簇规模")
    figure.tight_layout()
    save_figure(figure, stem, dpi)


def plot_umap(
    projections: dict[str, np.ndarray],
    frames: dict[str, pd.DataFrame],
    colors: dict[int, Any],
    stem: Path,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(
        2,
        len(projections),
        figsize=(5 * len(projections), 9),
        squeeze=False,
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    all_risk = np.concatenate([frames[name]["risk_score"].to_numpy() for name in projections])
    normalization = plt.Normalize(float(all_risk.min()), float(all_risk.max()))
    scatter = None
    for column, (name, points) in enumerate(projections.items()):
        frame = frames[name]
        for cluster, color in colors.items():
            selected = frame["pred_cluster"].eq(cluster).to_numpy()
            axes[0, column].scatter(
                points[selected, 0],
                points[selected, 1],
                s=8,
                alpha=0.65,
                color=color,
                label=f"簇{cluster}",
                rasterized=True,
            )
        axes[0, column].legend(markerscale=2, fontsize=9)
        scatter = axes[1, column].scatter(
            points[:, 0],
            points[:, 1],
            c=frame["risk_score"],
            cmap="viridis",
            norm=normalization,
            s=8,
            alpha=0.7,
            rasterized=True,
        )
        for row in range(2):
            axes[row, column].set(
                title=SPLIT_LABELS.get(name, name), xlabel="UMAP 1", ylabel="UMAP 2"
            )
    if scatter is not None:
        figure.colorbar(scatter, ax=list(axes[1]), label="预测风险（越高越危险）", shrink=0.8)
    figure.suptitle(
        "潜空间结构：上排按簇，下排按预测风险\nUMAP仅在训练集拟合；二维分离不代表聚类有效性"
    )
    save_figure(figure, stem, dpi)


def plot_confidence(
    frames: dict[str, pd.DataFrame], colors: dict[int, Any], stem: Path, dpi: int
) -> None:
    figure, axes = plt.subplots(
        2, len(frames), figsize=(4.5 * len(frames), 8), squeeze=False, layout="constrained"
    )
    for column, (name, frame) in enumerate(frames.items()):
        for row, (metric, label) in enumerate(
            (("confidence", "最大簇后验概率"), ("entropy", "归一化后验熵"))
        ):
            sns.boxplot(
                data=frame,
                x="pred_cluster",
                y=metric,
                hue="pred_cluster",
                palette=colors,
                saturation=1,
                order=list(colors),
                legend=False,
                showfliers=False,
                ax=axes[row, column],
            )
            axes[row, column].set(
                title=SPLIT_LABELS.get(name, name), xlabel="簇", ylabel=label, ylim=(0, 1.03)
            )
    figure.suptitle("簇分配不确定性（箱线图不显示离群点，统计包含全部患者）")
    save_figure(figure, stem, dpi)


def plot_km(
    models: dict[str, dict[int, KaplanMeierFitter]],
    summaries: dict[str, dict[str, Any]],
    colors: dict[int, Any],
    outcome: str,
    stem: Path,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 6.8), squeeze=False)
    for axis, (name, fits) in zip(axes.flat, models.items(), strict=True):
        for cluster, km in fits.items():
            km.plot_survival_function(
                ax=axis, color=colors[cluster], ci_show=True, show_censors=True
            )
        p_value = summaries[name]["总体log-rank_p值"]
        text = f"总体log-rank p={p_value:.3g}" if p_value is not None else "log-rank不可计算"
        axis.set(
            title=f"{SPLIT_LABELS.get(name, name)}\n{text}",
            xlabel="landmark后月份",
            ylabel=f"{outcome.upper()}生存率",
            ylim=(0, 1.03),
            xlim=(0, None),
        )
        add_at_risk_counts(
            *fits.values(),
            ax=axis,
            rows_to_show=["At risk"],
            ypos=-0.4,
            at_risk_count_from_start_of_period=True,
        )
    figure.suptitle("各簇Kaplan–Meier生存曲线（阴影为95% CI）")
    figure.subplots_adjust(bottom=0.3, top=0.85, wspace=0.3)
    # lifelines固定的英文风险表标签在该图中统一翻译。
    for axis in figure.axes:
        labels = [
            label.get_text().replace("At risk", "风险人数") for label in axis.get_xticklabels()
        ]
        if any("风险人数" in label for label in labels):
            axis.set_xticks(axis.get_xticks(), labels=labels)
    save_figure(figure, stem, dpi)


def plot_cox(
    table: pd.DataFrame,
    summaries: dict[str, dict[str, Any]],
    colors: dict[int, Any],
    stem: Path,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(
        1, len(summaries), figsize=(5.4 * len(summaries), 4.8), squeeze=False, layout="constrained"
    )
    for axis, (name, summary) in zip(axes.flat, summaries.items(), strict=True):
        rows = table.loc[table["数据集"].eq(SPLIT_LABELS.get(name, name))]
        if rows.empty:
            axis.text(
                0.5,
                0.5,
                summary["不可计算原因"],
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_axis_off()
            continue
        for index, (_, row) in enumerate(rows.iterrows()):
            axis.errorbar(
                row["HR"],
                index,
                xerr=[[row["HR"] - row["95%CI下限"]], [row["95%CI上限"] - row["HR"]]],
                fmt="o",
                color=colors[int(row["簇"])],
                capsize=3,
            )
            axis.annotate(
                f"{row['HR']:.2f} [{row['95%CI下限']:.2f}, {row['95%CI上限']:.2f}]",
                (row["HR"], index),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        axis.axvline(1, color=".5", linestyle="--")
        axis.set_xscale("log")
        axis.set(
            yticks=range(len(rows)),
            yticklabels=[
                f"簇{int(row['簇'])}" + ("（参考）" if row["参考簇"] else "")
                for _, row in rows.iterrows()
            ],
            xlabel="调整HR及95% CI（对数刻度）",
            title=SPLIT_LABELS.get(name, name),
            ylim=(-0.6, len(rows) - 0.4),
        )
        axis.invert_yaxis()
    figure.suptitle("簇间调整后生存关联\n调整年龄、性别、AJCC分期、原发部位；参考簇由训练集确定")
    save_figure(figure, stem, dpi)


def plot_numeric_baseline(
    frames: dict[str, pd.DataFrame],
    variables: list[str],
    colors: dict[int, Any],
    stem: Path,
    dpi: int,
) -> None:
    if not variables:
        return
    figure, axes = plt.subplots(
        len(variables),
        len(frames),
        figsize=(4.5 * len(frames), 3.5 * len(variables)),
        squeeze=False,
        layout="constrained",
    )
    for row, variable in enumerate(variables):
        for column, (name, frame) in enumerate(frames.items()):
            axis = axes[row, column]
            sns.boxplot(
                data=frame,
                x="pred_cluster",
                y=variable,
                hue="pred_cluster",
                palette=colors,
                saturation=1,
                order=list(colors),
                showfliers=False,
                legend=False,
                ax=axis,
            )
            axis.set(
                xlabel="簇",
                ylabel=BASELINE_LABELS.get(variable, variable),
                title=SPLIT_LABELS.get(name, name),
            )
    figure.suptitle("连续基线特征（统计含全部有效值；箱线图不显示离群点）")
    save_figure(figure, stem, dpi)


def plot_categorical_baseline(
    frames: dict[str, pd.DataFrame], variables: list[str], n_clusters: int, stem: Path, dpi: int
) -> None:
    figure, axes = plt.subplots(
        len(variables),
        len(frames),
        figsize=(5 * len(frames), 4 * len(variables)),
        squeeze=False,
        layout="constrained",
    )
    for row, variable in enumerate(variables):
        levels = sorted(
            set(
                pd.concat([frame[variable].fillna("缺失").astype(str) for frame in frames.values()])
            )
        )
        palette = sns.color_palette("Set2", len(levels))
        for column, (name, frame) in enumerate(frames.items()):
            axis = axes[row, column]
            counts = pd.crosstab(
                frame["pred_cluster"], frame[variable].fillna("缺失").astype(str)
            ).reindex(index=range(n_clusters), columns=levels, fill_value=0)
            proportions = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
            bottom = np.zeros(n_clusters)
            for level, color in zip(levels, palette, strict=True):
                values = proportions[level].to_numpy()
                axis.bar(np.arange(n_clusters), values, bottom=bottom, color=color, label=level)
                bottom += values
            axis.set(
                title=f"{SPLIT_LABELS.get(name, name)}：{BASELINE_LABELS.get(variable, variable)}",
                xlabel="簇",
                ylabel="簇内比例（%，含缺失）",
                xticks=range(n_clusters),
                ylim=(0, 100),
            )
            axis.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=2)
    save_figure(figure, stem, dpi)


def plot_trajectories(
    table: pd.DataFrame,
    features: list[str],
    title: str,
    colors: dict[int, Any],
    stem: Path,
    dpi: int,
) -> None:
    columns = min(3, len(features))
    figure, axes = plt.subplots(
        int(np.ceil(len(features) / columns)),
        columns,
        figsize=(5 * columns, 3.4 * int(np.ceil(len(features) / columns))),
        squeeze=False,
        layout="constrained",
    )
    for axis, feature in zip(axes.flat, features, strict=False):
        subset = table.loc[table["特征"].eq(feature)]
        for cluster, color in colors.items():
            rows = subset.loc[subset["簇"].eq(cluster)].sort_values("时间箱")
            x = rows["中点月份"].to_numpy(dtype=float)
            axis.plot(
                x,
                rows["中位数"].to_numpy(dtype=float),
                marker=".",
                color=color,
                label=f"簇{cluster}",
            )
            axis.fill_between(
                x,
                rows["下四分位数"].to_numpy(dtype=float),
                rows["上四分位数"].to_numpy(dtype=float),
                color=color,
                alpha=0.15,
            )
        axis.set(title=feature, xlabel="术后月份", ylabel="原始尺度中位数（IQR）")
        axis.grid(alpha=0.2)
    for axis in list(axes.flat)[len(features) :]:
        axis.set_axis_off()
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=len(colors))
    figure.suptitle(title + "\n先取患者箱内中位数，再汇总各簇；缺失箱不连线，样本量见Excel")
    save_figure(figure, stem, dpi)


def plot_trajectory_coverage(
    table: pd.DataFrame,
    features: list[str],
    colors: dict[int, Any],
    title: str,
    stem: Path,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(
        1,
        len(colors),
        figsize=(5 * len(colors), max(5, 0.22 * len(features))),
        squeeze=False,
        layout="constrained",
    )
    for axis, cluster in zip(axes.flat, colors, strict=True):
        matrix = (
            table.loc[table["簇"].eq(cluster)]
            .pivot(index="特征", columns="中点月份", values="覆盖率（%）")
            .reindex(features)
        )
        sns.heatmap(
            matrix,
            vmin=0,
            vmax=100,
            cmap="viridis",
            ax=axis,
            cbar=cluster == list(colors)[-1],
            cbar_kws={"label": "覆盖率（%）"},
        )
        axis.set(title=f"簇{cluster}", xlabel="术后月份（时间箱中点）", ylabel="特征")
    figure.suptitle(
        title + "：轨迹覆盖率\n分母为该数据集该簇全部患者；分子为箱内该特征至少一次观测的患者"
    )
    save_figure(figure, stem, dpi)


def plot_survival_metrics(
    curves: pd.DataFrame, metrics: pd.DataFrame, months: list[int], stem: Path, dpi: int
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.7), layout="constrained")
    positions = np.arange(len(metrics))
    for offset, metric, label in [
        (-0.18, "Harrell_C-index", "Harrell"),
        (0.18, "IPCW_C-index", f"IPCW（{months[-1]}月内）"),
    ]:
        bars = axes[0].bar(
            positions + offset, metrics[metric].astype(float), width=0.36, label=label
        )
        axes[0].bar_label(bars, fmt="%.3f", fontsize=8)
    axes[0].set(xticks=positions, xticklabels=metrics["数据集"], ylim=(0, 1), ylabel="C-index")
    axes[0].legend()
    for name, rows in curves.groupby("数据集", sort=False):
        axes[1].plot(rows["月份"], rows["动态AUC"], label=name)
        axes[2].plot(rows["月份"], rows["Brier"], label=name)
    for axis, ylabel in zip(axes[1:], ("动态AUC", "Brier分数"), strict=True):
        for month in months:
            axis.axvline(month, color=".7", linestyle=":", linewidth=0.8)
        axis.set(xlabel="landmark后月份", ylabel=ylabel, ylim=(0, 1), xticks=months)
        axis.legend()
        axis.grid(alpha=0.2)
    figure.suptitle("个体生存预测表现（删失分布仅由训练集估计）")
    save_figure(figure, stem, dpi)


def plot_calibration(table: pd.DataFrame, stem: Path, dpi: int) -> None:
    datasets, months = table["数据集"].unique(), table["月份"].unique()
    figure, axes = plt.subplots(
        len(datasets),
        len(months),
        figsize=(4.5 * len(months), 4.4 * len(datasets)),
        squeeze=False,
        layout="constrained",
    )
    for row, name in enumerate(datasets):
        for column, month in enumerate(months):
            axis = axes[row, column]
            rows = table.loc[table["数据集"].eq(name) & table["月份"].eq(month)].dropna(
                subset=["KM生存率"]
            )
            axis.plot([0, 1], [0, 1], "--", color=".5")
            axis.errorbar(
                rows["平均预测生存率"],
                rows["KM生存率"],
                yerr=np.vstack(
                    [rows["KM生存率"] - rows["95%CI下限"], rows["95%CI上限"] - rows["KM生存率"]]
                ),
                fmt="o-",
                capsize=3,
            )
            for _, item in rows.iterrows():
                axis.annotate(
                    f"n={int(item['人数'])}",
                    (item["平均预测生存率"], item["KM生存率"]),
                    xytext=(3, 5),
                    textcoords="offset points",
                    fontsize=8,
                )
            axis.set(
                title=f"{name}：{month}个月",
                xlabel="平均预测生存率",
                ylabel="KM观察生存率（95% CI）",
                xlim=(0, 1),
                ylim=(0, 1),
            )
            axis.set_aspect("equal")
    figure.suptitle("生存校准：按各时间点预测生存率分位分组")
    save_figure(figure, stem, dpi)


def plot_overall_survival(table: pd.DataFrame, stem: Path, dpi: int) -> None:
    datasets = table["数据集"].unique()
    figure, axes = plt.subplots(
        1, len(datasets), figsize=(5 * len(datasets), 4.7), squeeze=False, layout="constrained"
    )
    for axis, name in zip(axes.flat, datasets, strict=True):
        rows = table.loc[table["数据集"].eq(name)]
        x = rows["月份"].to_numpy(dtype=float)
        axis.plot(x, rows["平均预测生存率"], label="平均预测生存率")
        axis.step(x, rows["KM生存率"], where="post", label="KM观察生存率")
        axis.fill_between(
            x,
            rows["95%CI下限"].to_numpy(dtype=float),
            rows["95%CI上限"].to_numpy(dtype=float),
            step="post",
            alpha=0.2,
        )
        axis.set(title=name, xlabel="landmark后月份", ylabel="生存率", ylim=(0, 1.03))
        axis.legend()
    figure.suptitle("整体预测与观察生存曲线（阴影为KM的95% CI）")
    save_figure(figure, stem, dpi)


def plot_training(history: pd.DataFrame, stem: Path, dpi: int) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
    panels = [
        ("reconstruction_loss", "原始重构损失"),
        ("survival_loss", "原始生存损失"),
        ("vade_kl_loss", "原始聚类损失"),
        ("loss", "加权总损失"),
        ("cindex", "C-index"),
    ]
    for axis, (metric, title) in zip(axes.flat, panels, strict=False):
        for prefix, label in [("", "训练"), ("val_", "验证")]:
            if prefix + metric in history:
                axis.plot(history["global_epoch"], history[prefix + metric], label=label)
        axis.set(title=title, xlabel="全局epoch")
        axis.legend()
    weight_axis = axes.flat[-1]
    for metric, label in [
        ("reconstruction_loss_weight", "重构"),
        ("survival_loss_weight", "生存"),
        ("vade_kl_loss_weight", "聚类"),
    ]:
        if metric in history:
            weight_axis.plot(history["global_epoch"], history[metric], label=label)
    weight_axis.set(title="学习到的损失权重", xlabel="全局epoch")
    weight_axis.legend()
    changes = history.loc[history["stage"].ne(history["stage"].shift())]
    checkpoint = (
        history["best_global_epoch"].dropna().iloc[-1]
        if "best_global_epoch" in history and history["best_global_epoch"].notna().to_numpy().any()
        else None
    )
    for axis in axes.flat:
        if len(changes) > 1:
            axis.axvline(
                changes["global_epoch"].iloc[1] - 0.5,
                linestyle="--",
                color=".5",
                label="warmup结束",
            )
        if checkpoint is not None:
            axis.axvline(checkpoint, linestyle=":", color="#009E73", label="保存检查点")
        axis.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    figure.suptitle("最终模型训练过程；总损失的下降应结合原始分量和权重解释")
    save_figure(figure, stem, dpi)


def plot_k_selection(
    runs: pd.DataFrame, summary: pd.DataFrame, selected_k: int, minimum: float, stem: Path, dpi: int
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    for axis, metric, title in zip(
        axes.flat,
        ["selection_score", "cindex", "cluster_min_fraction", "cluster_empty_count"],
        ["验证选择分数", "验证C-index", "验证最小簇比例", "验证空簇数"],
        strict=True,
    ):
        for seed, rows in runs.groupby("seed", sort=True):
            axis.plot(rows["n_clusters"], rows[metric], ".--", alpha=0.6, label=str(seed))
        axis.axvline(selected_k, color="black", linestyle=":", label="选中K")
        axis.set(title=title, xlabel="候选K", xticks=summary["n_clusters"])
        axis.grid(alpha=0.2)
    axes[0, 0].errorbar(
        summary["n_clusters"],
        summary["mean_selection_score"],
        yerr=summary["se_selection_score"],
        color="black",
        fmt="o-",
        label="跨seed均值±SE",
    )
    passes = summary.loc[summary["passes_gate"]]
    axes[0, 0].scatter(
        passes["n_clusters"],
        passes["mean_selection_score"],
        s=140,
        facecolors="none",
        edgecolors="#009E73",
        linewidths=2,
        label="通过全部seed门槛",
    )
    axes[1, 0].axhline(minimum, color="red", linestyle="--", label=f"{minimum:.0%}门槛")
    axes[1, 0].legend(fontsize=8)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("K选择诊断（展示既有选择，不重新选K或seed）")
    save_figure(figure, stem, dpi)


def plot_stability(pairs: pd.DataFrame, selected_k: int, stem: Path, dpi: int) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    summary: pd.DataFrame = pairs.groupby("K").agg(  # type: ignore[reportAssignmentType]
        mean=("ARI", "mean"), min=("ARI", "min"), max=("ARI", "max")
    )
    x = summary.index.to_numpy(dtype=int)
    axes[0].plot(x, summary["mean"], "o-", label="两两ARI均值")
    axes[0].fill_between(
        x, summary["min"].to_numpy(), summary["max"].to_numpy(), alpha=0.2, label="最小–最大范围"
    )
    invalid = pairs.loc[~pairs["两候选均通过占用门槛"]]
    axes[0].scatter(invalid["K"], invalid["ARI"], color="red", marker="x", label="含退化候选的比较")
    axes[0].axvline(selected_k, color="black", linestyle=":")
    axes[0].set(
        xlabel="候选K", ylabel="ARI", xticks=x, ylim=(-1, 1.05), title="固定验证集上的跨seed一致性"
    )
    axes[0].legend(fontsize=8)
    selected = pairs.loc[pairs["K"].eq(selected_k)]
    seeds = sorted(set(selected["seed_1"]) | set(selected["seed_2"]))
    if seeds:
        matrix = pd.DataFrame(np.eye(len(seeds)), index=seeds, columns=seeds)
        for _, row in selected.iterrows():
            matrix.loc[int(row["seed_1"]), int(row["seed_2"])] = row["ARI"]
            matrix.loc[int(row["seed_2"]), int(row["seed_1"])] = row["ARI"]
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            vmin=-1,
            vmax=1,
            center=0,
            cmap="vlag",
            square=True,
            ax=axes[1],
        )
    axes[1].set_title(f"选中K={selected_k}：两两ARI")
    figure.suptitle("稳定性附录候选图；ARI需结合空簇和最小簇占比解释")
    save_figure(figure, stem, dpi)

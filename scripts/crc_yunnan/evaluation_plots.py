"""06单图绘制函数；统计结果由主流程提供，统一英文标签和簇颜色。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts

from .config import CRCPostprocessConfig
from .evaluation import SPLIT_LABELS

matplotlib.use("Agg")
import seaborn as sns  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.collections import Collection  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

CLUSTER_COLORS = [
    "#0072B2",
    "#E69F00",
    "#CC79A7",
    "#666666",
    "#D55E00",
    "#009E73",
    "#56B4E9",
    "#F0E442",
    "#000000",
]
CLUSTER_MARKERS = ["o", "^", "s", "D", "P", "v", "X", "<", ">", "h"]
CLUSTER_STYLES = ["-", "--", "-.", ":"]


def configure_plotting(config: CRCPostprocessConfig, n_clusters: int) -> dict[int, Any]:
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected = [name for name in config.plot.font_families if name in fonts]
    if not selected:
        raise ValueError("未找到配置中的绘图字体")
    plt.rcParams.update(
        {
            "font.family": selected,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 9,
        }
    )
    return {
        k: CLUSTER_COLORS[k]
        if k < len(CLUSTER_COLORS)
        else plt.get_cmap("tab20")((k - len(CLUSTER_COLORS)) % 20)
        for k in range(n_clusters)
    }


ENGLISH_LABELS = {
    "train": "Training",
    "validation": "Validation",
    "test": "Test",
    "训练集": "Training",
    "验证集": "Validation",
    "测试集": "Test",
    "缺失": "Missing",
    "未记录": "Unrecorded",
    "Preoperative_CEA": "Preoperative CEA",
    "Preoperative_CA242": "Preoperative CA242",
    "Age": "Age",
    "Sex": "Sex",
    "Primary_site": "Primary site",
    "Surgical_approach": "Surgical approach",
    "Tumor_differentiation": "Tumor differentiation",
    "AJCC_8th_ed_Stage": "AJCC 8th edition stage",
    "Lymph_node_yield": "Lymph node yield",
    "Mucinous_colloid_type": "Mucinous/colloid type",
    "Lymphovascular_invasion": "Lymphovascular invasion",
    "Perineural_invasion": "Perineural invasion",
    "Adjuvant_chemotherapy": "Adjuvant chemotherapy",
    "Tumor_marker": "Tumor markers",
    "Liver_and_kidney_function": "Liver and kidney function",
    "Blood_routine_examination": "Complete blood count",
    "Coagulation_function": "Coagulation function",
    "血清白蛋白": "Serum albumin",
    "血红蛋白": "Hemoglobin",
    "中性粒细胞绝对值": "Absolute neutrophil count",
    "淋巴细胞绝对值": "Absolute lymphocyte count",
    "血小板": "Platelet count",
    "肌酐": "Creatinine",
    "前白蛋白": "Prealbumin",
    "谷丙转氨酶": "ALT",
    "总胆红素": "Total bilirubin",
    "白细胞绝对值": "White blood cell count",
    "单核细胞绝对值": "Absolute monocyte count",
    "尿素": "Urea",
    "尿酸": "Uric acid",
    "谷草转氨酶": "AST",
    "直接胆红素": "Direct bilirubin",
    "碱性磷酸酶": "Alkaline phosphatase",
    "r谷氨酰转移酶": "Gamma-glutamyl transferase",
    "嗜碱性粒细胞绝对值": "Absolute basophil count",
    "嗜酸性粒细胞绝对值": "Absolute eosinophil count",
    "大型血小板比率": "Large platelet ratio",
    "平均血小板体积": "Mean platelet volume",
    "平均血红蛋白浓度": "Mean corpuscular hemoglobin concentration",
    "平均血红蛋白量": "Mean corpuscular hemoglobin",
    "红细胞": "Red blood cell count",
    "红细胞体积分布宽度": "Red cell distribution width (SD)",
    "红细胞体积变异系数": "Red cell distribution width (CV)",
    "红细胞压积": "Hematocrit",
    "红细胞平均体积": "Mean corpuscular volume",
    "血小板分布宽度": "Platelet distribution width",
    "血小板压积": "Plateletcrit",
    "PT_INR": "PT-INR",
    "凝血酶原时间": "Prothrombin time",
    "凝血酶时间": "Thrombin time",
    "活化部分凝血活酶时间": "Activated partial thromboplastin time",
    "纤维蛋白原": "Fibrinogen",
    "部分凝血活酶时间比率": "Partial thromboplastin time ratio",
    "C反应蛋白": "C-reactive protein",
    "aL岩藻糖甘酶": "Alpha-L-fucosidase",
    "a羟丁酸脱氢酶": "Alpha-hydroxybutyrate dehydrogenase",
    "乳酸脱氢酶": "Lactate dehydrogenase",
    "总胆固醇": "Total cholesterol",
    "总胆汁酸": "Total bile acids",
    "总蛋白": "Total protein",
    "白球比": "Albumin/globulin ratio",
    "肌酸激酶": "Creatine kinase",
    "肌酸激酶同工酶": "Creatine kinase isoenzyme",
    "胆碱酯酶": "Cholinesterase",
    "腺苷脱氨酶": "Adenosine deaminase",
    "血清低密度脂蛋白": "Serum LDL",
    "血清球蛋白": "Serum globulin",
    "血清高密度脂蛋白": "Serum HDL",
    "谷草_谷丙比值": "AST/ALT ratio",
    "间接胆红素": "Indirect bilirubin",
    "完整病例无事件": "No events among complete cases",
    "完整病例不足两个有患者的簇": "Fewer than two occupied clusters among complete cases",
    "当前数据集没有训练集确定的参考簇": "Training-defined reference cluster absent",
}


def english_label(value: str) -> str:
    """只翻译展示标签，不改变原始字段、类别顺序或统计分组。"""
    label = ENGLISH_LABELS.get(value, value.replace("_", " "))
    if re.search(r"[\u3400-\u9fff]", label):
        raise ValueError(f"缺少绘图标签的英文翻译：{value}")
    return label


def save_figure(figure: Figure, stem: Path, dpi: int) -> None:
    """每张图提供PNG和可编辑PDF/SVG，不自动栅格化色条。"""
    stem.parent.mkdir(parents=True, exist_ok=True)
    for collection in figure.findobj(match=Collection):
        collection.set_rasterized(False)
    figure.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
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
                f"n={n}\n{p:.1f}%" for n, p in zip(subset["人数"], subset["占比（%）"], strict=True)
            ],
            padding=3,
        )
        axis.set(title=english_label(name), xlabel="Cluster", ylabel="Patients (%)", ylim=(0, 110))
    figure.suptitle("Cluster sizes")
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
                label=f"Cluster {cluster}",
                rasterized=False,
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
            axes[row, column].set(title=english_label(name), xlabel="UMAP 1", ylabel="UMAP 2")
    if scatter is not None:
        figure.colorbar(
            scatter, ax=list(axes[1]), label="Predicted risk (higher = greater risk)", shrink=0.8
        )
    figure.suptitle(
        "Latent space: clusters (top) and predicted risk (bottom)\n"
        "UMAP fitted on training data only; 2D separation does not establish cluster validity"
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
            (
                ("confidence", "Maximum cluster probability"),
                ("entropy", "Normalized posterior entropy"),
            )
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
                title=english_label(name), xlabel="Cluster", ylabel=label, ylim=(0, 1.03)
            )
    figure.suptitle(
        "Cluster assignment uncertainty (outliers hidden; summaries include all patients)"
    )
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
        text = (
            f"Overall log-rank p={p_value:.3g}" if p_value is not None else "Log-rank unavailable"
        )
        axis.set(
            title=f"{english_label(name)}\n{text}",
            xlabel="Months after landmark",
            ylabel=f"{outcome.upper()} survival probability",
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
    figure.suptitle("Kaplan–Meier survival by cluster (shading: 95% CI)")
    figure.subplots_adjust(bottom=0.3, top=0.85, wspace=0.3)
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
                english_label(summary["不可计算原因"]),
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
                f"Cluster {int(row['簇'])}" + (" (reference)" if row["参考簇"] else "")
                for _, row in rows.iterrows()
            ],
            xlabel="Adjusted HR with 95% CI (log scale)",
            title=english_label(name),
            ylim=(-0.6, len(rows) - 0.4),
        )
        axis.invert_yaxis()
    figure.suptitle(
        "Adjusted survival association by cluster\n"
        "Adjusted for age, sex, AJCC stage and primary site; reference defined in training data"
    )
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
                xlabel="Cluster",
                ylabel=english_label(variable),
                title=english_label(name),
            )
    figure.suptitle(
        "Continuous baseline characteristics (all observed values summarized; outliers hidden)"
    )
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
                axis.bar(
                    np.arange(n_clusters),
                    values,
                    bottom=bottom,
                    color=color,
                    label=english_label(level),
                )
                bottom += values
            axis.set(
                title=f"{english_label(name)}: {english_label(variable)}",
                xlabel="Cluster",
                ylabel="Within-cluster proportion (%; missing included)",
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
                label=f"Cluster {cluster}",
            )
            axis.fill_between(
                x,
                rows["下四分位数"].to_numpy(dtype=float),
                rows["上四分位数"].to_numpy(dtype=float),
                color=color,
                alpha=0.15,
            )
        axis.set(
            title=english_label(feature),
            xlabel="Months after surgery",
            ylabel="Median (IQR), original scale",
        )
        axis.grid(alpha=0.2)
    for axis in list(axes.flat)[len(features) :]:
        axis.set_axis_off()
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=len(colors))
    figure.suptitle(
        english_label(title) + "\nPatient medians within bins, then cluster summaries; "
        "gaps remain unconnected; sample sizes in workbook"
    )
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
            cbar_kws={"label": "Coverage (%)"},
            yticklabels=[english_label(feature) for feature in features],
        )
        axis.set(
            title=f"Cluster {cluster}",
            xlabel="Months after surgery (bin midpoint)",
            ylabel="Feature",
        )
    figure.suptitle(
        english_label(title) + ": trajectory coverage\n"
        "Patients with at least one feature measurement per bin / all patients in the cluster"
    )
    save_figure(figure, stem, dpi)


def plot_survival_metrics(
    curves: pd.DataFrame, metrics: pd.DataFrame, months: list[int], stem: Path, dpi: int
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.7), layout="constrained")
    positions = np.arange(len(metrics))
    for offset, metric, label in [
        (-0.18, "Harrell_C-index", "Harrell"),
        (0.18, "IPCW_C-index", f"IPCW (within {months[-1]} months)"),
    ]:
        bars = axes[0].bar(
            positions + offset, metrics[metric].astype(float), width=0.36, label=label
        )
        axes[0].bar_label(bars, fmt="%.3f", fontsize=8)
    axes[0].set(
        xticks=positions,
        xticklabels=[english_label(name) for name in metrics["数据集"]],
        ylim=(0, 1),
        ylabel="C-index",
    )
    axes[0].legend()
    for name, rows in curves.groupby("数据集", sort=False):
        axes[1].plot(rows["月份"], rows["动态AUC"], label=english_label(str(name)))
        axes[2].plot(rows["月份"], rows["Brier"], label=english_label(str(name)))
    for axis, ylabel in zip(axes[1:], ("Time-dependent AUC", "Brier score"), strict=True):
        for month in months:
            axis.axvline(month, color=".7", linestyle=":", linewidth=0.8)
        axis.set(xlabel="Months after landmark", ylabel=ylabel, ylim=(0, 1), xticks=months)
        axis.legend()
        axis.grid(alpha=0.2)
    figure.suptitle(
        "Individual survival prediction (censoring distribution estimated from training data only)"
    )
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
                title=f"{english_label(name)}: {month} months",
                xlabel="Mean predicted survival probability",
                ylabel="Observed KM survival probability (95% CI)",
                xlim=(0, 1),
                ylim=(0, 1),
            )
            axis.set_aspect("equal")
    figure.suptitle(
        "Survival calibration: quantile groups of predicted survival at each time point"
    )
    save_figure(figure, stem, dpi)


def plot_overall_survival(table: pd.DataFrame, stem: Path, dpi: int) -> None:
    datasets = table["数据集"].unique()
    figure, axes = plt.subplots(
        1, len(datasets), figsize=(5 * len(datasets), 4.7), squeeze=False, layout="constrained"
    )
    for axis, name in zip(axes.flat, datasets, strict=True):
        rows = table.loc[table["数据集"].eq(name)]
        x = rows["月份"].to_numpy(dtype=float)
        axis.plot(x, rows["平均预测生存率"], label="Mean predicted survival")
        axis.step(x, rows["KM生存率"], where="post", label="Observed KM survival")
        axis.fill_between(
            x,
            rows["95%CI下限"].to_numpy(dtype=float),
            rows["95%CI上限"].to_numpy(dtype=float),
            step="post",
            alpha=0.2,
        )
        axis.set(
            title=english_label(name),
            xlabel="Months after landmark",
            ylabel="Survival probability",
            ylim=(0, 1.03),
        )
        axis.legend()
    figure.suptitle("Overall predicted and observed survival (shading: KM 95% CI)")
    save_figure(figure, stem, dpi)


def plot_training(history: pd.DataFrame, stem: Path, dpi: int) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
    panels = [
        ("reconstruction_loss", "Raw reconstruction loss"),
        ("survival_loss", "Raw survival loss"),
        ("vade_kl_loss", "Raw clustering loss"),
        ("loss", "Weighted total loss"),
        ("cindex", "C-index"),
    ]
    for axis, (metric, title) in zip(axes.flat, panels, strict=False):
        for prefix, label in [("", "Training"), ("val_", "Validation")]:
            if prefix + metric in history:
                axis.plot(history["global_epoch"], history[prefix + metric], label=label)
        axis.set(title=title, xlabel="Global epoch")
        axis.legend()
    weight_axis = axes.flat[-1]
    for metric, label in [
        ("reconstruction_loss_weight", "Reconstruction"),
        ("survival_loss_weight", "Survival"),
        ("vade_kl_loss_weight", "Clustering"),
    ]:
        if metric in history:
            weight_axis.plot(history["global_epoch"], history[metric], label=label)
    weight_axis.set(title="Loss weights", xlabel="Global epoch")
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
                label="End of warmup",
            )
        if checkpoint is not None:
            axis.axvline(checkpoint, linestyle=":", color="#009E73", label="Saved checkpoint")
        axis.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    figure.suptitle(
        "Model training history; interpret total loss together with raw components and weights"
    )
    save_figure(figure, stem, dpi)


def plot_k_metrics(
    runs: pd.DataFrame,
    minimum: float,
    stem: Path,
    dpi: int,
    *,
    expected_ks: list[int] | None = None,
    highlight_k: int | None = None,
) -> None:
    """六指标组图；缺失K断线，手动指定K仅作参照，不重新选择模型。"""
    datasets = runs["dataset"].drop_duplicates().tolist()
    metrics = [
        ("selection_score", "Composite score (higher is better)"),
        ("cindex", "C-index"),
        ("latent_mixture_bic", "Latent-mixture BIC"),
        ("cluster_min_fraction", "Minimum cluster fraction"),
        ("cluster_empty_count", "Empty clusters"),
        ("cluster_entropy", "Normalized cluster entropy"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    ks = expected_ks or sorted(runs["n_clusters"].unique().tolist())
    palette = {"train": "#0072B2", "validation": "#D55E00", "test": "#009E73"}
    styles = {"train": "-", "validation": "--", "test": "-."}
    for name in datasets:
        subset = runs.loc[runs["dataset"].eq(name)]
        for axis, (metric, title) in zip(axes.flat, metrics, strict=True):
            if metric not in subset:
                axis.text(
                    0.5,
                    0.5,
                    "Not available in historical run",
                    transform=axis.transAxes,
                    ha="center",
                )
            else:
                for seed, values in subset.groupby("seed"):
                    values = values.set_index("n_clusters").reindex(ks)
                    suffix = f" ({seed})" if runs["seed"].nunique() > 1 else ""
                    axis.plot(
                        ks,
                        values[metric],
                        marker="o",
                        markersize=4,
                        color=palette.get(name),
                        linestyle=styles.get(name, "-"),
                        label=f"{english_label(name)}{suffix}",
                    )
            axis.set(
                title=title,
                xlabel="Number of clusters (K)",
                xticks=ks,
            )
            axis.grid(alpha=0.15)
    axes.flat[3].axhline(
        minimum, color=".5", linestyle=":", linewidth=0.8, label=f"Minimum {minimum:.0%}"
    )
    axes.flat[3].axhline(0.10, color=".65", linestyle="--", linewidth=0.8, label="Preferred 10%")
    for axis in axes.flat:
        if highlight_k is not None:
            axis.axvline(highlight_k, color=".65", linestyle=":", linewidth=0.8)
        axis.legend(fontsize=7)
    missing = sorted(set(ks) - set(runs["n_clusters"]))
    figure.suptitle(
        "Candidate K diagnostics · same configuration / frozen splits\n"
        + (f"Missing K: {missing}; gaps are not interpolated. " if missing else "")
        + (f"Vertical line: manually fixed K={highlight_k}. " if highlight_k else "")
        + "Test-informed comparisons are exploratory",
        fontsize=11,
    )
    save_figure(figure, stem, dpi)


def plot_stability(pairs: pd.DataFrame, selected_k: int, stem: Path, dpi: int) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    summary: pd.DataFrame = pairs.groupby("K").agg(  # type: ignore[reportAssignmentType]
        mean=("ARI", "mean"), min=("ARI", "min"), max=("ARI", "max")
    )
    x = summary.index.to_numpy(dtype=int)
    axes[0].plot(x, summary["mean"], "o-", label="Mean pairwise ARI")
    axes[0].fill_between(
        x,
        summary["min"].to_numpy(),
        summary["max"].to_numpy(),
        alpha=0.2,
        label="Minimum–maximum range",
    )
    invalid = pairs.loc[~pairs["两候选均通过占用门槛"]]
    axes[0].scatter(
        invalid["K"],
        invalid["ARI"],
        color="red",
        marker="x",
        label="Comparison includes degenerate candidates",
    )
    axes[0].set(
        xlabel="Candidate K",
        ylabel="ARI",
        xticks=x,
        ylim=(-1, 1.05),
        title="Cross-seed agreement on frozen validation data",
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
    axes[1].set_title(f"Evaluated K={selected_k}: pairwise ARI")
    figure.suptitle(
        "Stability appendix: interpret ARI together with empty clusters "
        "and minimum cluster fraction"
    )
    save_figure(figure, stem, dpi)

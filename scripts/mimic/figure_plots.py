"""MIMIC综合主图及聚合SourceData；图形直接消费共享报告结果。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.transforms import Bbox
from sksurv.nonparametric import kaplan_meier_estimator

from .config import MimicFigureConfig
from .evaluation import ClusterTrajectoryAnalysis
from .intervention_evaluation import DISPLAY_LABELS
from .intervention_extraction import INTERVENTION_BINARY_COLUMNS
from .reporting import EvaluationReport

COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")
METHOD_LABELS = {
    "trails": "TRAILS",
    "summary_kmeans": "Summary K-means",
    "ufpca_kmeans": "UFPCA K-means",
    "mfpca_kmeans": "MFPCA K-means",
    "cox_risk_kmeans": "Cox-risk K-means",
    "trails_no_survival": "TRAILS (no survival)",
    "cox_ph": "Cox PH",
    "random_survival_forest": "RSF",
    "mpjlcmm": "mpjlcmm",
    "jmbayes2": "JMbayes2",
    "deep_cox_mixtures": "Deep Cox mixtures",
    "vadesc": "VaDeSC",
}
FEATURE_LABELS = {
    "heart_rate": "Heart rate",
    "sbp": "Systolic BP",
    "mbp": "Mean arterial pressure",
    "resp_rate": "Respiratory rate",
    "temperature": "Temperature",
    "spo2": "SpO2",
    "gcs": "GCS",
    "wbc": "WBC",
    "hemoglobin": "Hemoglobin",
    "platelet": "Platelets",
    "aniongap": "Anion gap",
    "bicarbonate": "Bicarbonate",
    "bun": "BUN",
    "creatinine": "Creatinine",
    "glucose": "Glucose",
    "sodium": "Sodium",
    "potassium": "Potassium",
    "bilirubin_total": "Total bilirubin",
    "inr": "INR",
    "lactate": "Lactate",
    "ph": "pH",
    "pao2fio2ratio": "PaO2 / FiO2",
    "uo_mlkghr_6hr": "Urine output",
}
TRAJECTORY_LABELS = {
    "lactate": "Lactate (mmol/L)",
    "mbp": "MAP (mmHg)",
    "creatinine": "Creatinine (mg/dL)",
    "platelet": "Platelets (10³/µL)",
    "pao2fio2ratio": "PaO2 / FiO2 (mmHg)",
    "gcs": "GCS (points)",
}


def configure_plotting() -> None:
    """统一印刷字号、嵌入字体与色盲友好配色。"""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "lines.linewidth": 1.1,
            "axes.prop_cycle": plt.cycler(color=COLORS),
        }
    )


def _title(axis: Axes, letter: str, title: str) -> None:
    axis.set_title(f"{letter}  {title}", loc="left", fontweight="bold", pad=6)


def _missing(axis: Axes, reason: str = "Not estimable") -> None:
    axis.text(0.5, 0.5, reason, ha="center", va="center", transform=axis.transAxes, fontsize=7)


def aggregate_tables(report: EvaluationReport) -> dict[str, pd.DataFrame]:
    """按文件类型汇集聚合表；显式包含方法和数据集，不包含患者记录。"""
    parts: dict[str, list[pd.DataFrame]] = {}
    for method in report.methods:
        for split in report.datasets:
            for path in sorted((report.output / method["key"] / split).glob("*.csv")):
                table = pd.read_csv(path)
                parts.setdefault(path.stem, []).append(
                    table.assign(method=method["key"], dataset=split)
                )
    return {
        "comparison": report.comparison,
        "label_agreement": report.agreement,
        **{name: pd.concat(tables, ignore_index=True) for name, tables in parts.items()},
    }


def save_source_data(output: Path, tables: dict[str, pd.DataFrame]) -> None:
    """一个工作簿汇集聚合数值；索引记录完整名称以免Excel工作表名截断混淆。"""
    index = []
    with pd.ExcelWriter(output / "SourceData.xlsx", engine="openpyxl") as writer:
        for number, (name, table) in enumerate(tables.items(), 1):
            sheet = f"{number:02d}_{name}"[:31]
            table.to_excel(writer, sheet_name=sheet, index=False)
            index.append({"sheet": sheet, "table": name, "rows": len(table)})
        pd.DataFrame(index).to_excel(writer, sheet_name="Index", index=False)


def _km_tables(report: EvaluationReport, tau: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    curves: list[dict[str, Any]] = []
    risks: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        frame = report.primary_frames[split]
        for cluster, group in frame.groupby("pred_cluster", observed=True):
            times, survival = kaplan_meier_estimator(
                group["event"].to_numpy(dtype=bool), group["survival_time"].to_numpy(dtype=float)
            )[:2]
            curves.extend(
                {"dataset": split, "cluster": cluster, "time": t, "survival": s}
                for t, s in zip(np.r_[0.0, times], np.r_[1.0, survival], strict=True)
            )
            risks.extend(
                {
                    "dataset": split,
                    "cluster": cluster,
                    "time": t,
                    "n_at_risk": int((group["survival_time"].to_numpy(dtype=float) >= t).sum()),
                }
                for t in np.linspace(0, tau, 5)
            )
    return pd.DataFrame(curves), pd.DataFrame(risks)


def plot_main_figure(
    report: EvaluationReport, config: MimicFigureConfig
) -> dict[str, pd.DataFrame]:
    """按方法比较→亚型解释组织A–J面板；不以测试性能重排方法或候选。"""
    configure_plotting()
    tables = aggregate_tables(report)
    selected = report.comparison.loc[
        report.comparison["split"].eq(config.comparison_dataset)
    ].reset_index(drop=True)
    labels = [
        f"{i + 1}. {METHOD_LABELS.get(name, name.replace('_', ' '))}"
        for i, name in enumerate(selected["method"])
    ]
    keys = selected["key"].tolist()
    colors = [COLORS[index % len(COLORS)] for index in range(len(selected))]
    frame = report.primary_frames[config.profile_dataset]
    analysis = ClusterTrajectoryAnalysis(
        report.datasets[config.profile_dataset],
        frame["pred_cluster"].to_numpy(dtype=int),
        report.preprocessing,
        config.n_clusters,
        config.trajectory_bin_hours,
        config.min_cell_patients,
    )
    phases = analysis.phase_profile()
    tables["phase_profile"] = phases.assign(dataset=config.profile_dataset)
    phases.to_csv(report.output / "phase_profile.csv", index=False)
    km, at_risk = _km_tables(report, config.tau)
    tables["km_curves"], tables["km_at_risk"] = km, at_risk
    fig = plt.figure(figsize=(config.width_mm / 25.4, config.height_mm / 25.4))
    # 固定物理尺寸；两个解释大区拥有更多空间，图例不覆盖数据。
    grid = fig.add_gridspec(
        6,
        3,
        left=0.145,
        right=0.925,
        top=0.963,
        bottom=0.067,
        height_ratios=[
            1.12,
            1.02,
            2.05,
            1.65,
            1.65 if config.n_clusters > 3 else 1.18,
            2.0 if config.n_clusters > 3 else 1.25,
        ],
        hspace=0.68,
        wspace=0.48,
    )

    # A：C-index、IBS各用独立坐标轴，并标出无能力或不支持的NA。
    axes_a = grid[0, 0].subgridspec(1, 2, wspace=0.35)
    for index, (metric, title) in enumerate(
        (("harrell_cindex", "C-index"), ("integrated_brier_score", "IBS"))
    ):
        axis = fig.add_subplot(axes_a[0, index])
        values = selected[metric].to_numpy(dtype=float)
        axis.barh(np.arange(len(selected)), values, color=colors)
        axis.set_yticks(np.arange(len(selected)), labels if index == 0 else [""] * len(labels))
        axis.invert_yaxis()
        axis.set_xlabel(title)
        axis.set_xlim(0, 1 if index == 0 else max(0.25, np.nanmax(np.r_[0.0, values]) * 1.15))
        for pos, value in enumerate(values):
            if not np.isfinite(value):
                axis.text(0.02, pos, "NA", va="center", transform=axis.get_yaxis_transform())
        _title(axis, "A" if index == 0 else "", title)

    axis = fig.add_subplot(grid[0, 1])
    for i, row in selected.iterrows():
        axis.plot(
            config.auc_times,
            [row[f"auc_{day:g}"] for day in config.auc_times],
            marker=("o", "s", "^", "D")[i % 4],
            color=colors[i],
            label=str(i + 1),
            ls=("-", "--", ":", "-.")[i % 4],
            ms=3,
        )
    axis.set(xlabel="Days after landmark", ylabel="AUC", ylim=(0, 1), xticks=config.auc_times)
    axis.axhline(0.5, color=".7", ls="--", lw=0.7)
    axis.legend(
        title="Method ID (panel A)",
        ncols=4,
        fontsize=5,
        title_fontsize=5,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.45),
        handlelength=1.5,
        columnspacing=0.7,
    )
    _title(axis, "B", "Time-specific discrimination")

    axis = fig.add_subplot(grid[0, 2])
    calibration = tables.get("calibration", pd.DataFrame())
    for i, key in enumerate(keys):
        rows = calibration.loc[
            (calibration["method"] == key)
            & (calibration["dataset"] == config.comparison_dataset)
            & (calibration["time"] == 21)
        ].dropna(subset=["observed_survival_km"])
        axis.plot(
            rows["mean_predicted_survival"],
            rows["observed_survival_km"],
            marker=("o", "s", "^", "D")[i % 4],
            ls=("-", "--", ":", "-.")[i % 4],
            ms=2.5,
            color=colors[i],
        )
    axis.plot([0, 1], [0, 1], color=".65", ls="--", lw=0.7)
    axis.set(xlabel="Predicted survival", ylabel="Observed KM survival", xlim=(0, 1), ylim=(0, 1))
    _title(axis, "C", "Day 21 calibration")

    cluster_methods = selected.dropna(subset=["n_clusters"])
    sections = np.array_split(np.arange(len(cluster_methods)), 2 if len(cluster_methods) > 6 else 1)
    d_grid = grid[1, :2].subgridspec(1, len(sections), wspace=0.15)
    for panel, indices in enumerate(sections):
        axis = fig.add_subplot(d_grid[0, panel])
        cells = [
            [
                f"M{i + 1}" if len(sections) > 1 else labels[int(i)],
                str(int(row["n_clusters"])),
                str(int(row["occupied_clusters"])),
                f"{row['min_cluster_fraction']:.1%}",
            ]
            for i, row in cluster_methods.iloc[indices].iterrows()
        ]
        axis.axis("off")
        table = axis.table(
            cellText=cells,
            colLabels=["Method", "K", "Occupied", "Min. fraction"],
            cellLoc="center",
            colWidths=[0.26, 0.14, 0.25, 0.35] if len(sections) > 1 else [0.43, 0.15, 0.2, 0.22],
            bbox=Bbox.from_bounds(0, -0.05, 1, 0.98),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(6.5)
        for cell in table.get_celld().values():
            cell.set_edgecolor(".85")
            cell.set_linewidth(0.4)
        if panel == 0:
            _title(axis, "D", "Cluster occupancy")

    axis = fig.add_subplot(grid[1, 2])
    cluster_keys = cluster_methods["key"].tolist()
    agreement = pd.DataFrame(np.eye(len(cluster_keys)), index=cluster_keys, columns=cluster_keys)
    for row in report.agreement.loc[
        report.agreement["split"] == config.comparison_dataset
    ].itertuples(index=False):
        if row.left in cluster_keys and row.right in cluster_keys:
            agreement.loc[row.left, row.right] = agreement.loc[row.right, row.left] = row.ari
    image = axis.imshow(agreement, cmap="cividis", vmin=-1, vmax=1, aspect="auto")
    axis.set_xticks(
        range(len(cluster_keys)),
        [f"M{i + 1}" for i in cluster_methods.index],
    )
    axis.set_yticks(range(len(cluster_keys)), [f"M{i + 1}" for i in cluster_methods.index])
    axis.tick_params(axis="both", labelsize=6)
    fig.colorbar(image, ax=axis, fraction=0.045, pad=0.02)
    _title(axis, "E", "ARI: label agreement")

    axis = fig.add_subplot(grid[2, :])
    phase_labels = ["0–6", "6–12", "12–24", "24–48"]
    matrix = phases.pivot(
        index="feature", columns=["phase", "pred_cluster"], values="median"
    ).reindex(report.datasets[config.profile_dataset].feature_names)
    support = phases.pivot(
        index="feature", columns=["phase", "pred_cluster"], values="low_support"
    ).reindex_like(matrix)
    data = matrix.mask(support).to_numpy(dtype=float)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(".85")
    scale = max(0.5, float(np.nanmax(np.abs(np.r_[0.0, data.ravel()]))))
    image = axis.imshow(data, aspect="auto", cmap=cmap, vmin=-scale, vmax=scale)
    axis.set_yticks(
        range(len(matrix)), [FEATURE_LABELS.get(str(name), str(name)) for name in matrix.index]
    )
    axis.set_xticks(
        range(len(matrix.columns)), [f"{phase_labels[p]}\nC{c}" for p, c in matrix.columns]
    )
    for pos in range(config.n_clusters, len(matrix.columns), config.n_clusters):
        axis.axvline(pos - 0.5, color="white", lw=1.3)
    fig.colorbar(image, ax=axis, fraction=0.015, pad=0.01, label="Median standardized value")
    _title(
        axis,
        "F",
        f"All-feature phase profiles ({config.profile_dataset}); "
        f"grey: n < {config.min_cell_patients}",
    )

    trajectories = tables["cluster_trajectories"]
    trajectories = trajectories.loc[
        (trajectories["method"] == "trails") & (trajectories["dataset"] == config.profile_dataset)
    ]
    axes_g = grid[3, :].subgridspec(2, 3, hspace=0.75, wspace=0.4)
    for index, (feature, label) in enumerate(TRAJECTORY_LABELS.items()):
        axis = fig.add_subplot(axes_g[index // 3, index % 3])
        for cluster in range(config.n_clusters):
            rows = trajectories.loc[
                (trajectories["feature"] == feature) & (trajectories["pred_cluster"] == cluster)
            ]
            x, y = (
                rows["time_bin_midpoint"].to_numpy(dtype=float),
                rows["median"].to_numpy(dtype=float),
            )
            enough = rows["n_patients"].ge(config.min_cell_patients).to_numpy()
            color = COLORS[cluster % len(COLORS)]
            axis.plot(
                x,
                np.where(enough, y, np.nan),
                color=color,
                marker="o",
                ms=1.7,
                ls=("-", "--", ":", "-.")[cluster % 4],
                label=f"C{cluster}",
            )
            axis.fill_between(
                x,
                rows["q25"].to_numpy(dtype=float),
                rows["q75"].to_numpy(dtype=float),
                where=enough,
                color=color,
                alpha=0.13,
            )
            axis.scatter(x[~enough], y[~enough], marker="x", color=color, s=10)
        axis.set(
            xlim=(0, 48),
            xticks=[0, 12, 24, 36, 48],
            xlabel="Hours after ICU admission" if index >= 3 else "",
        )
        _title(axis, "G" if index == 0 else "", label)
        if index == 2:
            axis.legend(
                ncols=config.n_clusters,
                loc="lower right",
                bbox_to_anchor=(1, 1.32),
                fontsize=5.5,
                frameon=False,
            )

    for index, split in enumerate(("validation", "test")):
        km_grid = grid[4, index].subgridspec(
            2,
            1,
            height_ratios=[1.2 if config.n_clusters > 3 else 2.5, 1.5],
            hspace=0.1,
        )
        axis = fig.add_subplot(km_grid[0, 0])
        risk_labels = []
        for cluster in range(config.n_clusters):
            rows = km.loc[(km["dataset"] == split) & (km["cluster"] == cluster)]
            axis.step(
                rows["time"],
                rows["survival"],
                where="post",
                color=COLORS[cluster % len(COLORS)],
                ls=("-", "--", ":", "-.")[cluster % 4],
            )
            counts = at_risk.loc[(at_risk["dataset"] == split) & (at_risk["cluster"] == cluster)]
            risk_labels.append(counts["n_at_risk"].tolist() or [0] * 5)
        axis.set(
            xlim=(0, config.tau),
            ylim=(0, 1.02),
            xticks=np.linspace(0, config.tau, 5),
            ylabel="Survival",
        )
        axis.tick_params(axis="x", labelbottom=False)
        _title(axis, "H" if index == 0 else "", f"{split.title()} KM")
        risk_axis = fig.add_subplot(km_grid[1, 0])
        risk_axis.axis("off")
        table = risk_axis.table(
            cellText=risk_labels,
            rowLabels=[f"C{k}" for k in range(config.n_clusters)],
            colLabels=[f"{t:g}d" for t in np.linspace(0, config.tau, 5)],
            cellLoc="center",
            bbox=Bbox.from_bounds(0, 0, 1, 0.95),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(6)
        for cell in table.get_celld().values():
            cell.set_edgecolor("white")
        risk_axis.text(0, -0.19, "Numbers at risk", transform=risk_axis.transAxes, fontsize=6)

    axis = fig.add_subplot(grid[4, 2])
    cox = tables.get("adjusted_cox", pd.DataFrame())
    if not cox.empty:
        cox = cox.loc[(cox["method"] == "trails") & (cox["dataset"] == "test")]
    if cox.empty:
        _missing(axis)
        axis.axis("off")
    else:
        hr = cox["hazard_ratio"].to_numpy(dtype=float)
        axis.errorbar(
            hr,
            np.arange(len(cox)),
            xerr=np.vstack(
                [
                    hr - cox["ci_lower_95"].to_numpy(dtype=float),
                    cox["ci_upper_95"].to_numpy(dtype=float) - hr,
                ]
            ),
            fmt="o",
            color=COLORS[0],
            capsize=2,
        )
        axis.set_yticks(range(len(cox)), [f"C{int(c)}" for c in cox["pred_cluster"]])
        axis.axvline(1, color=".5", ls="--", lw=0.8)
        axis.set_xscale("log")
    axis.set_xlabel("Adjusted HR (95% CI)", fontsize=6)
    _title(axis, "I", "Test adjusted Cox")
    # J：临床特征与治疗暴露并列，保留种族分组和明确分母。
    clinical = tables["clinical_characteristics"]
    clinical = clinical.loc[(clinical["method"] == "trails") & (clinical["dataset"] == "test")]
    group_cols = [f"cluster_{c}" for c in range(config.n_clusters)]
    cells = []
    clinical_rows = [
        ("Age, years", "mean (SD)", "Age, mean (SD)"),
        ("SOFA at sepsis onset", "median [IQR]", "SOFA, median [IQR]"),
        ("Gender", "M", "Male, n (%)"),
        *[
            ("Race", level, label + ", n (%)")
            for level, label in (
                ("WHITE", "White"),
                ("BLACK", "Black"),
                ("ASIAN", "Asian"),
                ("HISPANIC_OR_LATINO", "Hispanic / Latino"),
                ("OTHER_OR_UNKNOWN", "Other / unknown"),
            )
        ],
    ]
    for characteristic, level, label in clinical_rows:
        row = clinical.loc[
            (clinical["characteristic"] == characteristic) & (clinical["level"] == level)
        ]
        values = row[group_cols].iloc[0].tolist()
        if config.n_clusters > 3:
            values = [
                re.sub(r"\.0(?=\D|$)", "", str(value))
                .replace(", ", ",")
                .replace(" (", "\n(")
                .replace(" [", "\n[")
                for value in values
            ]
        cells.append([label, *values])
    treatments = []
    interventions = tables["organ_support_treatment_descriptive"]
    for feature in INTERVENTION_BINARY_COLUMNS:
        values = interventions.loc[
            (interventions["method"] == "trails")
            & (interventions["dataset"] == "test")
            & (interventions["variable"] == feature)
        ].set_index("group")
        treatments.append(
            [
                {
                    "rrt_any": "Renal replacement (RRT)",
                    "crrt_any": "Continuous RRT (CRRT)",
                    "hfnc_any": "High-flow nasal oxygen",
                }.get(feature, DISPLAY_LABELS[feature]),
                *[
                    f"{values.loc[f'cluster_{c}', 'percent']:.0f}%"
                    for c in range(config.n_clusters)
                ],
            ]
        )
    j_grid = grid[5, :].subgridspec(1, 2, wspace=0.12)
    for index, (body, title) in enumerate(
        ((cells, "Test clinical profile"), (treatments, "Test treatment exposure (%)"))
    ):
        axis = fig.add_subplot(j_grid[0, index])
        axis.axis("off")
        table = axis.table(
            cellText=body,
            colLabels=["", *[f"C{c}" for c in range(config.n_clusters)]],
            colWidths=[0.46, *([0.54 / config.n_clusters] * config.n_clusters)],
            cellLoc="center",
            bbox=Bbox.from_bounds(0, 0, 1, 1),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(5.2 if config.n_clusters > 3 else 6)
        for (_r, c), cell in table.get_celld().items():
            cell.set_edgecolor(".85")
            cell.set_linewidth(0.4)
            cell.get_text().set_linespacing(1.0)
            if c == 0:
                cell.set_text_props(ha="left")
        _title(axis, "J" if index == 0 else "", title)
    fig.suptitle(
        f"MIMIC-IV | TRAILS K={config.n_clusters}, seed={config.model_seed}"
        f" | comparison: {config.comparison_dataset}",
        fontsize=10,
        y=0.991,
    )
    footer = (
        "Trajectories: clipped clinical values, patient median / IQR; "
        f"×: n < {config.min_cell_patients}. "
        "NA: no capability or unsupported metric.\n"
        "Cox: age, sex, grouped race, SOFA; reference = lowest train KM mortality."
    )
    fig.text(0.145, 0.024, footer, fontsize=6)
    if config.selection_basis == "test_exploratory":
        fig.text(
            0.145,
            0.012,
            "EXPLORATORY: candidate choice used test metrics.",
            fontsize=6,
            fontweight="bold",
        )
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(report.output / f"main_figure.{suffix}", dpi=config.dpi)
    plt.close(fig)
    return tables

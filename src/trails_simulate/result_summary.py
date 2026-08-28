from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from .config import SummaryApplicationConfig
from .path import resolve_input_path

RUN_ID_PATTERN = re.compile(
    r"^(?P<scenario>[^/]+)/train_(?P<train_size>\d+)_test_(?P<test_size>\d+)/"
    r"k(?P<n_clusters>\d+)/(?P<repeat>\d+)$"
)
SCENARIOLESS_RUN_ID_PATTERN = re.compile(
    r"^train_(?P<train_size>\d+)_test_(?P<test_size>\d+)/"
    r"k(?P<n_clusters>\d+)/(?P<repeat>\d+)$"
)
PARSED_RUN_FIELDS = ("scenario", "train_size", "test_size", "n_clusters", "repeat")
GRID_FIGURE_METRIC_LABELS = {
    "acc": "ACC",
    "ari": "ARI",
    "nmi": "NMI",
    "cindex": "C-index",
}
OKABE_ITO_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
    "#F0E442",
)
PLOT_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h")


@dataclass(frozen=True)
class MetricInput:
    source: str
    root: Path
    label: str
    metrics_csv: Path


def summary_metric_inputs(config: SummaryApplicationConfig) -> list[MetricInput]:
    inputs: list[MetricInput] = []
    train_roots = tuple(resolve_input_path(root) for root in config.train_roots)
    train_labels = source_labels(train_roots, config.train_labels)
    for root, label in zip(train_roots, train_labels, strict=True):
        metrics_csv = root / "train_metrics.csv"
        require_file(metrics_csv)
        inputs.append(MetricInput("train", root, label, metrics_csv))

    baseline_roots = tuple(resolve_input_path(root) for root in config.baseline_roots)
    baseline_labels = source_labels(baseline_roots, config.baseline_labels)
    for root, label in zip(baseline_roots, baseline_labels, strict=True):
        metrics_csv = root / "baseline_metrics.csv"
        require_file(metrics_csv)
        inputs.append(MetricInput("baseline", root, label, metrics_csv))
    return inputs


def source_labels(roots: Sequence[Path], configured_labels: Sequence[str]) -> tuple[str, ...]:
    if configured_labels:
        return tuple(configured_labels)
    return automatic_source_labels(roots)


def automatic_source_labels(roots: Sequence[Path]) -> tuple[str, ...]:
    names = tuple(root.name or str(root) for root in roots)
    if len(set(names)) == len(names):
        return names
    parent_names = tuple(f"{root.parent.name}/{root.name}" for root in roots)
    if len(set(parent_names)) == len(parent_names):
        return parent_names
    return tuple(str(root) for root in roots)


def metric_input_payload(metric_input: MetricInput) -> dict[str, str]:
    return {
        "label": metric_input.label,
        "metrics_csv": str(metric_input.metrics_csv),
        "root": str(metric_input.root),
        "source": metric_input.source,
    }


def require_file(path: Path) -> None:
    if not path.exists():
        raise ValueError(f"Required summary input does not exist: {path}")


def read_metric_df(metric_input: MetricInput) -> pd.DataFrame:
    df = pd.read_csv(metric_input.metrics_csv)
    df["source"] = metric_input.source
    df["source_label"] = metric_input.label
    df["source_root"] = str(metric_input.root)
    return df


def add_run_id_fields(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    result = df.copy()
    warnings: list[str] = []
    parsed_records: list[dict[str, Any]] = []

    for _, row in result.iterrows():
        run_id = normalize_string(row.get("run_id", ""))
        match = RUN_ID_PATTERN.match(run_id)
        scenarioless_match = SCENARIOLESS_RUN_ID_PATTERN.match(run_id)

        # run_id 承载模拟场景、样本量、K 和 repeat，是后续跨重复聚合的主键来源。
        if match is not None:
            parsed_records.append(parsed_run_fields(match, scenario=match.group("scenario")))
        elif scenarioless_match is not None:
            parsed_records.append(
                parsed_run_fields(
                    scenarioless_match,
                    scenario=infer_scenario_from_row(row),
                )
            )
        else:
            parsed_records.append({field: pd.NA for field in PARSED_RUN_FIELDS})
            warnings.append(run_id)

    parsed_df = pd.DataFrame(parsed_records, index=result.index)
    for field in PARSED_RUN_FIELDS:
        result[field] = parsed_df[field]
    return result, warnings


def parsed_run_fields(match: re.Match[str], *, scenario: str) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "train_size": int(match.group("train_size")),
        "test_size": int(match.group("test_size")),
        "n_clusters": int(match.group("n_clusters")),
        "repeat": int(match.group("repeat")),
    }


def normalize_string(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def infer_scenario_from_row(row: pd.Series) -> str:
    data_root = row.get("data_root")
    if isinstance(data_root, str) and data_root:
        parts = Path(data_root).parts
        for index, part in enumerate(parts):
            if part.startswith("train_") and index > 0:
                return parts[index - 1]
    source_label = row.get("source_label")
    if isinstance(source_label, str) and source_label:
        return source_label
    return "unknown"


def add_method_labels(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["method"] = result["method"].astype(str)
    result["source_label"] = result["source_label"].astype(str)
    source_counts = result.groupby("method", dropna=False)["source_root"].transform("nunique")
    result["method_label"] = result["method"]
    duplicate_method_mask = source_counts > 1
    result.loc[duplicate_method_mask, "method_label"] = (
        result.loc[duplicate_method_mask, "method"]
        + " ("
        + result.loc[duplicate_method_mask, "source_label"]
        + ")"
    )
    return result


def available_numeric_metrics(df: pd.DataFrame) -> list[str]:
    excluded = {
        "data_root",
        "method",
        "method_label",
        "n_clusters",
        "prediction_path",
        "repeat",
        "run_id",
        "scenario",
        "source",
        "source_label",
        "source_root",
        "test_size",
        "train_size",
    }
    metrics = []
    for column in df.columns:
        if column in excluded:
            continue
        values = numeric_column(df, str(column))
        if values.notna().any():
            metrics.append(str(column))
    return sorted(metrics)


def group_metric_df(df: pd.DataFrame, *, metrics: Sequence[str]) -> pd.DataFrame:
    group_fields = ["scenario", "train_size", "test_size", "n_clusters", "method_label"]
    working_df = df.copy()
    for metric in metrics:
        working_df[metric] = numeric_column(working_df, metric)

    # 同一实验设置下，不同 repeat 的指标在这里合并，std 使用总体标准差以保持旧结果一致。
    aggregations: dict[str, Any] = {
        "method": pd.NamedAgg(column="method", aggfunc=joined_unique_values),
        "source": pd.NamedAgg(column="source", aggfunc=joined_unique_values),
        "source_label": pd.NamedAgg(column="source_label", aggfunc=joined_unique_values),
    }
    for metric in metrics:
        aggregations[f"{metric}_mean"] = pd.NamedAgg(column=metric, aggfunc="mean")
        aggregations[f"{metric}_std"] = pd.NamedAgg(column=metric, aggfunc=population_std)
        aggregations[f"{metric}_min"] = pd.NamedAgg(column=metric, aggfunc="min")
        aggregations[f"{metric}_max"] = pd.NamedAgg(column=metric, aggfunc="max")
        aggregations[f"{metric}_n"] = pd.NamedAgg(column=metric, aggfunc="count")

    grouped = (
        working_df.groupby(group_fields, dropna=False, sort=True).agg(**aggregations).reset_index()
    )
    return grouped.sort_values(group_fields).reset_index(drop=True)


def joined_unique_values(values: pd.Series) -> str:
    unique_values = sorted({str(value) for value in values.dropna().unique() if str(value) != ""})
    return "|".join(unique_values)


def population_std(values: pd.Series) -> float:
    finite_values = numeric_values(values).dropna().to_numpy(dtype=float)
    if finite_values.size == 0:
        return math.nan
    return float(np.std(finite_values, ddof=0))


def save_summary_figures(
    grouped_df: pd.DataFrame,
    *,
    metrics: Sequence[str],
    figures_dir: Path,
) -> dict[str, str]:
    if grouped_df.empty or not metrics:
        return {}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, str] = {}
    scenarios = sorted(
        str(value) for value in column_series(grouped_df, "scenario").dropna().unique()
    )
    scenario_values = string_column(grouped_df, "scenario")
    for scenario in scenarios:
        scenario_df = cast(pd.DataFrame, grouped_df.loc[scenario_values == scenario]).copy()
        if scenario_df.empty:
            continue
        filename = safe_filename(scenario)
        png_path = figures_dir / f"{filename}_metrics_by_train_size.png"
        pdf_path = figures_dir / f"{filename}_metrics_by_train_size.pdf"
        saved = plot_scenario_metric_grid(
            scenario_df,
            scenario=scenario,
            metrics=metrics,
            png_path=png_path,
            pdf_path=pdf_path,
            plt=plt,
        )
        if saved:
            figures[f"{filename}_metrics_by_train_size_png"] = str(png_path)
            figures[f"{filename}_metrics_by_train_size_pdf"] = str(pdf_path)
    return figures


def safe_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return sanitized or "unknown"


def plot_scenario_metric_grid(
    grouped_df: pd.DataFrame,
    *,
    scenario: str,
    metrics: Sequence[str],
    png_path: Path,
    pdf_path: Path,
    plt: Any,
) -> bool:
    cluster_values = numeric_column(grouped_df, "n_clusters").dropna()
    clusters = sorted({int(value) for value in cluster_values})
    methods = sorted(
        str(value) for value in string_column(grouped_df, "method_label").dropna().unique()
    )
    if not metrics or not clusters or not methods:
        return False

    set_publication_style(plt)
    fig, axes = plt.subplots(
        len(metrics),
        len(clusters),
        figsize=(3.4 * len(clusters), 2.55 * len(metrics)),
        sharex="col",
        squeeze=False,
    )
    plotted = False
    for row_index, metric in enumerate(metrics):
        metric_key = f"{metric}_mean"
        metric_std_key = f"{metric}_std"
        if metric_key not in grouped_df:
            continue
        for col_index, cluster in enumerate(clusters):
            ax = axes[row_index][col_index]
            cluster_mask = numeric_column(grouped_df, "n_clusters") == cluster
            subset = cast(pd.DataFrame, grouped_df.loc[cluster_mask]).copy()
            subset["train_size"] = numeric_column(subset, "train_size")
            subset[metric_key] = numeric_column(subset, metric_key)
            if metric_std_key in subset:
                subset[metric_std_key] = numeric_column(subset, metric_std_key)
            else:
                subset[metric_std_key] = 0.0
            subset = subset.dropna(subset=["train_size", metric_key])
            panel_lowers: list[float] = []
            panel_uppers: list[float] = []
            for method_index, method in enumerate(methods):
                method_mask = string_column(subset, "method_label") == method
                method_df = cast(pd.DataFrame, subset.loc[method_mask]).sort_values("train_size")
                if method_df.empty:
                    continue
                x = [int(value) for value in numeric_column(method_df, "train_size").tolist()]
                y = [float(value) for value in numeric_column(method_df, metric_key).tolist()]
                yerr = [
                    float(value)
                    for value in numeric_column(method_df, metric_std_key).fillna(0.0).tolist()
                ]
                panel_lowers.extend(value - error for value, error in zip(y, yerr, strict=True))
                panel_uppers.extend(value + error for value, error in zip(y, yerr, strict=True))
                ax.errorbar(
                    x,
                    y,
                    yerr=yerr,
                    color=OKABE_ITO_COLORS[method_index % len(OKABE_ITO_COLORS)],
                    marker=PLOT_MARKERS[method_index % len(PLOT_MARKERS)],
                    linewidth=1.8,
                    markersize=4.5,
                    capsize=3,
                    capthick=1.0,
                    elinewidth=1.0,
                    label=method,
                )
                plotted = True
            if row_index == 0:
                ax.set_title(f"K = {cluster}", pad=8)
            if col_index == 0:
                ax.set_ylabel(metric_label(metric))
            if row_index == len(metrics) - 1:
                ax.set_xlabel("Training sample size")
            # 每个子图按当前 metric 的均值和误差线独立缩放，避免不同 K 之间互相挤压。
            apply_independent_metric_limits(ax, metric, panel_lowers, panel_uppers)
            ax.grid(axis="y", alpha=0.22, linewidth=0.7)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="x", rotation=0)

    handles_by_label: dict[str, Any] = {}
    for ax in axes.ravel():
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels, strict=True):
            handles_by_label.setdefault(label, handle)
    if handles_by_label:
        fig.legend(
            list(handles_by_label.values()),
            list(handles_by_label.keys()),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            frameon=False,
            ncol=max(1, min(len(handles_by_label), 5)),
        )
    if plotted:
        fig.suptitle(f"{scenario} simulation", y=1.02, fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(png_path, dpi=320, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return plotted


def set_publication_style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "axes.titlesize": 9,
            "font.size": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )


def metric_label(metric: str) -> str:
    return GRID_FIGURE_METRIC_LABELS.get(metric, metric)


def column_series(df: pd.DataFrame, column: str) -> pd.Series:
    return cast(pd.Series, df[column])


def numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    return numeric_values(column_series(df, column))


def numeric_values(values: pd.Series) -> pd.Series:
    return cast(pd.Series, pd.to_numeric(values, errors="coerce"))


def string_column(df: pd.DataFrame, column: str) -> pd.Series:
    return column_series(df, column).astype(str)


def apply_independent_metric_limits(
    ax: Any,
    metric: str,
    lower_values: Sequence[float],
    upper_values: Sequence[float],
) -> None:
    finite_lowers = [value for value in lower_values if math.isfinite(value)]
    finite_uppers = [value for value in upper_values if math.isfinite(value)]
    if not finite_lowers or not finite_uppers:
        return
    y_min = min(finite_lowers)
    y_max = max(finite_uppers)

    natural_min = -1.0 if metric == "ari" else 0.0
    natural_max = 1.0 if metric in {"acc", "ari", "nmi", "cindex"} else y_max
    span = max(y_max - y_min, 0.04)
    padding = max(span * 0.18, 0.025)
    lower = max(natural_min, y_min - padding)
    upper = min(natural_max, y_max + padding)
    if upper - lower < 0.08:
        center = (upper + lower) / 2
        lower = max(natural_min, center - 0.04)
        upper = min(natural_max, center + 0.04)
    ax.set_ylim(lower, upper)


def write_metric_df(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered_metric_df(df).to_csv(path, index=False)


def ordered_metric_df(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "source",
        "run_id",
        "scenario",
        "train_size",
        "test_size",
        "n_clusters",
        "repeat",
        "method",
        "method_label",
        "source_label",
        "source_root",
        "data_root",
        "prediction_path",
    ]
    ordered = [name for name in preferred if name in df.columns]
    ordered.extend(sorted(set(df.columns) - set(ordered)))
    return df.loc[:, ordered]

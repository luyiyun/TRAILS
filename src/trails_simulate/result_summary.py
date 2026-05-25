from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from trails.artifacts import save_json

from .config import ApplicationConfig
from .path import resolve_path

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


def run_summary_command(
    config: ApplicationConfig,
    hydra_run_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    inputs = summary_metric_inputs(config, project_root)
    if not inputs:
        raise ValueError("command=summary requires at least one train or baseline metrics root.")

    dfs = [read_metric_df(mi) for mi in inputs]
    parse_warnings = add_run_id_fields(rows)
    add_method_labels(rows)
    grouped_rows = group_metric_rows(rows)
    requested_metrics = list(config.summary.metrics)
    available_metrics = available_numeric_metrics(rows)
    skipped_metrics = [metric for metric in requested_metrics if metric not in available_metrics]

    metrics_path = hydra_run_dir / "summary_metrics.csv"
    grouped_path = hydra_run_dir / "summary_metrics_grouped.csv"
    summary_path = hydra_run_dir / "summary_summary.json"
    figures_dir = hydra_run_dir / "figures"
    figures = save_summary_figures(
        grouped_rows,
        metrics=[metric for metric in requested_metrics if metric in available_metrics],
        figures_dir=figures_dir,
    )

    write_csv(metrics_path, rows)
    write_csv(grouped_path, grouped_rows)
    payload = {
        "command": "summary",
        "config": config.model_dump(mode="json"),
        "hydra_run_dir": str(hydra_run_dir),
        "inputs": [metric_input_payload(metric_input) for metric_input in inputs],
        "metrics": {
            "available": available_metrics,
            "requested": requested_metrics,
            "skipped": skipped_metrics,
        },
        "n_groups": len(grouped_rows),
        "n_rows": len(rows),
        "outputs": {
            "figures": figures,
            "grouped_csv": str(grouped_path),
            "metrics_csv": str(metrics_path),
            "summary": str(summary_path),
        },
        "parse_warnings": parse_warnings,
    }
    save_json(summary_path, payload)
    return payload


def summary_metric_inputs(config: ApplicationConfig, project_root: Path) -> list[MetricInput]:
    inputs: list[MetricInput] = []
    train_roots = tuple(resolve_path(root, project_root) for root in config.summary.train_roots)
    train_labels = source_labels(train_roots, config.summary.train_labels)
    for root, label in zip(train_roots, train_labels, strict=True):
        metrics_csv = root / "train_metrics.csv"
        require_file(metrics_csv)
        inputs.append(MetricInput("train", root, label, metrics_csv))

    baseline_roots = tuple(
        resolve_path(root, project_root) for root in config.summary.baseline_roots
    )
    baseline_labels = source_labels(baseline_roots, config.summary.baseline_labels)
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
    df = pd.read_csv(metric_input.metrics_csv, index_col=0)
    df["source"] = metric_input.source
    df["source_label"] = metric_input.label
    df["source_root"] = str(metric_input.root)
    return df


def add_run_id_fields(rows: Sequence[dict[str, Any]]) -> list[str]:
    warnings = []
    for row in rows:
        run_id = str(row.get("run_id", ""))
        match = RUN_ID_PATTERN.match(run_id)
        if match is not None:
            row["scenario"] = match.group("scenario")
            row["train_size"] = int(match.group("train_size"))
            row["test_size"] = int(match.group("test_size"))
            row["n_clusters"] = int(match.group("n_clusters"))
            row["repeat"] = int(match.group("repeat"))
            continue

        scenarioless_match = SCENARIOLESS_RUN_ID_PATTERN.match(run_id)
        if scenarioless_match is not None:
            row["scenario"] = infer_scenario_from_row(row)
            row["train_size"] = int(scenarioless_match.group("train_size"))
            row["test_size"] = int(scenarioless_match.group("test_size"))
            row["n_clusters"] = int(scenarioless_match.group("n_clusters"))
            row["repeat"] = int(scenarioless_match.group("repeat"))
            continue

        for field in PARSED_RUN_FIELDS:
            row[field] = ""
        warnings.append(run_id)
    return warnings


def infer_scenario_from_row(row: Mapping[str, Any]) -> str:
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


def add_method_labels(rows: Sequence[dict[str, Any]]) -> None:
    method_sources: dict[str, set[str]] = {}
    for row in rows:
        method = str(row.get("method", ""))
        source_key = str(row.get("source_root", ""))
        method_sources.setdefault(method, set()).add(source_key)

    for row in rows:
        method = str(row.get("method", ""))
        if len(method_sources.get(method, set())) > 1:
            row["method_label"] = f"{method} ({row.get('source_label', '')})"
        else:
            row["method_label"] = method


def available_numeric_metrics(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    excluded = {
        "n_clusters",
        "repeat",
        "test_size",
        "train_size",
    }
    names = {
        name
        for row in rows
        for name, value in row.items()
        if name not in excluded and isinstance(value, int | float)
    }
    return sorted(names)


def group_metric_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    group_fields = ("scenario", "train_size", "test_size", "n_clusters", "method_label")
    metric_names = available_numeric_metrics(rows)
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in group_fields)
        groups.setdefault(key, []).append(row)

    grouped_rows = []
    for key, group_rows in sorted(
        groups.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        grouped: dict[str, Any] = dict(zip(group_fields, key, strict=True))
        grouped["method"] = joined_unique_values(group_rows, "method")
        grouped["source"] = joined_unique_values(group_rows, "source")
        grouped["source_label"] = joined_unique_values(group_rows, "source_label")
        for metric in metric_names:
            values = [
                float(row[metric])
                for row in group_rows
                if isinstance(row.get(metric), int | float) and math.isfinite(float(row[metric]))
            ]
            if not values:
                continue
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            grouped[f"{metric}_mean"] = mean
            grouped[f"{metric}_std"] = math.sqrt(variance)
            grouped[f"{metric}_min"] = min(values)
            grouped[f"{metric}_max"] = max(values)
            grouped[f"{metric}_n"] = len(values)
        grouped_rows.append(grouped)
    return grouped_rows


def joined_unique_values(rows: Sequence[Mapping[str, Any]], name: str) -> str:
    values = sorted({str(row.get(name, "")) for row in rows if row.get(name, "") != ""})
    return "|".join(values)


def save_summary_figures(
    grouped_rows: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[str],
    figures_dir: Path,
) -> dict[str, str]:
    if not grouped_rows or not metrics:
        return {}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    figures = {}
    scenarios = sorted({str(row.get("scenario", "")) for row in grouped_rows})
    for scenario in scenarios:
        scenario_rows = [row for row in grouped_rows if row.get("scenario") == scenario]
        if not scenario_rows:
            continue
        filename = safe_filename(scenario)
        png_path = figures_dir / f"{filename}_metrics_by_train_size.png"
        pdf_path = figures_dir / f"{filename}_metrics_by_train_size.pdf"
        plot_scenario_metric_grid(
            scenario_rows,
            scenario=scenario,
            metrics=metrics,
            png_path=png_path,
            pdf_path=pdf_path,
            plt=plt,
        )
        figures[f"{filename}_metrics_by_train_size_png"] = str(png_path)
        figures[f"{filename}_metrics_by_train_size_pdf"] = str(pdf_path)
    return figures


def safe_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return sanitized or "unknown"


def plot_scenario_metric_grid(
    grouped_rows: Sequence[Mapping[str, Any]],
    *,
    scenario: str,
    metrics: Sequence[str],
    png_path: Path,
    pdf_path: Path,
    plt: Any,
) -> None:
    clusters = sorted(
        {int(row["n_clusters"]) for row in grouped_rows if isinstance(row.get("n_clusters"), int)}
    )
    methods = sorted({str(row.get("method_label", "")) for row in grouped_rows})
    if not metrics or not clusters or not methods:
        return

    set_publication_style(plt)
    fig, axes = plt.subplots(
        len(metrics),
        len(clusters),
        figsize=(3.4 * len(clusters), 2.55 * len(metrics)),
        sharex="col",
        squeeze=False,
    )
    for row_index, metric in enumerate(metrics):
        metric_key = f"{metric}_mean"
        for col_index, cluster in enumerate(clusters):
            ax = axes[row_index][col_index]
            subset = [
                row
                for row in grouped_rows
                if row.get("n_clusters") == cluster and isinstance(row.get(metric_key), int | float)
            ]
            for method_index, method in enumerate(methods):
                method_rows = sorted(
                    [row for row in subset if row.get("method_label") == method],
                    key=lambda row: int(row["train_size"]),
                )
                if not method_rows:
                    continue
                x = [int(row["train_size"]) for row in method_rows]
                y = [float(row[metric_key]) for row in method_rows]
                yerr = [float(row.get(f"{metric}_std", 0.0)) for row in method_rows]
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
            if row_index == 0:
                ax.set_title(f"K = {cluster}", pad=8)
            if col_index == 0:
                ax.set_ylabel(metric_label(metric))
            if row_index == len(metrics) - 1:
                ax.set_xlabel("Training sample size")
            apply_independent_metric_limits(ax, metric)
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
    fig.suptitle(f"{scenario} simulation", y=1.02, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


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


def apply_independent_metric_limits(ax: Any, metric: str) -> None:
    data_limits = ax.dataLim
    if data_limits.width == float("-inf") or data_limits.height == float("-inf"):
        return
    y_min = float(data_limits.ymin)
    y_max = float(data_limits.ymax)
    if not math.isfinite(y_min) or not math.isfinite(y_max):
        return

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


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = csv_fieldnames(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def csv_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
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
    available = {name for row in rows for name in row}
    ordered = [name for name in preferred if name in available]
    ordered.extend(sorted(available - set(ordered)))
    return ordered

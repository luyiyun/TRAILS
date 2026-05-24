from __future__ import annotations

import csv
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from trails.artifacts import save_json

from .config import ApplicationConfig
from .path import resolve_path

RUN_ID_PATTERN = re.compile(
    r"^(?P<scenario>[^/]+)/train_(?P<train_size>\d+)_test_(?P<test_size>\d+)/"
    r"k(?P<n_clusters>\d+)/(?P<repeat>\d+)$"
)
PARSED_RUN_FIELDS = ("scenario", "train_size", "test_size", "n_clusters", "repeat")


def run_summary_command(
    config: ApplicationConfig,
    hydra_run_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    if config.summary.train_root is None or config.summary.baseline_root is None:
        raise ValueError("command=summary requires summary.train_root and summary.baseline_root.")

    train_root = resolve_path(config.summary.train_root, project_root)
    baseline_root = resolve_path(config.summary.baseline_root, project_root)
    train_csv = train_root / "train_metrics.csv"
    baseline_csv = baseline_root / "baseline_metrics.csv"
    require_file(train_csv)
    require_file(baseline_csv)

    rows = [
        *read_metric_rows(train_csv, source="train"),
        *read_metric_rows(baseline_csv, source="baseline"),
    ]
    parse_warnings = add_run_id_fields(rows)
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
        "inputs": {
            "baseline_metrics": str(baseline_csv),
            "baseline_root": str(baseline_root),
            "train_metrics": str(train_csv),
            "train_root": str(train_root),
        },
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


def require_file(path: Path) -> None:
    if not path.exists():
        raise ValueError(f"Required summary input does not exist: {path}")


def read_metric_rows(path: Path, *, source: str) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            parsed = {name: parse_csv_value(value) for name, value in row.items()}
            parsed["source"] = source
            rows.append(parsed)
    return rows


def parse_csv_value(value: str | None) -> Any:
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except ValueError:
        return value
    if math.isfinite(number):
        return number
    return value


def add_run_id_fields(rows: Sequence[dict[str, Any]]) -> list[str]:
    warnings = []
    for row in rows:
        run_id = str(row.get("run_id", ""))
        match = RUN_ID_PATTERN.match(run_id)
        if match is None:
            for field in PARSED_RUN_FIELDS:
                row[field] = ""
            warnings.append(run_id)
            continue
        row["scenario"] = match.group("scenario")
        row["train_size"] = int(match.group("train_size"))
        row["test_size"] = int(match.group("test_size"))
        row["n_clusters"] = int(match.group("n_clusters"))
        row["repeat"] = int(match.group("repeat"))
    return warnings


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
    group_fields = ("scenario", "train_size", "test_size", "n_clusters", "method")
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
    for metric in metrics:
        metric_path = figures_dir / f"{metric}_by_train_size.png"
        plot_metric_by_train_size(grouped_rows, metric=metric, path=metric_path, plt=plt)
        figures[f"{metric}_by_train_size"] = str(metric_path)

    overview_path = figures_dir / "method_metric_overview.png"
    plot_method_metric_overview(grouped_rows, metrics=metrics, path=overview_path, plt=plt)
    figures["method_metric_overview"] = str(overview_path)
    return figures


def plot_metric_by_train_size(
    grouped_rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    path: Path,
    plt: Any,
) -> None:
    metric_key = f"{metric}_mean"
    rows = [row for row in grouped_rows if isinstance(row.get(metric_key), int | float)]
    scenarios = sorted({str(row.get("scenario", "")) for row in rows})
    clusters = sorted(
        {int(row["n_clusters"]) for row in rows if isinstance(row.get("n_clusters"), int)}
    )
    methods = sorted({str(row.get("method", "")) for row in rows})
    if not scenarios or not clusters or not methods:
        return

    fig, axes = plt.subplots(
        len(scenarios),
        len(clusters),
        figsize=(4.2 * len(clusters), 3.2 * len(scenarios)),
        squeeze=False,
    )
    for row_index, scenario in enumerate(scenarios):
        for col_index, cluster in enumerate(clusters):
            ax = axes[row_index][col_index]
            subset = [
                row
                for row in rows
                if row.get("scenario") == scenario and row.get("n_clusters") == cluster
            ]
            for method in methods:
                method_rows = sorted(
                    [row for row in subset if row.get("method") == method],
                    key=lambda row: int(row["train_size"]),
                )
                if not method_rows:
                    continue
                x = [int(row["train_size"]) for row in method_rows]
                y = [float(row[metric_key]) for row in method_rows]
                yerr = [float(row.get(f"{metric}_std", 0.0)) for row in method_rows]
                ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.5, capsize=3, label=method)
            ax.set_title(f"{scenario} K={cluster}")
            ax.set_xlabel("train size")
            ax.set_ylabel(metric)
            ax.grid(alpha=0.25)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, min(len(labels), 4)))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_method_metric_overview(
    grouped_rows: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[str],
    path: Path,
    plt: Any,
) -> None:
    methods = sorted({str(row.get("method", "")) for row in grouped_rows})
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.6), squeeze=False)
    for index, metric in enumerate(metrics):
        ax = axes[0][index]
        values = [
            overall_method_mean(grouped_rows, method=method, metric=metric) for method in methods
        ]
        ax.bar(methods, values)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def overall_method_mean(
    grouped_rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    metric: str,
) -> float:
    key = f"{metric}_mean"
    values = [
        float(row[key])
        for row in grouped_rows
        if row.get("method") == method and isinstance(row.get(key), int | float)
    ]
    return sum(values) / len(values) if values else float("nan")


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
        "data_root",
        "prediction_path",
    ]
    available = {name for row in rows for name in row}
    ordered = [name for name in preferred if name in available]
    ordered.extend(sorted(available - set(ordered)))
    return ordered

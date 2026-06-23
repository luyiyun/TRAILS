from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def format_summary(command: str, result: Mapping[str, Any]) -> str:
    if command == "simulate":
        return format_simulate_summary(result)
    if command == "train":
        return format_train_summary(result)
    if command == "optim":
        return format_optim_summary(result)
    if command == "baseline":
        return format_baseline_summary(result)
    if command == "summary":
        return format_result_summary(result)
    raise ValueError(f"Unsupported command summary: {command}")


def format_simulate_summary(result: Mapping[str, Any]) -> str:
    runs = [dict(run) for run in result["runs"]]
    outputs = dict(result["outputs"])
    lines = [
        "TRAILS simulate complete",
        f"Run dir: {result['run_dir']}",
        f"Data root: {result['data_root']}",
        f"Runs: {len(runs)}",
        f"Seeds: {format_seed_list([int(run['seed']) for run in runs])}",
        "",
        "Saved summaries:",
        f"  summary: {outputs['summary']}",
        f"  manifest: {outputs['manifest']}",
    ]
    if runs:
        lines.append("")
        lines.append("Generated splits:")
        for run in runs[:20]:
            splits = dict(run["splits"])
            train_split = dict(splits["train"])
            test_split = dict(splits["test"])
            lines.append(
                "  "
                f"{run['run_id']:<30} "
                f"train={train_split['n_patients']} "
                f"test={test_split['n_patients']} "
                f"k={run['n_clusters']} "
                f"seed={run['seed']} "
                f"data_root={run['data_root']}"
            )
        if len(runs) > 20:
            lines.append(f"  ... {len(runs) - 20} more")
    return "\n".join(lines)


def format_train_summary(result: Mapping[str, Any]) -> str:
    outputs = dict(result["outputs"])
    runs = [dict(run) for run in result["runs"]]
    data_source = dict(result.get("data_source", {}))
    data_source_label = str(data_source.get("source", "configured path"))
    lines = [
        "TRAILS train complete",
        f"Run dir: {result['run_dir']}",
        f"Data root: {data_source.get('data_root', 'unknown')}",
        f"Data source: {data_source_label}",
        f"Runs: {len(runs)}",
        "",
        "Saved summaries:",
        f"  summary: {outputs['summary']}",
        f"  metrics csv: {outputs['metrics_csv']}",
    ]
    lines.extend(format_metric_summary_block(dict(result["metrics_summary"])))
    lines.extend(format_run_results_block(runs))
    return "\n".join(lines)


def format_optim_summary(result: Mapping[str, Any]) -> str:
    paths = dict(result["paths"])
    selected_run_ids = [str(run_id) for run_id in result.get("selected_run_ids", [])]
    outputs = dict(result.get("outputs", {}))
    lines = [
        "TRAILS optim complete",
        f"Run dir: {result['run_dir']}",
        f"Splits: {len(selected_run_ids)}",
        f"Trials added: {result['n_trials_requested']}",
        (
            "Completed trials: "
            f"{result.get('completed_before', 0)}->{result.get('completed_after', 0)}"
        ),
        "",
        "Saved summaries:",
        f"  summary: {paths['optim_summary']}",
    ]
    if "trials_csv" in outputs:
        lines.append(f"  trials csv: {outputs['trials_csv']}")
    figures = dict(outputs.get("figures", {}))
    if figures:
        lines.append(f"  pareto figure: {figures.get('pareto_png', 'not written')}")
    if selected_run_ids:
        lines.append("")
        lines.append("Optim splits:")
        for run_id in selected_run_ids[:8]:
            lines.append(f"  {run_id}")
        if len(selected_run_ids) > 8:
            lines.append(f"  ... {len(selected_run_ids) - 8} more")
    return "\n".join(lines)


def format_baseline_summary(result: Mapping[str, Any]) -> str:
    outputs = dict(result["outputs"])
    runs = [dict(run) for run in result["runs"]]
    data_source = dict(result.get("data_source", {}))
    data_source_label = str(data_source.get("source", "configured path"))
    lines = [
        "TRAILS baseline complete",
        f"Run dir: {result['run_dir']}",
        f"Data root: {data_source.get('data_root', 'unknown')}",
        f"Data source: {data_source_label}",
        f"Runs: {len(runs)}",
        "",
        "Saved summaries:",
        f"  summary: {outputs['summary']}",
        f"  metrics csv: {outputs['metrics_csv']}",
    ]
    lines.extend(format_metric_summary_block(dict(result["metrics_summary"])))
    if runs:
        lines.append("")
        lines.append("Baseline results:")
        for run in runs:
            for method in list(run["methods"]):
                method_result = dict(method)
                metrics = dict(method_result["metrics"])
                metric_text = format_inline_metrics(metrics)
                lines.append(f"  {run['run_id']:<30} {method_result['method']:<24} {metric_text}")
    return "\n".join(lines)


def format_result_summary(result: Mapping[str, Any]) -> str:
    outputs = dict(result["outputs"])
    metrics = dict(result["metrics"])
    figures = dict(outputs.get("figures", {}))
    lines = [
        "TRAILS summary complete",
        f"Run dir: {result['run_dir']}",
        f"Rows: {result['n_rows']}",
        f"Groups: {result['n_groups']}",
        f"Available metrics: {', '.join(metrics.get('available', []))}",
        f"Skipped metrics: {', '.join(metrics.get('skipped', [])) or 'none'}",
        "",
        "Saved summaries:",
        f"  summary: {outputs['summary']}",
        f"  metrics csv: {outputs['metrics_csv']}",
        f"  grouped csv: {outputs['grouped_csv']}",
        f"  figures: {len(figures)}",
    ]
    return "\n".join(lines)


def format_metric_summary_block(summary: Mapping[str, Any]) -> list[str]:
    names = ordered_metric_names(summary.keys())
    if not names:
        return []
    lines = ["", "Metric summary:"]
    header = f"  {'metric':<22} {'mean':>10} {'std':>10} {'min':>10} {'max':>10} {'n':>4}"
    lines.append(header)
    for name in names:
        stats = dict(summary[name])
        lines.append(
            "  "
            f"{name:<22} "
            f"{format_float(stats['mean']):>10} "
            f"{format_float(stats['std']):>10} "
            f"{format_float(stats['min']):>10} "
            f"{format_float(stats['max']):>10} "
            f"{int(stats['n']):>4}"
        )
    return lines


def format_run_results_block(runs: Sequence[Mapping[str, Any]]) -> list[str]:
    if not runs:
        return []
    lines = ["", "Run results:"]
    for run in runs:
        metrics = dict(run["metrics"])
        metric_text = format_inline_metrics(metrics)
        lines.append(
            "  "
            f"{run['run_id']} seed={run['seed']} "
            f"{metric_text} "
            f"prediction={run['prediction_path']}"
        )
    return lines


def format_inline_metrics(metrics: Mapping[str, Any]) -> str:
    return ", ".join(
        f"{name}={format_float(metrics[name])}"
        for name in ordered_metric_names(metrics.keys())[:4]
        if isinstance(metrics.get(name), int | float)
    )


def ordered_metric_names(names: Iterable[str]) -> list[str]:
    preferred = [
        "loss",
        "cindex",
        "c_index",
        "acc",
        "ari",
        "nmi",
        "cluster_empty_count",
        "cluster_min_fraction",
        "cluster_max_fraction",
        "cluster_entropy",
        "reconstruction_loss",
        "survival_loss",
        "vade_kl_loss",
    ]
    available = set(names)
    ordered = [name for name in preferred if name in available]
    ordered.extend(sorted(available - set(ordered)))
    return ordered


def format_seed_list(seeds: Sequence[int]) -> str:
    if len(seeds) <= 8:
        return ", ".join(str(seed) for seed in seeds)
    head = ", ".join(str(seed) for seed in seeds[:4])
    tail = ", ".join(str(seed) for seed in seeds[-2:])
    return f"{head}, ..., {tail}"


def format_float(value: Any) -> str:
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return str(number)
    if abs(number) >= 1000 or (0 < abs(number) < 0.001):
        return f"{number:.4e}"
    return f"{number:.4f}"

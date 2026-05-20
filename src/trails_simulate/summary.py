from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def format_run_summary(result: Mapping[str, Any]) -> str:
    command = str(result["command"])
    if command == "simulate":
        return format_simulate_summary(result)
    if command == "train":
        return format_train_summary(result)
    if command == "optim":
        return format_optim_summary(result)
    if command == "baseline":
        return format_baseline_summary(result)
    raise ValueError(f"Unsupported command summary: {command}")


def format_simulate_summary(result: Mapping[str, Any]) -> str:
    repeats = [dict(repeat) for repeat in result["repeats"]]
    outputs = dict(result["outputs"])
    lines = [
        "TRAILS simulate complete",
        f"Hydra run: {result['hydra_run_dir']}",
        f"Data root: {result['data_root']}",
        f"Runs: {len(repeats)}",
        f"Seeds: {format_seed_list([int(repeat['seed']) for repeat in repeats])}",
        "",
        "Saved summaries:",
        f"  summary: {outputs['summary']}",
    ]
    if repeats:
        lines.append("")
        lines.append("Generated splits:")
        for repeat in repeats:
            splits = dict(repeat["splits"])
            train_split = dict(splits["train"])
            test_split = dict(splits["test"])
            lines.append(
                "  "
                f"{repeat['run_id']:<10} "
                f"train={train_split['n_patients']} "
                f"test={test_split['n_patients']} "
                f"seed={repeat['seed']} "
                f"data_root={repeat['data_root']}"
            )
    return "\n".join(lines)


def format_train_summary(result: Mapping[str, Any]) -> str:
    outputs = dict(result["outputs"])
    runs = [dict(run) for run in result["runs"]]
    lines = [
        "TRAILS train complete",
        f"Hydra run: {result['hydra_run_dir']}",
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
    pareto_trials = [dict(trial) for trial in result["pareto_trials"]]
    lines = [
        "TRAILS optim complete",
        f"Hydra run: {result['hydra_run_dir']}",
        f"Optim root: {result['optim_root']}",
        f"Study: {result['study_name']}",
        f"Storage: {result['storage']}",
        "Trials: "
        f"{result['n_completed_before']} -> {result['n_completed_after']} "
        f"(requested {result['n_trials_requested']})",
        "",
        "Saved summaries:",
        f"  summary: {paths['optim_summary']}",
        f"  trials csv: {paths['trials_csv']}",
        f"  pareto: {paths['pareto_trials']}",
    ]
    if pareto_trials:
        lines.append("")
        lines.append("Pareto front:")
        for trial in pareto_trials[:8]:
            values = trial.get("values")
            params = dict(trial.get("params", {}))
            metric_text = format_optim_objectives(values)
            param_text = format_optim_params(params)
            lines.append(f"  trial {trial['number']:<4} {metric_text} {param_text}")
    return "\n".join(lines)


def format_baseline_summary(result: Mapping[str, Any]) -> str:
    outputs = dict(result["outputs"])
    baseline = dict(result["baseline"])
    runs = [dict(run) for run in result["runs"]]
    lines = [
        "TRAILS baseline complete",
        f"Hydra run: {result['hydra_run_dir']}",
        f"Runs: {len(runs)}",
        f"Clusters: {baseline['n_clusters_resolved']}",
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
                lines.append(f"  {run['run_id']:<10} {method_result['method']:<24} {metric_text}")
    return "\n".join(lines)


def format_optim_objectives(values: Any) -> str:
    if not isinstance(values, Sequence) or len(values) < 2:
        return "cindex=NA ari=NA"
    return f"cindex={format_float(values[0])} ari={format_float(values[1])}"


def format_optim_params(params: Mapping[str, Any]) -> str:
    selected = [
        "encoder_input_kind",
        "encoder_mapping_kind",
        "decoder_kind",
        "decoder_conditioning",
        "hidden_dim",
        "latent_dim",
        "learning_rate",
    ]
    chunks = [
        f"{name}={format_optim_param_value(params[name])}" for name in selected if name in params
    ]
    return " ".join(chunks)


def format_optim_param_value(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format_float(value)
    return str(value)


def format_metrics_block(title: str, metrics: Mapping[str, Any]) -> list[str]:
    names = ordered_metric_names(metrics.keys())
    if not names:
        return []
    lines = ["", f"{title}:"]
    for name in names:
        value = metrics[name]
        if isinstance(value, int | float):
            lines.append(f"  {name:<22} {format_float(value)}")
    return lines


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

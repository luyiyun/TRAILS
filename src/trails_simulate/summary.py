from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def format_run_summary(result: Mapping[str, Any]) -> str:
    command = str(result["command"])
    if command == "simulate":
        return format_simulate_summary(result)
    if command == "train":
        return format_train_summary(result)
    if command == "experiment":
        return format_experiment_summary(result)
    if command == "optim":
        return format_optim_summary(result)
    raise ValueError(f"Unsupported command summary: {command}")


def format_simulate_summary(result: Mapping[str, Any]) -> str:
    lines = ["TRAILS simulate complete", f"Hydra run: {result['hydra_run_dir']}"]
    if "splits" in result:
        lines.append(f"Data root: {result['out_dir']}")
        lines.append("")
        lines.append("Splits:")
        splits = dict(result["splits"])
        for name in ("train", "val", "test"):
            if name not in splits:
                continue
            split = dict(splits[name])
            lines.append(
                "  "
                f"{name:<5} patients={split['n_patients']} "
                f"seed={split['seed']} "
                f"censoring={format_float(split['censoring_rate'])} "
                f"path={split['out']}"
            )
    else:
        lines.extend(
            [
                f"Dataset: {result['out']}",
                f"Patients: {result['n_patients']}",
                f"Clusters: {result['clusters']}",
                f"Features: {result['n_features']}",
                f"Seed: {result['seed']}",
                f"Censoring rate: {format_float(result['censoring_rate'])}",
            ]
        )
    return "\n".join(lines)


def format_train_summary(result: Mapping[str, Any]) -> str:
    paths = dict(result["paths"])
    lines = [
        "TRAILS train complete",
        f"Hydra run: {result['hydra_run_dir']}",
        f"Seed: {result['seed']}",
        f"Train data: {paths['data']}",
    ]
    lines.append(f"Test data: {paths['test_data'] or paths['data']}")
    lines.append(f"Artifacts: {result['run_dir'] or 'not saved'}")
    lines.extend(format_metrics_block("Test metrics", dict(result["test"])))
    return "\n".join(lines)


def format_experiment_summary(result: Mapping[str, Any]) -> str:
    run_dir = Path(str(result["hydra_run_dir"]))
    repeats = [dict(repeat) for repeat in result["repeats"]]
    lines = [
        "TRAILS experiment complete",
        f"Hydra run: {run_dir}",
        f"Repeats: {len(repeats)}",
        f"Seeds: {format_seed_list([int(repeat['seed']) for repeat in repeats])}",
        "",
        "Saved summaries:",
        f"  experiment: {run_dir / 'experiment_summary.json'}",
        f"  metrics csv: {run_dir / 'test_metrics.csv'}",
        f"  metrics summary: {run_dir / 'test_metrics_summary.json'}",
    ]
    lines.extend(format_metric_summary_block(dict(result["metrics_summary"])))
    lines.extend(format_repeat_block(repeats))
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
        "survival_weight",
        "cluster_weight",
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


def format_repeat_block(repeats: Sequence[Mapping[str, Any]]) -> list[str]:
    if not repeats:
        return []
    lines = ["", "Repeat results:"]
    for repeat in repeats:
        metrics = dict(repeat["metrics"])
        metric_text = ", ".join(
            f"{name}={format_float(metrics[name])}"
            for name in ordered_metric_names(metrics.keys())[:4]
            if isinstance(metrics.get(name), int | float)
        )
        lines.append(
            "  "
            f"{repeat['repeat']} seed={repeat['seed']} "
            f"{metric_text} "
            f"artifacts={repeat['train_run_dir'] or 'not saved'}"
        )
    return lines


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

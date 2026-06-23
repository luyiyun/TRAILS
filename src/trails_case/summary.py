from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def format_case_summary(result: Mapping[str, Any]) -> str:
    outputs = dict(result["outputs"])
    data = dict(result["data"])
    metrics = dict(result["metrics"])
    lines = [
        "TRAILS case complete",
        f"Run dir: {result['run_dir']}",
        f"Patients: {data['n_patients']}",
        f"Features: {data['n_features']}",
        f"Observations: {data['n_observations']}",
        "",
        "Saved outputs:",
        f"  summary: {outputs['case_summary']}",
        f"  patient clusters: {outputs['patient_clusters']}",
        f"  cluster summary: {outputs['cluster_summary']}",
        f"  feature summary: {outputs['cluster_feature_summary']}",
        f"  model predictions: {outputs['predictions']}",
        f"  dataset: {outputs['dataset']}",
    ]
    metric_text = format_inline_metrics(metrics)
    if metric_text:
        lines.extend(["", f"Metrics: {metric_text}"])
    return "\n".join(lines)


def format_inline_metrics(metrics: Mapping[str, Any]) -> str:
    names = [
        "cindex",
        "acc",
        "ari",
        "nmi",
        "cluster_empty_count",
        "cluster_min_fraction",
        "cluster_max_fraction",
        "cluster_entropy",
    ]
    return ", ".join(
        f"{name}={format_float(metrics[name])}"
        for name in names
        if name in metrics and isinstance(metrics.get(name), int | float)
    )


def format_float(value: Any) -> str:
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return str(number)
    if abs(number) >= 1000 or (0 < abs(number) < 0.001):
        return f"{number:.4e}"
    return f"{number:.4f}"

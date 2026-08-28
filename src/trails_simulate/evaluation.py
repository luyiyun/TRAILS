from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch import Tensor

from trails.data import ClinicalTimeSeriesDataset
from trails.metrics import cluster_accuracy, cluster_assignment_diagnostics, concordance_index


class PredictionPayload(TypedDict):
    sample_index: Tensor
    pred_cluster: Tensor
    risk_score: Tensor
    survival_time: Tensor
    event: Tensor
    true_cluster: NotRequired[Tensor]
    cluster_probabilities: NotRequired[Tensor]


def prediction_payload_from_dataset(
    data: ClinicalTimeSeriesDataset,
    *,
    pred_cluster: Tensor,
    risk_score: Tensor,
    cluster_probabilities: Tensor | None = None,
) -> PredictionPayload:
    payload: PredictionPayload = {
        "sample_index": torch.arange(len(data), dtype=torch.long),
        "pred_cluster": pred_cluster.detach().cpu().long(),
        "risk_score": risk_score.detach().cpu().float(),
        "survival_time": torch.stack([data[index].survival_time for index in range(len(data))])
        .detach()
        .cpu()
        .float(),
        "event": torch.stack([data[index].event for index in range(len(data))])
        .detach()
        .cpu()
        .float(),
    }
    if data.has_cluster_labels:
        labels = []
        for index in range(len(data)):
            label = data[index].cluster_label
            if label is not None:
                labels.append(label)
        if labels:
            payload["true_cluster"] = torch.stack(labels).detach().cpu().long()
    if cluster_probabilities is not None:
        payload["cluster_probabilities"] = cluster_probabilities.detach().cpu().float()
    return payload


def evaluate_predictions(
    payload: PredictionPayload,
    *,
    n_clusters: int,
) -> dict[str, float]:
    pred_cluster = payload["pred_cluster"].detach().cpu().long()
    metrics = {
        "cindex": float(
            concordance_index(
                payload["risk_score"].detach().cpu().float(),
                payload["survival_time"].detach().cpu().float(),
                payload["event"].detach().cpu().float(),
            )
        ),
        **cluster_assignment_diagnostics(pred_cluster, n_clusters=n_clusters),
    }
    true_cluster = payload.get("true_cluster")
    if true_cluster is not None:
        y_true = true_cluster.detach().cpu().numpy()
        y_pred = pred_cluster.numpy()
        metrics["acc"] = cluster_accuracy(pred_cluster, true_cluster)
        metrics["ari"] = float(adjusted_rand_score(y_true, y_pred))
        metrics["nmi"] = float(normalized_mutual_info_score(y_true, y_pred))
    return metrics


def save_prediction_payload(path: Path, payload: PredictionPayload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def save_metrics_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = metric_csv_fieldnames(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def metric_csv_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    preferred = ["run_id", "method", "data_root", "prediction_path"]
    numeric_names = sorted(
        {
            name
            for row in rows
            for name, value in row.items()
            if name not in preferred and isinstance(value, int | float)
        }
    )
    other_names = sorted(
        {
            name
            for row in rows
            for name in row
            if name not in preferred and name not in numeric_names
        }
    )
    return (
        [name for name in preferred if any(name in row for row in rows)]
        + other_names
        + numeric_names
    )


def summarize_metric_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metric_names = sorted(
        {
            name
            for row in rows
            for name, value in row.items()
            if isinstance(value, int | float)
            and name not in {"seed"}
            and math.isfinite(float(value))
        }
    )
    summary: dict[str, Any] = {}
    for name in metric_names:
        values = [
            float(row[name])
            for row in rows
            if name in row
            and isinstance(row[name], int | float)
            and math.isfinite(float(row[name]))
        ]
        if not values:
            continue
        array = np.asarray(values, dtype=float)
        summary[name] = {
            "max": float(array.max()),
            "mean": float(array.mean()),
            "min": float(array.min()),
            "n": len(values),
            "std": float(array.std(ddof=0)),
        }
    return summary


def json_safe_metrics(metrics: Mapping[str, float]) -> dict[str, float | str]:
    payload: dict[str, float | str] = {}
    for name, value in metrics.items():
        number = float(value)
        payload[name] = number if math.isfinite(number) else str(number)
    return payload

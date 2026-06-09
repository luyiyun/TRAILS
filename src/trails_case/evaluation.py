from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch import Tensor

from trails.data import ClinicalTimeSeriesDataset
from trails.metrics import cluster_accuracy, cluster_assignment_diagnostics, concordance_index

from .data import CasePatientSummary


class CasePredictionPayload(TypedDict):
    sample_index: Tensor
    pred_cluster: Tensor
    risk_score: Tensor
    survival_time: Tensor
    event: Tensor
    patient_id: list[str]
    true_cluster: NotRequired[Tensor]
    cluster_probabilities: NotRequired[Tensor]


def prediction_payload_from_case_dataset(
    data: ClinicalTimeSeriesDataset,
    *,
    patient_ids: Sequence[str],
    pred_cluster: Tensor,
    risk_score: Tensor,
    cluster_probabilities: Tensor | None = None,
) -> CasePredictionPayload:
    payload: CasePredictionPayload = {
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
        "patient_id": list(patient_ids),
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


def evaluate_case_predictions(
    payload: CasePredictionPayload,
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


def save_prediction_payload(path: Path, payload: CasePredictionPayload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def save_patient_clusters_csv(
    path: Path,
    *,
    payload: CasePredictionPayload,
    patient_summaries: Sequence[CasePatientSummary],
) -> None:
    probabilities = payload.get("cluster_probabilities")
    probability_columns = (
        []
        if probabilities is None
        else [f"cluster_prob_{index}" for index in range(probabilities.shape[1])]
    )
    fieldnames = [
        "patient_id",
        "sample_index",
        "pred_cluster",
        "risk_score",
        "survival_time",
        "event",
        *probability_columns,
        "n_observations",
        "n_visits",
        "first_time",
        "last_time",
        "missing_fraction",
    ]
    rows = patient_cluster_rows(
        payload=payload,
        patient_summaries=patient_summaries,
        probability_columns=probability_columns,
    )
    write_csv(path, rows, fieldnames=fieldnames)


def patient_cluster_rows(
    *,
    payload: CasePredictionPayload,
    patient_summaries: Sequence[CasePatientSummary],
    probability_columns: Sequence[str],
) -> list[dict[str, Any]]:
    probabilities = payload.get("cluster_probabilities")
    rows: list[dict[str, Any]] = []
    for index, summary in enumerate(patient_summaries):
        row: dict[str, Any] = {
            "patient_id": payload["patient_id"][index],
            "sample_index": int(payload["sample_index"][index].item()),
            "pred_cluster": int(payload["pred_cluster"][index].item()),
            "risk_score": float(payload["risk_score"][index].item()),
            "survival_time": float(payload["survival_time"][index].item()),
            "event": float(payload["event"][index].item()),
            "n_observations": summary.n_observations,
            "n_visits": summary.n_visits,
            "first_time": summary.first_time,
            "last_time": summary.last_time,
            "missing_fraction": summary.missing_fraction,
        }
        if probabilities is not None:
            for column_index, column in enumerate(probability_columns):
                row[column] = float(probabilities[index, column_index].item())
        rows.append(row)
    return rows


def cluster_summary_rows(
    payload: CasePredictionPayload,
    *,
    n_clusters: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pred_cluster = payload["pred_cluster"].detach().cpu().long()
    risk_score = payload["risk_score"].detach().cpu().float()
    survival_time = payload["survival_time"].detach().cpu().float()
    event = payload["event"].detach().cpu().float()
    total = int(pred_cluster.numel())

    for cluster in range(n_clusters):
        mask = pred_cluster == cluster
        indices = torch.nonzero(mask, as_tuple=False).flatten()
        n_patients = int(indices.numel())
        if n_patients == 0:
            rows.append(
                {
                    "pred_cluster": cluster,
                    "n_patients": 0,
                    "fraction": 0.0,
                    "event_count": 0,
                    "event_rate": "",
                    "mean_survival_time": "",
                    "median_survival_time": "",
                    "mean_risk_score": "",
                    "median_risk_score": "",
                }
            )
            continue
        cluster_events = event[indices]
        cluster_survival = survival_time[indices]
        cluster_risk = risk_score[indices]
        rows.append(
            {
                "pred_cluster": cluster,
                "n_patients": n_patients,
                "fraction": n_patients / float(total),
                "event_count": int(cluster_events.sum().item()),
                "event_rate": float(cluster_events.mean().item()),
                "mean_survival_time": float(cluster_survival.mean().item()),
                "median_survival_time": float(cluster_survival.median().item()),
                "mean_risk_score": float(cluster_risk.mean().item()),
                "median_risk_score": float(cluster_risk.median().item()),
            }
        )
    return rows


def cluster_feature_summary_rows(
    data: ClinicalTimeSeriesDataset,
    payload: CasePredictionPayload,
    *,
    n_clusters: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pred_cluster = payload["pred_cluster"].detach().cpu().long()
    for cluster in range(n_clusters):
        sample_indices = [
            index for index in range(len(data)) if int(pred_cluster[index].item()) == cluster
        ]
        for feature_index, feature in enumerate(data.feature_names):
            values: list[float] = []
            times: list[float] = []
            observed_patient_ids: set[str] = set()
            for sample_index in sample_indices:
                sample = data[sample_index].to_aligned()
                observed = sample.mask[:, feature_index] > 0
                if bool(observed.any()):
                    observed_patient_ids.add(str(payload["patient_id"][sample_index]))
                    values.extend(sample.x[observed, feature_index].detach().cpu().float().tolist())
                    times.extend(sample.times[observed].detach().cpu().float().tolist())
            rows.append(
                {
                    "pred_cluster": cluster,
                    "feature": feature,
                    "n_patients": len(sample_indices),
                    "n_patients_observed": len(observed_patient_ids),
                    "n_observations": len(values),
                    "mean_value": mean_or_empty(values),
                    "std_value": std_or_empty(values),
                    "min_value": min(values) if values else "",
                    "max_value": max(values) if values else "",
                    "mean_time": mean_or_empty(times),
                }
            )
    return rows


def save_cluster_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    write_csv(
        path,
        rows,
        fieldnames=[
            "pred_cluster",
            "n_patients",
            "fraction",
            "event_count",
            "event_rate",
            "mean_survival_time",
            "median_survival_time",
            "mean_risk_score",
            "median_risk_score",
        ],
    )


def save_cluster_feature_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    write_csv(
        path,
        rows,
        fieldnames=[
            "pred_cluster",
            "feature",
            "n_patients",
            "n_patients_observed",
            "n_observations",
            "mean_value",
            "std_value",
            "min_value",
            "max_value",
            "mean_time",
        ],
    )


def json_safe_metrics(metrics: Mapping[str, float]) -> dict[str, float | str]:
    payload: dict[str, float | str] = {}
    for name, value in metrics.items():
        number = float(value)
        payload[name] = number if math.isfinite(number) else str(number)
    return payload


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def mean_or_empty(values: Sequence[float]) -> float | str:
    if not values:
        return ""
    return sum(values) / len(values)


def std_or_empty(values: Sequence[float]) -> float | str:
    if len(values) <= 1:
        return ""
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)

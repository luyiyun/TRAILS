from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NotRequired, TypedDict, cast

import numpy as np
import pandas as pd
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


@dataclass(frozen=True)
class CaseResultTables:
    payload: CasePredictionPayload

    def patient_clusters(self, patient_summaries: Sequence[CasePatientSummary]) -> pd.DataFrame:
        if len(patient_summaries) != len(self.payload["patient_id"]):
            raise ValueError("patient_summaries length must match prediction payload length.")

        frame = pd.DataFrame(
            {
                "patient_id": self.payload["patient_id"],
                "sample_index": tensor_array(self.payload["sample_index"], dtype=np.int64),
                "pred_cluster": tensor_array(self.payload["pred_cluster"], dtype=np.int64),
                "risk_score": tensor_array(self.payload["risk_score"], dtype=np.float64),
                "survival_time": tensor_array(self.payload["survival_time"], dtype=np.float64),
                "event": tensor_array(self.payload["event"], dtype=np.float64),
            }
        )
        probabilities = self.payload.get("cluster_probabilities")
        if probabilities is not None:
            probability_values = tensor_array(probabilities, dtype=np.float64)
            for cluster_index in range(probability_values.shape[1]):
                frame[f"cluster_prob_{cluster_index}"] = probability_values[:, cluster_index]

        summary_frame = pd.DataFrame(asdict(summary) for summary in patient_summaries)
        for column in [
            "n_observations",
            "n_visits",
            "first_time",
            "last_time",
            "missing_fraction",
        ]:
            frame[column] = summary_frame[column].to_numpy()
        return cast(pd.DataFrame, frame[patient_cluster_columns(probabilities)])

    def cluster_summary(self, *, n_clusters: int) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "pred_cluster": tensor_array(self.payload["pred_cluster"], dtype=np.int64),
                "risk_score": tensor_array(self.payload["risk_score"], dtype=np.float64),
                "survival_time": tensor_array(self.payload["survival_time"], dtype=np.float64),
                "event": tensor_array(self.payload["event"], dtype=np.float64),
            }
        )
        total = len(frame)
        rows: list[dict[str, Any]] = []
        for cluster in range(n_clusters):
            cluster_frame = frame.loc[frame["pred_cluster"] == cluster]
            n_patients = int(len(cluster_frame))
            if n_patients == 0:
                rows.append(empty_cluster_summary_row(cluster))
                continue
            rows.append(
                {
                    "pred_cluster": cluster,
                    "n_patients": n_patients,
                    "fraction": n_patients / float(total),
                    "event_count": int(cluster_frame["event"].sum()),
                    "event_rate": float(cluster_frame["event"].mean()),
                    "mean_survival_time": float(cluster_frame["survival_time"].mean()),
                    "median_survival_time": float(cluster_frame["survival_time"].median()),
                    "mean_risk_score": float(cluster_frame["risk_score"].mean()),
                    "median_risk_score": float(cluster_frame["risk_score"].median()),
                }
            )
        return pd.DataFrame(rows, columns=CLUSTER_SUMMARY_COLUMNS)

    def cluster_feature_summary(
        self,
        data: ClinicalTimeSeriesDataset,
        *,
        n_clusters: int,
    ) -> pd.DataFrame:
        observation_frame = self.observed_feature_frame(data)
        cluster_index = pd.Index(range(n_clusters), name="pred_cluster")
        feature_index = pd.Index(data.feature_names, name="feature")
        full_index = pd.MultiIndex.from_product([cluster_index, feature_index])
        n_patients = (
            pd.Series(tensor_array(self.payload["pred_cluster"], dtype=np.int64))
            .value_counts()
            .reindex(range(n_clusters), fill_value=0)
        )

        if observation_frame.empty:
            summary = pd.DataFrame(
                index=full_index,
                columns=[
                    "n_patients_observed",
                    "n_observations",
                    "mean_value",
                    "std_value",
                    "min_value",
                    "max_value",
                    "mean_time",
                ],
            )
        else:
            summary = observation_frame.groupby(["pred_cluster", "feature"]).agg(
                n_patients_observed=("patient_id", "nunique"),
                n_observations=("value", "size"),
                mean_value=("value", "mean"),
                std_value=("value", "std"),
                min_value=("value", "min"),
                max_value=("value", "max"),
                mean_time=("time", "mean"),
            )
            summary = summary.reindex(full_index)

        result = summary.reset_index()
        cluster_values = np.asarray(result["pred_cluster"].to_numpy(), dtype=np.int64)
        result["n_patients"] = [int(n_patients.loc[int(cluster)]) for cluster in cluster_values]
        result["n_patients_observed"] = result["n_patients_observed"].fillna(0).astype(int)
        result["n_observations"] = result["n_observations"].fillna(0).astype(int)
        result = result[
            [
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
            ]
        ]
        return cast(pd.DataFrame, result.replace({np.nan: ""}))

    def observed_feature_frame(self, data: ClinicalTimeSeriesDataset) -> pd.DataFrame:
        pred_cluster = tensor_array(self.payload["pred_cluster"], dtype=np.int64)
        rows: list[dict[str, Any]] = []
        for sample_index in range(len(data)):
            sample = data[sample_index].to_aligned()
            for feature_index, feature in enumerate(data.feature_names):
                observed = sample.mask[:, feature_index] > 0
                if not bool(observed.any()):
                    continue
                values = sample.x[observed, feature_index].detach().cpu().numpy()
                times = sample.times[observed].detach().cpu().numpy()
                rows.extend(
                    {
                        "pred_cluster": int(pred_cluster[sample_index]),
                        "feature": feature,
                        "patient_id": self.payload["patient_id"][sample_index],
                        "value": float(value),
                        "time": float(time),
                    }
                    for value, time in zip(values, times, strict=True)
                )
        return pd.DataFrame(
            rows,
            columns=["pred_cluster", "feature", "patient_id", "value", "time"],
        )

    def save_patient_clusters_csv(
        self,
        path: Path,
        *,
        patient_summaries: Sequence[CasePatientSummary],
    ) -> None:
        write_dataframe_csv(path, self.patient_clusters(patient_summaries))

    def save_cluster_summary_csv(self, path: Path, *, n_clusters: int) -> None:
        write_dataframe_csv(path, self.cluster_summary(n_clusters=n_clusters))

    def save_cluster_feature_summary_csv(
        self,
        path: Path,
        data: ClinicalTimeSeriesDataset,
        *,
        n_clusters: int,
    ) -> None:
        write_dataframe_csv(path, self.cluster_feature_summary(data, n_clusters=n_clusters))


def save_patient_clusters_csv(
    path: Path,
    *,
    payload: CasePredictionPayload,
    patient_summaries: Sequence[CasePatientSummary],
) -> None:
    CaseResultTables(payload).save_patient_clusters_csv(
        path,
        patient_summaries=patient_summaries,
    )


def patient_cluster_rows(
    *,
    payload: CasePredictionPayload,
    patient_summaries: Sequence[CasePatientSummary],
    probability_columns: Sequence[str],
) -> list[dict[str, Any]]:
    frame = CaseResultTables(payload).patient_clusters(patient_summaries)
    columns = [*patient_cluster_base_columns(), *probability_columns, *patient_summary_columns()]
    return dataframe_records(cast(pd.DataFrame, frame[columns]))


def cluster_summary_rows(
    payload: CasePredictionPayload,
    *,
    n_clusters: int,
) -> list[dict[str, Any]]:
    return dataframe_records(CaseResultTables(payload).cluster_summary(n_clusters=n_clusters))


def cluster_feature_summary_rows(
    data: ClinicalTimeSeriesDataset,
    payload: CasePredictionPayload,
    *,
    n_clusters: int,
) -> list[dict[str, Any]]:
    frame = CaseResultTables(payload).cluster_feature_summary(data, n_clusters=n_clusters)
    return dataframe_records(frame)


def save_cluster_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    write_csv(
        path,
        rows,
        fieldnames=CLUSTER_SUMMARY_COLUMNS,
    )


def save_cluster_feature_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    write_csv(path, rows, fieldnames=CLUSTER_FEATURE_SUMMARY_COLUMNS)


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
    frame = pd.DataFrame(rows, columns=list(fieldnames))
    write_dataframe_csv(path, frame)


def write_dataframe_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], frame.to_dict(orient="records"))


def tensor_array(tensor: Tensor, *, dtype: Any) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(dtype, copy=False)


def patient_cluster_base_columns() -> list[str]:
    return ["patient_id", "sample_index", "pred_cluster", "risk_score", "survival_time", "event"]


def patient_summary_columns() -> list[str]:
    return ["n_observations", "n_visits", "first_time", "last_time", "missing_fraction"]


def patient_cluster_columns(probabilities: Tensor | None) -> list[str]:
    probability_columns = (
        []
        if probabilities is None
        else [f"cluster_prob_{index}" for index in range(probabilities.shape[1])]
    )
    return [*patient_cluster_base_columns(), *probability_columns, *patient_summary_columns()]


def empty_cluster_summary_row(cluster: int) -> dict[str, Any]:
    return {
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


CLUSTER_SUMMARY_COLUMNS = [
    "pred_cluster",
    "n_patients",
    "fraction",
    "event_count",
    "event_rate",
    "mean_survival_time",
    "median_survival_time",
    "mean_risk_score",
    "median_risk_score",
]

CLUSTER_FEATURE_SUMMARY_COLUMNS = [
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
]

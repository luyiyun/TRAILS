from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from trails.data import ClinicalTimeSeriesDataset


@dataclass(frozen=True)
class CasePatientSummary:
    patient_id: str
    sample_index: int
    n_observations: int
    n_visits: int
    first_time: float
    last_time: float
    missing_fraction: float


def case_dataset_summary(dataset: ClinicalTimeSeriesDataset) -> dict[str, Any]:
    patient_summaries = patient_summaries_from_metadata(dataset.metadata)
    event_count = sum(float(sample.event) for sample in dataset)
    feature_observation_counts = {
        feature: int(
            sum(float(sample.to_aligned().mask[:, index].sum()) for sample in dataset.samples)
        )
        for index, feature in enumerate(dataset.feature_names)
    }
    return {
        "censoring_rate": 1.0 - event_count / len(dataset),
        "description": dataset.description,
        "event_rate": event_count / len(dataset),
        "feature_observation_counts": feature_observation_counts,
        "features": dataset.feature_names,
        "has_cluster_labels": dataset.has_cluster_labels,
        "n_features": dataset.n_features,
        "n_observations": int(sum(summary.n_observations for summary in patient_summaries)),
        "n_patients": len(dataset),
        "patient_summaries": [asdict(summary) for summary in patient_summaries],
        "source": {
            "observations_csv": dataset.metadata.get("observations_csv"),
            "patients_csv": dataset.metadata.get("patients_csv"),
        },
    }


def patient_summaries_from_metadata(metadata: Mapping[str, Any]) -> list[CasePatientSummary]:
    raw_summaries = metadata.get("patient_summaries", [])
    if not isinstance(raw_summaries, Sequence) or isinstance(raw_summaries, str | bytes):
        raise ValueError("dataset metadata patient_summaries must be a sequence.")

    summaries: list[CasePatientSummary] = []
    for raw_summary in raw_summaries:
        if not isinstance(raw_summary, Mapping):
            raise ValueError("dataset metadata patient_summaries entries must be mappings.")
        summaries.append(
            CasePatientSummary(
                patient_id=str(raw_summary["patient_id"]),
                sample_index=int(raw_summary["sample_index"]),
                n_observations=int(raw_summary["n_observations"]),
                n_visits=int(raw_summary["n_visits"]),
                first_time=float(raw_summary["first_time"]),
                last_time=float(raw_summary["last_time"]),
                missing_fraction=float(raw_summary["missing_fraction"]),
            )
        )
    return summaries

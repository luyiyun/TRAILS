from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from trails.data import ClinicalTimeSeriesDataset, compute_delta_time, make_clinical_sample

from .config import CaseColumnsConfig


@dataclass(frozen=True)
class CasePatientRecord:
    patient_id: str
    survival_time: float
    event: float
    cluster_label: int | None


@dataclass(frozen=True)
class CaseObservationRecord:
    patient_id: str
    time: float
    feature: str
    value: float


@dataclass(frozen=True)
class CasePatientSummary:
    patient_id: str
    sample_index: int
    n_observations: int
    n_visits: int
    first_time: float
    last_time: float
    missing_fraction: float


@dataclass(frozen=True)
class ImportedCaseDataset:
    dataset: ClinicalTimeSeriesDataset
    patient_summaries: list[CasePatientSummary]


def load_case_dataset_from_csv(
    *,
    patients_csv: Path,
    observations_csv: Path,
    columns: CaseColumnsConfig,
    description: str,
    feature_order: Sequence[str],
) -> ImportedCaseDataset:
    patients = read_patient_records(patients_csv, columns=columns)
    observations, feature_names = read_observation_records(
        observations_csv,
        columns=columns,
        patient_ids=[patient.patient_id for patient in patients],
        feature_order=feature_order,
    )
    return build_case_dataset(
        patients=patients,
        observations=observations,
        feature_names=feature_names,
        patients_csv=patients_csv,
        observations_csv=observations_csv,
        columns=columns,
        description=description,
    )


def read_patient_records(path: Path, *, columns: CaseColumnsConfig) -> list[CasePatientRecord]:
    patient_columns = columns.patients
    required = (
        patient_columns.patient_id,
        patient_columns.survival_time,
        patient_columns.event,
    )
    records: list[CasePatientRecord] = []
    seen_ids: set[str] = set()
    cluster_label_presence: list[bool] = []

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        validate_required_columns(reader.fieldnames, required=required, path=path)
        has_cluster_label_column = patient_columns.cluster_label in (reader.fieldnames or [])
        for row_number, row in enumerate(reader, start=2):
            patient_id = required_text(row, patient_columns.patient_id, path, row_number)
            if patient_id in seen_ids:
                raise ValueError(f"{path}:{row_number} duplicate patient_id: {patient_id}")
            seen_ids.add(patient_id)
            cluster_label = parse_optional_cluster_label(
                row,
                patient_columns.cluster_label,
                path,
                row_number,
                enabled=has_cluster_label_column,
            )
            cluster_label_presence.append(cluster_label is not None)
            records.append(
                CasePatientRecord(
                    patient_id=patient_id,
                    survival_time=parse_positive_float(
                        row,
                        patient_columns.survival_time,
                        path,
                        row_number,
                    ),
                    event=parse_event(row, patient_columns.event, path, row_number),
                    cluster_label=cluster_label,
                )
            )

    if not records:
        raise ValueError(f"{path} must contain at least one patient row.")
    if any(cluster_label_presence) and not all(cluster_label_presence):
        raise ValueError(
            f"{path} column {patient_columns.cluster_label!r} must be provided for every "
            "patient or omitted for all patients."
        )
    return records


def read_observation_records(
    path: Path,
    *,
    columns: CaseColumnsConfig,
    patient_ids: Sequence[str],
    feature_order: Sequence[str],
) -> tuple[dict[str, list[CaseObservationRecord]], list[str]]:
    observation_columns = columns.observations
    required = (
        observation_columns.patient_id,
        observation_columns.time,
        observation_columns.feature,
        observation_columns.value,
    )
    known_patients = set(patient_ids)
    observations: dict[str, list[CaseObservationRecord]] = {
        patient_id: [] for patient_id in patient_ids
    }
    configured_features = list(feature_order)
    validate_unique_names(configured_features, label="case.feature_order")
    feature_seen: list[str] = []
    feature_seen_set: set[str] = set()
    duplicate_keys: set[tuple[str, float, str]] = set()

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        validate_required_columns(reader.fieldnames, required=required, path=path)
        for row_number, row in enumerate(reader, start=2):
            patient_id = required_text(row, observation_columns.patient_id, path, row_number)
            if patient_id not in known_patients:
                raise ValueError(f"{path}:{row_number} unknown patient_id: {patient_id}")
            feature = required_text(row, observation_columns.feature, path, row_number)
            if configured_features and feature not in configured_features:
                raise ValueError(
                    f"{path}:{row_number} feature {feature!r} is not listed in case.feature_order."
                )
            if feature not in feature_seen_set:
                feature_seen.append(feature)
                feature_seen_set.add(feature)
            time = parse_float(row, observation_columns.time, path, row_number)
            value = parse_float(row, observation_columns.value, path, row_number)
            key = (patient_id, time, feature)
            if key in duplicate_keys:
                raise ValueError(
                    f"{path}:{row_number} duplicate observation for patient_id={patient_id!r}, "
                    f"time={time:g}, feature={feature!r}."
                )
            duplicate_keys.add(key)
            observations[patient_id].append(
                CaseObservationRecord(
                    patient_id=patient_id,
                    time=time,
                    feature=feature,
                    value=value,
                )
            )

    if configured_features:
        missing_features = sorted(set(configured_features) - feature_seen_set)
        if missing_features:
            raise ValueError(
                "case.feature_order includes feature(s) with no observations: "
                f"{', '.join(missing_features)}."
            )
        feature_names = configured_features
    else:
        feature_names = feature_seen

    if not feature_names:
        raise ValueError(f"{path} must contain at least one observed feature.")

    empty_patients = [patient_id for patient_id in patient_ids if not observations[patient_id]]
    if empty_patients:
        preview = ", ".join(empty_patients[:5])
        suffix = "" if len(empty_patients) <= 5 else f", ... ({len(empty_patients)} total)"
        raise ValueError(
            f"Every patient must have at least one observation; missing: {preview}{suffix}"
        )
    return observations, feature_names


def build_case_dataset(
    *,
    patients: Sequence[CasePatientRecord],
    observations: Mapping[str, Sequence[CaseObservationRecord]],
    feature_names: list[str],
    patients_csv: Path,
    observations_csv: Path,
    columns: CaseColumnsConfig,
    description: str,
) -> ImportedCaseDataset:
    feature_to_index = {feature: index for index, feature in enumerate(feature_names)}
    samples = []
    patient_summaries: list[CasePatientSummary] = []

    for sample_index, patient in enumerate(patients):
        patient_observations = list(observations[patient.patient_id])
        times = sorted({observation.time for observation in patient_observations})
        time_to_index = {time: index for index, time in enumerate(times)}
        x = torch.zeros((len(times), len(feature_names)), dtype=torch.float32)
        mask = torch.zeros_like(x)

        # 将真实队列的 observation-level 长表还原为每位患者自己的 aligned 时间轴。
        for observation in patient_observations:
            visit_index = time_to_index[observation.time]
            feature_index = feature_to_index[observation.feature]
            x[visit_index, feature_index] = observation.value
            mask[visit_index, feature_index] = 1.0

        times_tensor = torch.as_tensor(times, dtype=torch.float32)
        samples.append(
            make_clinical_sample(
                times=times_tensor,
                x=x,
                mask=mask,
                delta_time=compute_delta_time(times_tensor, mask),
                survival_time=patient.survival_time,
                event=patient.event,
                cluster_label=patient.cluster_label,
            )
        )
        total_slots = len(times) * len(feature_names)
        patient_summaries.append(
            CasePatientSummary(
                patient_id=patient.patient_id,
                sample_index=sample_index,
                n_observations=len(patient_observations),
                n_visits=len(times),
                first_time=float(times[0]),
                last_time=float(times[-1]),
                missing_fraction=1.0 - (len(patient_observations) / float(total_slots)),
            )
        )

    dataset = ClinicalTimeSeriesDataset(
        samples,
        feature_names=feature_names,
        description=description,
        metadata={
            "case_columns": columns.model_dump(mode="json"),
            "feature_names": feature_names,
            "has_cluster_labels": all(patient.cluster_label is not None for patient in patients),
            "n_features": len(feature_names),
            "n_observations": sum(summary.n_observations for summary in patient_summaries),
            "n_patients": len(patients),
            "observations_csv": str(observations_csv),
            "patient_ids": [patient.patient_id for patient in patients],
            "patient_summaries": [asdict(summary) for summary in patient_summaries],
            "patients_csv": str(patients_csv),
            "source": "case_csv",
        },
    )
    return ImportedCaseDataset(dataset=dataset, patient_summaries=patient_summaries)


def case_dataset_summary(imported: ImportedCaseDataset) -> dict[str, Any]:
    dataset = imported.dataset
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
        "n_observations": int(
            sum(summary.n_observations for summary in imported.patient_summaries)
        ),
        "n_patients": len(dataset),
        "patient_summaries": [asdict(summary) for summary in imported.patient_summaries],
        "source": {
            "observations_csv": dataset.metadata.get("observations_csv"),
            "patients_csv": dataset.metadata.get("patients_csv"),
        },
    }


def validate_required_columns(
    fieldnames: Sequence[str] | None,
    *,
    required: Sequence[str],
    path: Path,
) -> None:
    available = set(fieldnames or ())
    missing = [name for name in required if name not in available]
    if missing:
        raise ValueError(f"{path} missing required column(s): {', '.join(missing)}.")


def validate_unique_names(values: Sequence[str], *, label: str) -> None:
    seen: set[str] = set()
    duplicates = sorted({value for value in values if value in seen or seen.add(value)})
    if duplicates:
        raise ValueError(f"{label} cannot contain duplicates: {', '.join(duplicates)}.")


def required_text(
    row: Mapping[str, str | None],
    column: str,
    path: Path,
    row_number: int,
) -> str:
    value = row.get(column)
    text = "" if value is None else value.strip()
    if text == "":
        raise ValueError(f"{path}:{row_number} column {column!r} cannot be empty.")
    return text


def parse_optional_cluster_label(
    row: Mapping[str, str | None],
    column: str,
    path: Path,
    row_number: int,
    *,
    enabled: bool,
) -> int | None:
    if not enabled:
        return None
    value = row.get(column)
    text = "" if value is None else value.strip()
    if text == "":
        return None
    try:
        return int(text)
    except ValueError as error:
        raise ValueError(f"{path}:{row_number} column {column!r} must be an integer.") from error


def parse_event(
    row: Mapping[str, str | None],
    column: str,
    path: Path,
    row_number: int,
) -> float:
    value = parse_float(row, column, path, row_number)
    if value not in {0.0, 1.0}:
        raise ValueError(f"{path}:{row_number} column {column!r} must be 0 or 1.")
    return value


def parse_positive_float(
    row: Mapping[str, str | None],
    column: str,
    path: Path,
    row_number: int,
) -> float:
    value = parse_float(row, column, path, row_number)
    if value <= 0.0:
        raise ValueError(f"{path}:{row_number} column {column!r} must be positive.")
    return value


def parse_float(
    row: Mapping[str, str | None],
    column: str,
    path: Path,
    row_number: int,
) -> float:
    text = required_text(row, column, path, row_number)
    try:
        value = float(text)
    except ValueError as error:
        raise ValueError(f"{path}:{row_number} column {column!r} must be numeric.") from error
    if not math.isfinite(value):
        raise ValueError(f"{path}:{row_number} column {column!r} must be finite.")
    return value

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .config import DataConfig, TrainerConfig

Batch = dict[str, Tensor]


@dataclass(frozen=True)
class ClinicalSample:
    times: Tensor
    x: Tensor
    mask: Tensor
    delta_time: Tensor
    survival_time: Tensor
    event: Tensor
    cluster_label: Tensor

    def __post_init__(self) -> None:
        validate_clinical_sample(self)


class ClinicalTimeSeriesDataset(Dataset[ClinicalSample]):
    def __init__(
        self,
        samples: list[ClinicalSample],
        *,
        feature_names: list[str],
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not samples:
            raise ValueError("ClinicalTimeSeriesDataset requires at least one sample.")
        if len(feature_names) != int(samples[0].x.shape[-1]):
            raise ValueError("feature_names length must match sample feature dimension.")
        for sample in samples:
            if int(sample.x.shape[-1]) != len(feature_names):
                raise ValueError("All samples must share the same feature dimension.")
        self.samples = samples
        self.feature_names = feature_names
        self.description = description
        self.metadata = metadata or {}
        self.feature_means = compute_feature_means(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ClinicalSample:
        return self.samples[index]

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def save(self, path: str | Path):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "description": self.description,
            "feature_names": self.feature_names,
            "metadata": self.metadata,
            "samples": [
                {
                    "times": sample.times,
                    "x": sample.x,
                    "mask": sample.mask,
                    "delta_time": sample.delta_time,
                    "survival_time": sample.survival_time,
                    "event": sample.event,
                    "cluster_label": sample.cluster_label,
                }
                for sample in self.samples
            ],
        }
        torch.save(payload, destination)

    @classmethod
    def load(cls, path: str | Path) -> ClinicalTimeSeriesDataset:
        payload: dict[str, Any] = torch.load(Path(path), map_location="cpu", weights_only=False)
        samples = [
            make_clinical_sample(
                times=sample["times"],
                x=sample["x"],
                mask=sample["mask"],
                delta_time=sample["delta_time"],
                survival_time=sample["survival_time"],
                event=sample["event"],
                cluster_label=sample["cluster_label"],
            )
            for sample in payload["samples"]
        ]
        return ClinicalTimeSeriesDataset(
            samples,
            feature_names=list(payload["feature_names"]),
            description=str(payload.get("description", "")),
            metadata=dict(payload.get("metadata", {})),
        )


def make_clinical_sample(
    *,
    times: Tensor,
    x: Tensor,
    mask: Tensor,
    delta_time: Tensor,
    survival_time: float | Tensor,
    event: float | Tensor,
    cluster_label: int | Tensor,
) -> ClinicalSample:
    return ClinicalSample(
        times=times.float(),
        x=x.float(),
        mask=mask.float(),
        delta_time=delta_time.float(),
        survival_time=torch.as_tensor(survival_time, dtype=torch.float32),
        event=torch.as_tensor(event, dtype=torch.float32),
        cluster_label=torch.as_tensor(cluster_label, dtype=torch.long),
    )


def validate_clinical_sample(sample: ClinicalSample) -> None:
    if sample.times.ndim != 1:
        raise ValueError("times must have shape (n_visits,).")
    if sample.x.ndim != 2:
        raise ValueError("x must have shape (n_visits, n_features).")
    if sample.mask.shape != sample.x.shape:
        raise ValueError("mask must have the same shape as x.")
    if sample.delta_time.shape != sample.x.shape:
        raise ValueError("delta_time must have the same shape as x.")
    if sample.times.shape[0] != sample.x.shape[0]:
        raise ValueError("times length must match x visit dimension.")
    if torch.any(sample.delta_time < 0):
        raise ValueError("delta_time values must be non-negative.")
    if torch.any((sample.mask < 0) | (sample.mask > 1)):
        raise ValueError("mask values must be in [0, 1].")
    if float(sample.survival_time) <= 0:
        raise ValueError("survival_time must be positive.")
    if float(sample.event) < 0 or float(sample.event) > 1:
        raise ValueError("event must be in [0, 1].")


def compute_feature_means(samples: list[ClinicalSample]) -> Tensor:
    n_features = int(samples[0].x.shape[-1])
    numerator = torch.zeros(n_features, dtype=torch.float32)
    denominator = torch.zeros(n_features, dtype=torch.float32)
    for sample in samples:
        numerator += (sample.x * sample.mask).sum(dim=0)
        denominator += sample.mask.sum(dim=0)
    return numerator / denominator.clamp_min(1.0)


def clinical_collate_fn(samples: list[ClinicalSample]) -> Batch:
    if not samples:
        raise ValueError("clinical_collate_fn requires at least one sample.")
    batch_size = len(samples)
    max_length = max(int(sample.times.shape[0]) for sample in samples)
    n_features = int(samples[0].x.shape[-1])

    times = torch.zeros(batch_size, max_length, dtype=torch.float32)
    x = torch.zeros(batch_size, max_length, n_features, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_length, n_features, dtype=torch.float32)
    delta_time = torch.zeros(batch_size, max_length, n_features, dtype=torch.float32)
    sequence_lengths = torch.zeros(batch_size, dtype=torch.long)

    for row, sample in enumerate(samples):
        length = int(sample.times.shape[0])
        times[row, :length] = sample.times
        x[row, :length] = sample.x
        mask[row, :length] = sample.mask
        delta_time[row, :length] = sample.delta_time
        sequence_lengths[row] = length

    return {
        "times": times,
        "x": x,
        "mask": mask,
        "delta_time": delta_time,
        "sequence_lengths": sequence_lengths,
        "survival_time": torch.stack([sample.survival_time for sample in samples]).float(),
        "event": torch.stack([sample.event for sample in samples]).float(),
        "cluster_label": torch.stack([sample.cluster_label for sample in samples]).long(),
    }


def infer_data_config(dataset: ClinicalTimeSeriesDataset) -> DataConfig:
    return DataConfig(n_features=dataset.n_features)


def make_data_loader(
    data: ClinicalTimeSeriesDataset,
    trainer_config: TrainerConfig,
    *,
    shuffle: bool,
) -> DataLoader[Batch]:
    return cast(
        DataLoader[Batch],
        DataLoader(
            data,
            batch_size=trainer_config.batch_size,
            shuffle=shuffle,
            collate_fn=clinical_collate_fn,
        ),
    )

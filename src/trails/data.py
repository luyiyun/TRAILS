from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .config import DataConfig, TrainerConfig

type SampleKind = Literal["aligned", "compact"]
Batch = dict[str, Tensor]
type AlignedBatch = dict[str, Tensor]
type CompactBatch = dict[str, Tensor]
PATIENT_LEVEL_METADATA_KEYS = frozenset({"latent_z", "sequence_lengths"})


@dataclass(frozen=True)
class AlignedClinicalSample:
    times: Tensor
    x: Tensor
    mask: Tensor
    delta_time: Tensor
    survival_time: Tensor
    event: Tensor
    cluster_label: Tensor | None

    def __post_init__(self) -> None:
        validate_aligned_clinical_sample(self)

    def to_aligned(self) -> AlignedClinicalSample:
        return self

    def to_compact(self) -> CompactClinicalSample:
        observed = self.mask > 0
        feature_lengths = observed.sum(dim=0).long()
        max_length = max(1, int(feature_lengths.max().item()))
        n_features = int(self.x.shape[-1])
        compact_times = self.x.new_zeros(max_length, n_features)
        compact_x = self.x.new_zeros(max_length, n_features)
        compact_mask = self.x.new_zeros(max_length, n_features)

        # compact 视图按变量独立左对齐，只保留真实观测点。
        for feature_index in range(n_features):
            feature_observed = observed[:, feature_index]
            length = int(feature_lengths[feature_index].item())
            if length == 0:
                continue
            compact_times[:length, feature_index] = self.times[feature_observed]
            compact_x[:length, feature_index] = self.x[feature_observed, feature_index]
            compact_mask[:length, feature_index] = 1.0

        return CompactClinicalSample(
            times=compact_times,
            x=compact_x,
            mask=compact_mask,
            feature_lengths=feature_lengths,
            survival_time=self.survival_time,
            event=self.event,
            cluster_label=self.cluster_label,
        )


@dataclass(frozen=True)
class CompactClinicalSample:
    times: Tensor
    x: Tensor
    mask: Tensor
    feature_lengths: Tensor
    survival_time: Tensor
    event: Tensor
    cluster_label: Tensor | None

    def __post_init__(self) -> None:
        validate_compact_clinical_sample(self)

    def to_compact(self) -> CompactClinicalSample:
        return self

    def to_aligned(self) -> AlignedClinicalSample:
        observed_times = self.times[self.mask > 0].float()
        if observed_times.numel() == 0:
            raise ValueError("compact samples must contain at least one observed value.")
        aligned_times = torch.unique(observed_times, sorted=True)
        n_visits = int(aligned_times.shape[0])
        n_features = int(self.x.shape[-1])
        aligned_x = self.x.new_zeros(n_visits, n_features)
        aligned_mask = self.x.new_zeros(n_visits, n_features)

        # 将每个变量自己的 compact 时间轴放回统一 aligned 时间轴。
        for feature_index in range(n_features):
            length = int(self.feature_lengths[feature_index].item())
            if length == 0:
                continue
            feature_times = self.times[:length, feature_index].float().contiguous()
            if int(torch.unique(feature_times).shape[0]) != length:
                raise ValueError("compact samples cannot contain duplicate feature-time pairs.")
            target_indices = torch.searchsorted(aligned_times, feature_times)
            aligned_x[target_indices, feature_index] = self.x[:length, feature_index]
            aligned_mask[target_indices, feature_index] = 1.0

        return AlignedClinicalSample(
            times=aligned_times,
            x=aligned_x,
            mask=aligned_mask,
            delta_time=compute_delta_time(aligned_times, aligned_mask),
            survival_time=self.survival_time,
            event=self.event,
            cluster_label=self.cluster_label,
        )


type DatasetSample = AlignedClinicalSample | CompactClinicalSample


class ClinicalTimeSeriesDataset(Dataset[DatasetSample]):
    def __init__(
        self,
        samples: Sequence[DatasetSample],
        *,
        feature_names: list[str],
        description: str = "",
        metadata: dict[str, Any] | None = None,
        return_kind: SampleKind = "aligned",
    ) -> None:
        if return_kind not in {"aligned", "compact"}:
            raise ValueError("return_kind must be 'aligned' or 'compact'.")
        input_samples = list(samples)
        if not input_samples:
            raise ValueError("ClinicalTimeSeriesDataset requires at least one sample.")
        if len(feature_names) != int(input_samples[0].x.shape[-1]):
            raise ValueError("feature_names length must match sample feature dimension.")
        for sample in input_samples:
            if int(sample.x.shape[-1]) != len(feature_names):
                raise ValueError("All samples must share the same feature dimension.")
        converted_samples: list[DatasetSample] = [
            sample.to_aligned() if return_kind == "aligned" else sample.to_compact()
            for sample in input_samples
        ]
        self.has_cluster_labels: bool = converted_samples[0].cluster_label is not None
        for sample in converted_samples:
            if (sample.cluster_label is not None) != self.has_cluster_labels:
                raise ValueError(
                    "ClinicalTimeSeriesDataset cannot mix labeled and unlabeled samples."
                )
        self.samples = converted_samples
        self.feature_names = feature_names
        self.description = description
        self.metadata = metadata or {}
        self.feature_means = compute_feature_means(converted_samples)
        self.return_kind: SampleKind = return_kind

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> DatasetSample:
        return self.samples[index]

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def with_return_kind(self, return_kind: SampleKind) -> ClinicalTimeSeriesDataset:
        return ClinicalTimeSeriesDataset(
            self.samples,
            feature_names=self.feature_names,
            description=self.description,
            metadata=self.metadata,
            return_kind=return_kind,
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        aligned_samples = [sample.to_aligned() for sample in self.samples]
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
                for sample in aligned_samples
            ],
        }
        torch.save(payload, destination)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        return_kind: SampleKind = "aligned",
    ) -> ClinicalTimeSeriesDataset:
        payload: dict[str, Any] = torch.load(Path(path), map_location="cpu", weights_only=False)
        samples = [
            make_clinical_sample(
                times=sample["times"],
                x=sample["x"],
                mask=sample["mask"],
                delta_time=sample["delta_time"],
                survival_time=sample["survival_time"],
                event=sample["event"],
                cluster_label=sample.get("cluster_label"),
            )
            for sample in payload["samples"]
        ]
        return ClinicalTimeSeriesDataset(
            samples,
            feature_names=list(payload["feature_names"]),
            description=str(payload.get("description", "")),
            metadata=dict(payload.get("metadata", {})),
            return_kind=return_kind,
        )

    def split(self, fraction: list[float], seed: int = 0) -> list[ClinicalTimeSeriesDataset]:
        if not np.isclose(float(sum(fraction)), 1.0):
            raise ValueError("Split fractions must sum to 1.")

        if len(fraction) == 1:
            return [self]

        n_samples = len(self.samples)
        split_indices = np.cumsum(np.array(fraction) * n_samples).astype(int).tolist()
        counts = [
            end - start for start, end in zip([0] + split_indices[:-1], split_indices, strict=True)
        ]
        return self.split_counts(counts, seed=seed, split_fractions=fraction)

    def split_counts(
        self,
        counts: list[int],
        seed: int = 0,
        *,
        split_fractions: list[float] | None = None,
    ) -> list[ClinicalTimeSeriesDataset]:
        if not counts:
            raise ValueError("Split counts must contain at least one split.")
        if any(count <= 0 for count in counts):
            raise ValueError("Split counts must be positive.")
        if sum(counts) != len(self.samples):
            raise ValueError("Split counts must sum to dataset length.")
        if len(counts) == 1:
            return [self]

        rng = np.random.default_rng(seed)
        indices = np.arange(len(self.samples))
        rng.shuffle(indices)
        split_indices = np.cumsum(np.array(counts)).astype(int).tolist()
        res = []
        for split_index, (start, end) in enumerate(
            zip([0] + split_indices[:-1], split_indices, strict=True)
        ):
            split_sample_indices = indices[start:end]
            samples_i = [self.samples[int(i)] for i in split_sample_indices]
            res.append(
                ClinicalTimeSeriesDataset(
                    samples_i,
                    feature_names=self.feature_names,
                    description=f"{self.description} (split {split_index + 1}/{len(counts)})",
                    metadata=self._split_metadata(
                        split_sample_indices,
                        split_index=split_index,
                        split_count=len(counts),
                        split_fraction=(
                            None if split_fractions is None else split_fractions[split_index]
                        ),
                    ),
                    return_kind=self.return_kind,
                )
            )

        return res

    def _split_metadata(
        self,
        indices: np.ndarray,
        *,
        split_index: int,
        split_count: int,
        split_fraction: float | None,
    ) -> dict[str, Any]:
        metadata = dict(self.metadata)
        for key in PATIENT_LEVEL_METADATA_KEYS:
            if key in metadata:
                metadata[key] = _slice_patient_metadata(
                    metadata[key],
                    indices,
                    source_count=len(self.samples),
                )
        metadata.update(
            {
                "source_patient_count": len(self.samples),
                "split_index": split_index,
                "split_count": split_count,
                "split_patient_count": int(indices.shape[0]),
            }
        )
        if split_fraction is not None:
            metadata["split_fraction"] = split_fraction
        return metadata


def _slice_patient_metadata(value: Any, indices: np.ndarray, *, source_count: int) -> Any:
    if isinstance(value, torch.Tensor) and value.ndim > 0 and int(value.shape[0]) == source_count:
        tensor_indices = torch.as_tensor(indices, dtype=torch.long)
        return value[tensor_indices]
    if isinstance(value, np.ndarray) and value.ndim > 0 and int(value.shape[0]) == source_count:
        return value[indices]
    if isinstance(value, list) and len(value) == source_count:
        return [value[int(index)] for index in indices]
    if isinstance(value, tuple) and len(value) == source_count:
        return tuple(value[int(index)] for index in indices)
    return value


def make_clinical_sample(
    *,
    times: Tensor,
    x: Tensor,
    mask: Tensor,
    delta_time: Tensor,
    survival_time: float | Tensor,
    event: float | Tensor,
    cluster_label: int | Tensor | None = None,
) -> AlignedClinicalSample:
    return AlignedClinicalSample(
        times=times.float(),
        x=x.float(),
        mask=mask.float(),
        delta_time=delta_time.float(),
        survival_time=torch.as_tensor(survival_time, dtype=torch.float32),
        event=torch.as_tensor(event, dtype=torch.float32),
        cluster_label=(
            None if cluster_label is None else torch.as_tensor(cluster_label, dtype=torch.long)
        ),
    )


def compute_delta_time(times: Tensor, mask: Tensor) -> Tensor:
    delta_time = torch.zeros_like(mask)
    for step in range(1, int(times.shape[0])):
        gap = times[step] - times[step - 1]
        delta_time[step] = torch.where(mask[step - 1] > 0, gap, delta_time[step - 1] + gap)
    return delta_time


def validate_aligned_clinical_sample(sample: AlignedClinicalSample) -> None:
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
    if sample.cluster_label is not None and sample.cluster_label.ndim > 0:
        raise ValueError("cluster_label must be a scalar tensor when provided.")


def validate_compact_clinical_sample(sample: CompactClinicalSample) -> None:
    if sample.times.ndim != 2:
        raise ValueError("compact times must have shape (max_observations, n_features).")
    if sample.x.shape != sample.times.shape:
        raise ValueError("compact x must have the same shape as compact times.")
    if sample.mask.shape != sample.x.shape:
        raise ValueError("compact mask must have the same shape as compact x.")
    if sample.feature_lengths.shape != (sample.x.shape[-1],):
        raise ValueError("feature_lengths must have shape (n_features,).")
    if torch.any((sample.mask < 0) | (sample.mask > 1)):
        raise ValueError("compact mask values must be in [0, 1].")
    if torch.any(sample.feature_lengths < 0):
        raise ValueError("feature_lengths values must be non-negative.")
    if torch.any(sample.feature_lengths > sample.x.shape[0]):
        raise ValueError("feature_lengths cannot exceed compact sequence length.")
    positions = torch.arange(int(sample.x.shape[0]), device=sample.mask.device).unsqueeze(1)
    expected_mask = positions < sample.feature_lengths.to(sample.mask.device).unsqueeze(0)
    if not torch.equal(sample.mask > 0, expected_mask):
        raise ValueError("compact mask must contain left-aligned observations only.")
    if float(sample.survival_time) <= 0:
        raise ValueError("survival_time must be positive.")
    if float(sample.event) < 0 or float(sample.event) > 1:
        raise ValueError("event must be in [0, 1].")
    if sample.cluster_label is not None and sample.cluster_label.ndim > 0:
        raise ValueError("cluster_label must be a scalar tensor when provided.")


def compute_feature_means(samples: Sequence[DatasetSample]) -> Tensor:
    n_features = int(samples[0].x.shape[-1])
    numerator = torch.zeros(n_features, dtype=torch.float32)
    denominator = torch.zeros(n_features, dtype=torch.float32)
    for sample in samples:
        aligned_sample = sample.to_aligned()
        numerator += (aligned_sample.x * aligned_sample.mask).sum(dim=0)
        denominator += aligned_sample.mask.sum(dim=0)
    return numerator / denominator.clamp_min(1.0)


def clinical_collate_fn(samples: list[DatasetSample]) -> Batch:
    if not samples:
        raise ValueError("clinical_collate_fn requires at least one sample.")
    if isinstance(samples[0], CompactClinicalSample):
        if not all(isinstance(sample, CompactClinicalSample) for sample in samples):
            raise ValueError("clinical_collate_fn cannot mix aligned and compact samples.")
        return collate_compact_samples(cast(list[CompactClinicalSample], samples))
    if not all(isinstance(sample, AlignedClinicalSample) for sample in samples):
        raise ValueError("clinical_collate_fn cannot mix aligned and compact samples.")
    return collate_aligned_samples(cast(list[AlignedClinicalSample], samples))


def collate_aligned_samples(samples: list[AlignedClinicalSample]) -> AlignedBatch:
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

    has_cluster_labels = samples[0].cluster_label is not None
    for sample in samples:
        if (sample.cluster_label is not None) != has_cluster_labels:
            raise ValueError("clinical_collate_fn cannot mix labeled and unlabeled samples.")

    batch = {
        "times": times,
        "x": x,
        "mask": mask,
        "delta_time": delta_time,
        "sequence_lengths": sequence_lengths,
        "survival_time": torch.stack([sample.survival_time for sample in samples]).float(),
        "event": torch.stack([sample.event for sample in samples]).float(),
    }
    if has_cluster_labels:
        cluster_labels = [
            sample.cluster_label for sample in samples if sample.cluster_label is not None
        ]
        batch["cluster_label"] = torch.stack(cluster_labels).long()
    return batch


def collate_compact_samples(samples: list[CompactClinicalSample]) -> CompactBatch:
    batch_size = len(samples)
    max_length = max(int(sample.x.shape[0]) for sample in samples)
    n_features = int(samples[0].x.shape[-1])

    times = torch.zeros(batch_size, max_length, n_features, dtype=torch.float32)
    x = torch.zeros(batch_size, max_length, n_features, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_length, n_features, dtype=torch.float32)
    feature_lengths = torch.zeros(batch_size, n_features, dtype=torch.long)

    for row, sample in enumerate(samples):
        length = int(sample.x.shape[0])
        times[row, :length] = sample.times
        x[row, :length] = sample.x
        mask[row, :length] = sample.mask
        feature_lengths[row] = sample.feature_lengths

    has_cluster_labels = samples[0].cluster_label is not None
    for sample in samples:
        if (sample.cluster_label is not None) != has_cluster_labels:
            raise ValueError("clinical_collate_fn cannot mix labeled and unlabeled samples.")

    batch = {
        "times": times,
        "x": x,
        "mask": mask,
        "feature_lengths": feature_lengths,
        "survival_time": torch.stack([sample.survival_time for sample in samples]).float(),
        "event": torch.stack([sample.event for sample in samples]).float(),
    }
    if has_cluster_labels:
        cluster_labels = [
            sample.cluster_label for sample in samples if sample.cluster_label is not None
        ]
        batch["cluster_label"] = torch.stack(cluster_labels).long()
    return batch


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

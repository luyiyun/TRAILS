from pathlib import Path
from typing import cast

import pandas as pd
import pytest
import torch
from pydantic import ValidationError

from trails import (
    AlignedClinicalSample,
    ClinicalTimeSeriesDataset,
    CompactClinicalSample,
    DecoderConfig,
    EncoderInputConfig,
    ModelConfig,
    clinical_collate_fn,
    resolve_batch_size,
)
from trails.data import make_clinical_sample
from trails_simulate import (
    ClinicalTimeSeriesDatasetGenerator,
    ClinicalTimeSeriesDatasetGeneratorConfig,
)


def simulate_dataset(
    *,
    patients: int,
    n_clusters: int,
    min_visits: int,
    max_visits: int,
    hidden_size: int,
    latent_dim: int,
    attention_layers: int,
    seed: int,
) -> ClinicalTimeSeriesDataset:
    config = ClinicalTimeSeriesDatasetGeneratorConfig(
        n_clusters=n_clusters,
        min_visits=min_visits,
        max_visits=max_visits,
        hidden_size=hidden_size,
        latent_dim=latent_dim,
        attention_layers=attention_layers,
    )
    return ClinicalTimeSeriesDatasetGenerator(config, mechanism_seed=seed).simulate(
        n_patients=patients,
        seed=seed,
    )


@pytest.mark.parametrize(
    ("n_samples", "expected"),
    [
        (8, 8),
        (64, 16),
        (500, 32),
        (1000, 64),
        (2000, 128),
        (3000, 256),
        (5000, 256),
    ],
)
def test_resolve_batch_size_uses_auto_rule(n_samples: int, expected: int) -> None:
    assert resolve_batch_size(n_samples, None) == expected


def test_resolve_batch_size_preserves_explicit_override() -> None:
    assert resolve_batch_size(8, 64) == 64


def test_encoder_input_config_accepts_supported_kinds_and_rejects_unknown() -> None:
    for kind in ("grud", "mtan", "mtan2"):
        assert EncoderInputConfig(kind=kind).kind == kind

    with pytest.raises(ValidationError):
        EncoderInputConfig.model_validate({"kind": "unknown"})


def test_clinical_dataset_and_collate_shapes() -> None:
    dataset = simulate_dataset(
        patients=6,
        n_clusters=2,
        min_visits=3,
        max_visits=5,
        hidden_size=12,
        latent_dim=4,
        attention_layers=2,
        seed=7,
    )
    sample = dataset.samples[0].to_aligned()
    batch = clinical_collate_fn([dataset[0], dataset[1]])

    assert len(dataset) == 6
    assert sample.x.shape == sample.mask.shape
    assert sample.delta_time.shape == sample.x.shape
    assert batch["x"].shape[0] == 2
    assert batch["x"].shape[-1] == dataset.n_features
    assert batch["sequence_lengths"].shape == (2,)
    assert dataset.has_cluster_labels
    assert batch["cluster_label"].shape == (2,)
    assert {"latent_z", "cluster_means", "survival_coefficients", "generation_params"} <= set(
        dataset.metadata
    )


def test_compact_dataset_view_and_collate() -> None:
    dataset = simulate_dataset(
        patients=4,
        n_clusters=2,
        min_visits=3,
        max_visits=5,
        hidden_size=12,
        latent_dim=4,
        attention_layers=2,
        seed=9,
    )
    compact_dataset = dataset.with_return_kind("compact")
    aligned_sample = dataset.samples[0].to_aligned()
    compact_sample = cast(CompactClinicalSample, compact_dataset[0])

    assert compact_dataset.return_kind == "compact"
    assert isinstance(compact_dataset.samples[0], CompactClinicalSample)
    assert isinstance(compact_dataset[0], CompactClinicalSample)
    assert compact_sample.x.shape == compact_sample.times.shape == compact_sample.mask.shape
    assert compact_sample.x.ndim == 2
    assert compact_sample.x.shape[1] == dataset.n_features
    assert compact_sample.feature_lengths.shape == (dataset.n_features,)
    assert torch.equal(compact_sample.feature_lengths, aligned_sample.mask.sum(dim=0).long())
    for feature_index in range(dataset.n_features):
        observed = aligned_sample.mask[:, feature_index] > 0
        length = int(compact_sample.feature_lengths[feature_index])
        assert torch.allclose(
            compact_sample.times[:length, feature_index],
            aligned_sample.times[observed],
        )
        assert torch.allclose(
            compact_sample.x[:length, feature_index],
            aligned_sample.x[observed, feature_index],
        )
        assert torch.all(compact_sample.mask[length:, feature_index] == 0)

    batch = clinical_collate_fn([compact_dataset[0], compact_dataset[1]])
    assert "feature_lengths" in batch
    assert "delta_time" not in batch
    assert batch["x"].ndim == 3
    assert batch["x"].shape == batch["times"].shape == batch["mask"].shape
    assert batch["x"].shape[0] == 2
    assert batch["x"].shape[-1] == dataset.n_features

    with pytest.raises(ValueError, match="cannot mix aligned and compact"):
        clinical_collate_fn([dataset[0], compact_dataset[0]])


def test_sample_view_conversion_methods_and_dataset_init(tmp_path: Path) -> None:
    aligned_sample = make_clinical_sample(
        times=torch.tensor([0.0, 1.0, 3.0]),
        x=torch.tensor([[1.0, 0.0], [2.0, 3.0], [0.0, 4.0]]),
        mask=torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        delta_time=torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
        survival_time=5.0,
        event=1.0,
        cluster_label=0,
    )

    compact_sample = aligned_sample.to_compact()

    assert isinstance(compact_sample, CompactClinicalSample)
    assert torch.equal(compact_sample.feature_lengths, torch.tensor([2, 2]))
    assert torch.allclose(
        compact_sample.times,
        torch.tensor([[0.0, 1.0], [1.0, 3.0]]),
    )
    assert torch.allclose(
        compact_sample.x,
        torch.tensor([[1.0, 3.0], [2.0, 4.0]]),
    )
    assert torch.all(compact_sample.mask == 1)

    roundtrip_sample = compact_sample.to_aligned()
    assert isinstance(roundtrip_sample, AlignedClinicalSample)
    assert torch.allclose(roundtrip_sample.times, aligned_sample.times)
    assert torch.allclose(roundtrip_sample.x, aligned_sample.x)
    assert torch.allclose(roundtrip_sample.mask, aligned_sample.mask)
    assert torch.allclose(roundtrip_sample.delta_time, aligned_sample.delta_time)

    compact_dataset = ClinicalTimeSeriesDataset(
        [aligned_sample],
        feature_names=["a", "b"],
        return_kind="compact",
    )
    assert isinstance(compact_dataset.samples[0], CompactClinicalSample)
    assert isinstance(compact_dataset[0], CompactClinicalSample)

    aligned_dataset = ClinicalTimeSeriesDataset(
        [compact_sample],
        feature_names=["a", "b"],
        return_kind="aligned",
    )
    assert isinstance(aligned_dataset.samples[0], AlignedClinicalSample)
    assert torch.allclose(aligned_dataset.feature_means, torch.tensor([1.5, 3.5]))

    save_path = tmp_path / "compact.pt"
    compact_dataset.save(save_path)
    loaded = ClinicalTimeSeriesDataset.load(save_path, return_kind="compact")
    assert isinstance(loaded.samples[0], CompactClinicalSample)
    assert torch.allclose(loaded.samples[0].x, compact_sample.x)


def test_clinical_dataset_loads_default_csv_format(tmp_path: Path) -> None:
    patients_csv = tmp_path / "patients.csv"
    observations_csv = tmp_path / "observations.csv"
    pd.DataFrame(
        [
            {"patient_id": "p1", "survival_time": 5, "event": 1, "cluster_label": 0},
            {"patient_id": "p2", "survival_time": 4, "event": 0, "cluster_label": 1},
        ]
    ).to_csv(patients_csv, index=False)
    pd.DataFrame(
        [
            {"patient_id": "p1", "time": 0, "feature": "crp", "value": 1.0},
            {"patient_id": "p1", "time": 2, "feature": "albumin", "value": 3.0},
            {"patient_id": "p2", "time": 1, "feature": "crp", "value": 2.0},
            {"patient_id": "p2", "time": 1, "feature": "albumin", "value": 4.0},
        ]
    ).to_csv(observations_csv, index=False)

    dataset = ClinicalTimeSeriesDataset.load_from_csv(
        patients_csv=patients_csv,
        observations_csv=observations_csv,
        description="csv test",
        metadata={"source": "unit_test"},
    )
    first = dataset[0].to_aligned()

    assert len(dataset) == 2
    assert dataset.description == "csv test"
    assert dataset.feature_names == ["crp", "albumin"]
    assert dataset.has_cluster_labels
    assert dataset.metadata["source"] == "unit_test"
    assert dataset.metadata["patient_ids"] == ["p1", "p2"]
    assert dataset.metadata["n_observations"] == 4
    assert dataset.metadata["patient_summaries"][0]["missing_fraction"] == 0.5
    assert dataset.metadata["csv_columns"]["patients"]["patient_id"] == "patient_id"
    assert torch.allclose(first.times, torch.tensor([0.0, 2.0]))
    assert torch.allclose(first.x, torch.tensor([[1.0, 0.0], [0.0, 3.0]]))
    assert torch.allclose(first.mask, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    assert torch.allclose(first.delta_time, torch.tensor([[0.0, 0.0], [2.0, 2.0]]))


def test_clinical_dataset_loads_custom_csv_columns_and_compact_view(tmp_path: Path) -> None:
    patients_csv = tmp_path / "subjects.csv"
    observations_csv = tmp_path / "measurements.csv"
    pd.DataFrame(
        [
            {"subject": "p1", "duration": 5, "status": 1},
            {"subject": "p2", "duration": 4, "status": 0},
        ]
    ).to_csv(patients_csv, index=False)
    pd.DataFrame(
        [
            {"subject": "p1", "visit_time": 0, "marker": "crp", "reading": 1.0},
            {"subject": "p1", "visit_time": 2, "marker": "albumin", "reading": 3.0},
            {"subject": "p2", "visit_time": 1, "marker": "crp", "reading": 2.0},
            {"subject": "p2", "visit_time": 1, "marker": "albumin", "reading": 4.0},
        ]
    ).to_csv(observations_csv, index=False)

    dataset = ClinicalTimeSeriesDataset.load_from_csv(
        patients_csv=patients_csv,
        observations_csv=observations_csv,
        patient_id_col="subject",
        survival_time_col="duration",
        event_col="status",
        observation_id_col="subject",
        time_col="visit_time",
        feature_col="marker",
        value_col="reading",
        use_features=["albumin", "crp"],
        return_kind="compact",
    )
    first = cast(CompactClinicalSample, dataset[0])

    assert dataset.return_kind == "compact"
    assert dataset.feature_names == ["albumin", "crp"]
    assert not dataset.has_cluster_labels
    assert torch.equal(first.feature_lengths, torch.tensor([1, 1]))
    assert torch.allclose(first.x, torch.tensor([[3.0, 1.0]]))
    assert dataset.metadata["csv_columns"]["observations"]["value"] == "reading"


def test_clinical_dataset_save_to_csv_roundtrips_compact_dataset(tmp_path: Path) -> None:
    first = make_clinical_sample(
        times=torch.tensor([0.0, 1.0]),
        x=torch.tensor([[1.0, 0.0], [2.0, 3.0]]),
        mask=torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
        delta_time=torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
        survival_time=5.0,
        event=1.0,
        cluster_label=0,
    )
    second = make_clinical_sample(
        times=torch.tensor([2.0]),
        x=torch.tensor([[0.0, 4.0]]),
        mask=torch.tensor([[0.0, 1.0]]),
        delta_time=torch.tensor([[0.0, 0.0]]),
        survival_time=6.0,
        event=0.0,
        cluster_label=1,
    )
    source = ClinicalTimeSeriesDataset(
        [first, second],
        feature_names=["crp", "albumin"],
        metadata={"patient_ids": ["alpha", "beta"]},
        return_kind="compact",
    )
    patients_csv = tmp_path / "patients.csv"
    observations_csv = tmp_path / "observations.csv"

    source.save_to_csv(patients_csv=patients_csv, observations_csv=observations_csv)
    patient_rows = pd.read_csv(patients_csv, keep_default_na=False, dtype=str).to_dict("records")
    observation_rows = pd.read_csv(
        observations_csv,
        keep_default_na=False,
        dtype=str,
    ).to_dict("records")
    loaded = ClinicalTimeSeriesDataset.load_from_csv(
        patients_csv=patients_csv,
        observations_csv=observations_csv,
    )

    assert patient_rows[0]["patient_id"] == "alpha"
    assert patient_rows[1]["patient_id"] == "beta"
    assert len(observation_rows) == 4
    assert loaded.metadata["patient_ids"] == ["alpha", "beta"]
    assert loaded.has_cluster_labels
    assert torch.allclose(loaded[0].to_aligned().x, first.x)
    assert torch.allclose(loaded[0].to_aligned().mask, first.mask)
    assert torch.allclose(loaded[1].to_aligned().x, second.x)
    assert torch.allclose(loaded[1].to_aligned().mask, second.mask)


def test_clinical_dataset_save_to_csv_accepts_explicit_patient_ids(tmp_path: Path) -> None:
    sample = make_clinical_sample(
        times=torch.tensor([0.0]),
        x=torch.tensor([[1.0]]),
        mask=torch.tensor([[1.0]]),
        delta_time=torch.tensor([[0.0]]),
        survival_time=5.0,
        event=1.0,
    )
    dataset = ClinicalTimeSeriesDataset([sample], feature_names=["crp"])
    patients_csv = tmp_path / "patients.csv"
    observations_csv = tmp_path / "observations.csv"

    dataset.save_to_csv(
        patients_csv=patients_csv,
        observations_csv=observations_csv,
        patient_ids=["manual"],
    )

    rows = pd.read_csv(patients_csv, keep_default_na=False, dtype=str).to_dict("records")
    assert rows == [{"patient_id": "manual", "survival_time": "5.0", "event": "1.0"}]


def test_compact_to_aligned_rejects_duplicate_feature_times() -> None:
    compact_sample = CompactClinicalSample(
        times=torch.tensor([[0.0], [0.0]]),
        x=torch.tensor([[1.0], [2.0]]),
        mask=torch.tensor([[1.0], [1.0]]),
        feature_lengths=torch.tensor([2]),
        survival_time=torch.tensor(5.0),
        event=torch.tensor(1.0),
        cluster_label=torch.tensor(0),
    )

    with pytest.raises(ValueError, match="duplicate feature-time"):
        compact_sample.to_aligned()


def test_compact_dataset_allows_duplicate_feature_times_for_feature_means() -> None:
    compact_sample = CompactClinicalSample(
        times=torch.tensor([[0.0, 1.0], [0.0, 2.0], [0.0, 0.0]]),
        x=torch.tensor([[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]),
        mask=torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]]),
        feature_lengths=torch.tensor([2, 2]),
        survival_time=torch.tensor(5.0),
        event=torch.tensor(1.0),
        cluster_label=torch.tensor(0),
    )

    compact_dataset = ClinicalTimeSeriesDataset(
        [compact_sample],
        feature_names=["a", "b"],
        return_kind="compact",
    )

    assert isinstance(compact_dataset.samples[0], CompactClinicalSample)
    assert torch.allclose(compact_dataset.feature_means, torch.tensor([2.0, 3.0]))
    with pytest.raises(ValueError, match="duplicate feature-time"):
        compact_sample.to_aligned()


def test_dataset_split_counts_preserves_sizes_and_metadata() -> None:
    dataset = simulate_dataset(
        patients=7,
        n_clusters=2,
        min_visits=3,
        max_visits=5,
        hidden_size=12,
        latent_dim=4,
        attention_layers=2,
        seed=17,
    )

    train, test = dataset.split_counts([5, 2], seed=23)

    assert len(train) == 5
    assert len(test) == 2
    assert train.metadata["latent_z"].shape[0] == 5
    assert test.metadata["latent_z"].shape[0] == 2
    assert train.metadata["sequence_lengths"].shape[0] == 5
    assert test.metadata["sequence_lengths"].shape[0] == 2
    assert torch.allclose(train.metadata["cluster_means"], dataset.metadata["cluster_means"])
    assert torch.allclose(test.metadata["cluster_means"], dataset.metadata["cluster_means"])


def test_unlabeled_dataset_collates_without_cluster_labels() -> None:
    labeled = simulate_dataset(
        patients=4,
        n_clusters=2,
        min_visits=3,
        max_visits=4,
        hidden_size=12,
        latent_dim=4,
        attention_layers=2,
        seed=19,
    )
    aligned_samples = [sample.to_aligned() for sample in labeled.samples]
    samples = [
        make_clinical_sample(
            times=sample.times,
            x=sample.x,
            mask=sample.mask,
            delta_time=sample.delta_time,
            survival_time=sample.survival_time,
            event=sample.event,
        )
        for sample in aligned_samples
    ]
    unlabeled = ClinicalTimeSeriesDataset(samples, feature_names=labeled.feature_names)
    batch = clinical_collate_fn([unlabeled[0], unlabeled[1]])

    assert not unlabeled.has_cluster_labels
    assert "cluster_label" not in batch


def test_dataset_rejects_mixed_cluster_label_availability() -> None:
    labeled = simulate_dataset(
        patients=4,
        n_clusters=2,
        min_visits=3,
        max_visits=4,
        hidden_size=12,
        latent_dim=4,
        attention_layers=2,
        seed=23,
    )
    labeled_sample = labeled.samples[1].to_aligned()
    unlabeled_sample = make_clinical_sample(
        times=labeled_sample.times,
        x=labeled_sample.x,
        mask=labeled_sample.mask,
        delta_time=labeled_sample.delta_time,
        survival_time=labeled_sample.survival_time,
        event=labeled_sample.event,
    )

    with pytest.raises(ValueError, match="cannot mix labeled and unlabeled"):
        ClinicalTimeSeriesDataset(
            [labeled.samples[0].to_aligned(), unlabeled_sample],
            feature_names=labeled.feature_names,
        )


def test_transformer_decoder_rejects_initial_state_conditioning() -> None:
    with pytest.raises(ValidationError, match="Transformer decoder only supports"):
        DecoderConfig(kind="transformer", conditioning="initial_state")


def test_model_config_rejects_stale_flat_architecture_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ModelConfig.model_validate({"encoder_hidden_dim": 8})

from pathlib import Path
from typing import Literal, cast

import pandas as pd
import pytest
import torch
from pydantic import ValidationError

from trails.config import (
    DataConfig,
    DecoderConfig,
    EncoderConfig,
    EncoderInputConfig,
    EncoderMappingConfig,
    LossConfig,
    ModelConfig,
    TrainerConfig,
    resolve_batch_size,
)
from trails.data import (
    AlignedClinicalSample,
    Batch,
    ClinicalTimeSeriesDataset,
    CompactClinicalSample,
    clinical_collate_fn,
    make_clinical_sample,
    make_data_loader,
)
from trails.metrics import (
    ClusteringAccuracy,
    masked_mse,
    vade_kl_loss,
    weibull_mixture_negative_log_likelihood,
)
from trails.model import (
    MTAN2InputLayer,
    MTANInputLayer,
    MultiTimeAttention,
    SequencePool,
    TimeEmbedding,
    TrailsModelOutput,
    TrailsSurvVaderModel,
)
from trails_simulate import (
    ClinicalTimeSeriesDatasetGenerator,
    ClinicalTimeSeriesDatasetGeneratorConfig,
)

EncoderInputKind = Literal["grud", "mtan", "mtan2"]
EncoderMappingKind = Literal["gru", "lstm", "transformer"]
DecoderKind = Literal["gru", "lstm", "transformer"]
DecoderConditioning = Literal["initial_state", "concat_time"]


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
    censoring_rate: float = 0.3,
) -> ClinicalTimeSeriesDataset:
    config = ClinicalTimeSeriesDatasetGeneratorConfig(
        n_clusters=n_clusters,
        min_visits=min_visits,
        max_visits=max_visits,
        hidden_size=hidden_size,
        latent_dim=latent_dim,
        attention_layers=attention_layers,
        censoring_rate=censoring_rate,
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


def test_data_loader_accepts_auto_batch_size() -> None:
    dataset = simulate_dataset(
        patients=8,
        n_clusters=2,
        min_visits=3,
        max_visits=5,
        hidden_size=12,
        latent_dim=4,
        attention_layers=2,
        seed=5,
    )
    loader = make_data_loader(dataset, TrainerConfig(batch_size=None), shuffle=False)
    batch = next(iter(loader))

    assert batch["x"].shape[0] == 8


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


def test_generator_reuses_mechanism_across_sample_seeds() -> None:
    config = ClinicalTimeSeriesDatasetGeneratorConfig(
        n_clusters=2,
        min_visits=3,
        max_visits=5,
        hidden_size=12,
        latent_dim=4,
        attention_layers=1,
    )
    generator = ClinicalTimeSeriesDatasetGenerator(config, mechanism_seed=101)

    first = generator.simulate(n_patients=8, seed=201)
    second = generator.simulate(n_patients=8, seed=202)

    assert torch.allclose(first.metadata["cluster_means"], second.metadata["cluster_means"])
    assert torch.allclose(
        first.metadata["survival_coefficients"],
        second.metadata["survival_coefficients"],
    )
    assert first.metadata["generation_params"]["sample_seed"] == 201
    assert second.metadata["generation_params"]["sample_seed"] == 202
    assert not torch.allclose(first.metadata["latent_z"], second.metadata["latent_z"])


def test_generator_is_reproducible_with_same_mechanism_and_sample_seed() -> None:
    config = ClinicalTimeSeriesDatasetGeneratorConfig(
        n_clusters=2,
        min_visits=3,
        max_visits=5,
        hidden_size=12,
        latent_dim=4,
        attention_layers=1,
    )

    first = ClinicalTimeSeriesDatasetGenerator(config, mechanism_seed=101).simulate(
        n_patients=8,
        seed=201,
    )
    second = ClinicalTimeSeriesDatasetGenerator(config, mechanism_seed=101).simulate(
        n_patients=8,
        seed=201,
    )

    assert torch.allclose(first.metadata["cluster_means"], second.metadata["cluster_means"])
    assert torch.allclose(first.metadata["latent_z"], second.metadata["latent_z"])
    for index in range(len(first)):
        first_sample = first[index]
        second_sample = second[index]
        assert torch.allclose(first_sample.times, second_sample.times)
        assert torch.allclose(first_sample.x, second_sample.x)
        assert torch.allclose(first_sample.mask, second_sample.mask)
        assert torch.allclose(first_sample.survival_time, second_sample.survival_time)
        assert torch.allclose(first_sample.event, second_sample.event)


def test_generator_mechanism_seed_changes_mechanism_parameters() -> None:
    config = ClinicalTimeSeriesDatasetGeneratorConfig(
        n_clusters=2,
        min_visits=3,
        max_visits=5,
        hidden_size=12,
        latent_dim=4,
        attention_layers=1,
    )

    first = ClinicalTimeSeriesDatasetGenerator(config, mechanism_seed=101).simulate(
        n_patients=8,
        seed=201,
    )
    second = ClinicalTimeSeriesDatasetGenerator(config, mechanism_seed=102).simulate(
        n_patients=8,
        seed=201,
    )

    assert not torch.allclose(first.metadata["cluster_means"], second.metadata["cluster_means"])


def test_generator_rejects_patient_count_not_exceeding_clusters() -> None:
    config = ClinicalTimeSeriesDatasetGeneratorConfig(
        n_clusters=3,
        min_visits=3,
        max_visits=5,
        hidden_size=12,
        latent_dim=4,
        attention_layers=1,
    )
    generator = ClinicalTimeSeriesDatasetGenerator(config, mechanism_seed=101)

    with pytest.raises(ValueError, match="n_patients must be greater than n_clusters"):
        generator.simulate(n_patients=3, seed=201)


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


def test_simulation_has_asynchronous_masks_and_valid_delta_time() -> None:
    dataset = simulate_dataset(
        patients=64,
        n_clusters=2,
        min_visits=4,
        max_visits=5,
        hidden_size=12,
        latent_dim=4,
        attention_layers=2,
        censoring_rate=0.3,
        seed=11,
    )
    aligned_samples = [sample.to_aligned() for sample in dataset.samples]
    sequence_lengths = {int(sample.times.shape[0]) for sample in aligned_samples}
    assert len(sequence_lengths) > 1

    partial_visit_exists = any(
        bool(
            torch.any((sample.mask.sum(dim=1) > 0) & (sample.mask.sum(dim=1) < dataset.n_features))
        )
        for sample in aligned_samples
    )
    assert partial_visit_exists

    for sample in aligned_samples:
        assert sample.times.max() <= sample.survival_time
        assert torch.all(sample.delta_time >= 0)
        for step in range(1, int(sample.times.shape[0])):
            missing_previously = sample.mask[step - 1] == 0
            assert torch.all(
                sample.delta_time[step][missing_previously]
                >= sample.delta_time[step - 1][missing_previously]
            )

    observed_values = torch.cat([sample.x[sample.mask > 0] for sample in aligned_samples])
    assert observed_values.dtype.is_floating_point
    assert not torch.allclose(observed_values, observed_values.round())

    event_rate = torch.stack([sample.event for sample in aligned_samples]).float().mean()
    assert 0.45 <= float(event_rate) <= 0.95


def test_generator_cluster_prior_power_creates_imbalanced_clusters() -> None:
    config = ClinicalTimeSeriesDatasetGeneratorConfig(
        n_clusters=4,
        cluster_prior_power=1.5,
        min_visits=3,
        max_visits=5,
        hidden_size=12,
        latent_dim=4,
        attention_layers=1,
        feature_names=["hemoglobin", "albumin", "tumor_size"],
    )

    dataset = ClinicalTimeSeriesDatasetGenerator(config, mechanism_seed=101).simulate(
        n_patients=200,
        seed=201,
    )
    prior = dataset.metadata["cluster_prior"]
    labels = torch.stack(
        [sample.cluster_label for sample in dataset if sample.cluster_label is not None]
    )
    counts = torch.bincount(labels.long(), minlength=4)

    assert torch.isclose(prior.sum(), torch.tensor(1.0))
    assert prior[0] > prior[-1]
    assert counts[0] > counts[-1]
    assert dataset.metadata["generation_params"]["n_features"] == 3
    assert dataset.metadata["generation_params"]["feature_names"] == [
        "hemoglobin",
        "albumin",
        "tumor_size",
    ]


def test_sparse_observation_config_reduces_observed_density() -> None:
    common = {
        "n_clusters": 2,
        "min_visits": 4,
        "max_visits": 6,
        "hidden_size": 12,
        "latent_dim": 4,
        "attention_layers": 1,
        "feature_names": ["a", "b", "c", "d"],
    }
    dense = ClinicalTimeSeriesDatasetGeneratorConfig(**common)
    sparse = ClinicalTimeSeriesDatasetGeneratorConfig(
        **common,
        observation_rate_low=0.05,
        observation_rate_high=0.15,
        observation_severity_weight=0.0,
        observation_value_weight=0.0,
    )

    dense_dataset = ClinicalTimeSeriesDatasetGenerator(dense, mechanism_seed=101).simulate(
        n_patients=64,
        seed=201,
    )
    sparse_dataset = ClinicalTimeSeriesDatasetGenerator(sparse, mechanism_seed=101).simulate(
        n_patients=64,
        seed=201,
    )

    def observed_density(dataset: ClinicalTimeSeriesDataset) -> float:
        observed = 0.0
        total = 0.0
        for sample in dataset:
            aligned = sample.to_aligned()
            observed += float(aligned.mask.sum().item())
            total += float(aligned.mask.numel())
        return observed / total

    assert observed_density(sparse_dataset) < observed_density(dense_dataset)
    assert (
        sparse_dataset.metadata["generation_params"]["observation_rate_high"]
        == sparse.observation_rate_high
    )


def test_sequence_pool_masks_padding_visits() -> None:
    hidden_sequence = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    sequence_lengths = torch.tensor([2, 3])
    pool = SequencePool(hidden_size=3)
    with torch.no_grad():
        pool.score.weight.zero_()
        assert pool.score.bias is not None
        pool.score.bias.zero_()

    weights = pool.attention_weights(hidden_sequence, sequence_lengths)
    pooled = pool(hidden_sequence, sequence_lengths)

    assert weights.shape == (2, 4)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2))
    assert torch.allclose(weights[0, 2:], torch.zeros(2))
    assert torch.allclose(weights[1, 3:], torch.zeros(1))
    assert torch.allclose(pooled[0], hidden_sequence[0, :2].mean(dim=0))
    assert torch.allclose(pooled[1], hidden_sequence[1, :3].mean(dim=0))


def test_mtan_time_embedding_and_attention_shapes_without_nan() -> None:
    time_embedding = TimeEmbedding(embedding_dim=6, learn_embedding=False, frequency=10.0)
    query = time_embedding(torch.linspace(0.0, 1.0, 4).unsqueeze(0))
    key = time_embedding(torch.linspace(0.0, 1.0, 5).unsqueeze(0))
    value = torch.randn(1, 5, 3)
    attention = MultiTimeAttention(
        input_dim=3,
        hidden_dim=7,
        time_embedding_dim=6,
        n_heads=2,
        dropout=0.0,
    )

    output = attention(
        query=query,
        key=key,
        value=value,
        mask=torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool),
    )

    assert output.shape == (1, 4, 7)
    assert torch.isfinite(output).all()


def test_mtan_learned_time_embedding_is_linear_projection() -> None:
    time_embedding = TimeEmbedding(embedding_dim=6, learn_embedding=True, frequency=10.0)
    times = torch.linspace(0.0, 1.0, 4).reshape(2, 2)

    output = time_embedding(times)

    assert output.shape == (2, 2, 6)
    assert torch.isfinite(output).all()


def test_mtan_input_layer_uses_aligned_attention_shape() -> None:
    config = EncoderInputConfig(
        kind="mtan",
        hidden_dim=4,
        n_heads=2,
        num_ref_points=5,
        learn_time_embedding=True,
        time_embedding_dim=6,
    )
    layer = MTANInputLayer(input_size=3, config=config, dropout=0.0)
    layer.set_reference_time_range(0.0, 4.0)
    times = torch.tensor([[0.0, 1.0, 2.0], [0.0, 1.5, 3.0]])
    x = torch.randn(2, 3, 3)
    mask = torch.tensor(
        [
            [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        ]
    )

    output, query_times, sequence_lengths = layer(times=times, x=x, mask=mask)

    assert output.shape == (2, 5, 4)
    assert query_times.shape == (2, 5)
    assert torch.allclose(query_times[0], torch.linspace(0.0, 4.0, 5))
    assert torch.equal(sequence_lengths, torch.full((2,), 5))
    assert torch.isfinite(output).all()


def test_mtan2_input_layer_uses_per_feature_attention_shape() -> None:
    config = EncoderInputConfig(
        kind="mtan2",
        hidden_dim=4,
        n_heads=2,
        num_ref_points=5,
        learn_time_embedding=True,
        time_embedding_dim=6,
    )
    layer = MTAN2InputLayer(input_size=3, config=config, dropout=0.0)
    layer.set_reference_time_range(0.0, 4.0)
    times = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 4.0, 0.0]],
            [[0.0, 0.5, 0.0], [1.5, 1.0, 0.0], [3.0, 2.0, 0.0]],
        ]
    )
    x = torch.randn(2, 3, 3)
    mask = torch.tensor(
        [
            [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        ]
    )

    output, query_times, sequence_lengths = layer(times=times, x=x, mask=mask)

    assert output.shape == (2, 5, 12)
    assert query_times.shape == (2, 5)
    assert torch.allclose(query_times[0], torch.linspace(0.0, 4.0, 5))
    assert torch.equal(sequence_lengths, torch.full((2,), 5))
    assert torch.isfinite(output).all()
    assert torch.allclose(output[:, :, 8:12], torch.zeros_like(output[:, :, 8:12]))


def test_clustering_accuracy_matches_permuted_labels() -> None:
    metric = ClusteringAccuracy()
    metric.update(torch.tensor([1, 1, 0, 0]), torch.tensor([0, 0, 1, 1]))

    assert torch.allclose(metric.compute(), torch.tensor(1.0))


def make_architecture_config(
    *,
    encoder_input_kind: EncoderInputKind = "grud",
    encoder_mapping_kind: EncoderMappingKind = "gru",
    decoder_kind: DecoderKind = "gru",
    decoder_conditioning: DecoderConditioning = "initial_state",
    survival_head_hidden_layers: int = 0,
) -> ModelConfig:
    return ModelConfig(
        n_clusters=2,
        latent_dim=4,
        survival_head_hidden_layers=survival_head_hidden_layers,
        encoder=EncoderConfig(
            input=EncoderInputConfig(kind=encoder_input_kind, hidden_dim=8, n_heads=2),
            mapping=EncoderMappingConfig(kind=encoder_mapping_kind, hidden_dim=8, n_layers=1),
        ),
        decoder=DecoderConfig(
            kind=decoder_kind,
            conditioning=decoder_conditioning,
            hidden_dim=8,
            n_layers=1,
            n_heads=2,
        ),
    )


def assert_model_forward_shapes(model_config: ModelConfig) -> None:
    dataset = simulate_dataset(
        patients=4,
        n_clusters=2,
        min_visits=3,
        max_visits=4,
        hidden_size=12,
        latent_dim=4,
        attention_layers=2,
        seed=13,
    )
    model_dataset = (
        dataset.with_return_kind("compact")
        if model_config.encoder.input.kind == "mtan2"
        else dataset.with_return_kind("aligned")
    )
    batch = clinical_collate_fn(
        [model_dataset[0], model_dataset[1], model_dataset[2], model_dataset[3]]
    )
    model = TrailsSurvVaderModel(
        DataConfig(n_features=dataset.n_features),
        model_config,
    )
    model.set_feature_means(dataset.feature_means)
    if "feature_lengths" in batch:
        output = model(
            times=batch["times"],
            x=batch["x"],
            mask=batch["mask"],
            feature_lengths=batch["feature_lengths"],
        )
    else:
        output = model(
            times=batch["times"],
            x=batch["x"],
            mask=batch["mask"],
            delta_time=batch["delta_time"],
            sequence_lengths=batch["sequence_lengths"],
        )

    assert output.reconstruction.shape == batch["x"].shape
    assert output.cluster_logits.shape == (4, 2)
    assert output.cluster_probabilities.shape == (4, 2)
    assert torch.allclose(
        output.cluster_probabilities.sum(dim=-1),
        torch.ones(4),
        atol=1e-5,
    )
    assert output.weibull_shape.shape == (4, 2)
    assert torch.all(output.weibull_scale > 0)
    assert model.mixture_means.shape == (2, 4)
    assert model.mixture_log_variances.shape == (2, 4)


def make_loss_test_state(
    model_config: ModelConfig,
) -> tuple[TrailsSurvVaderModel, TrailsModelOutput, Batch]:
    dataset = simulate_dataset(
        patients=4,
        n_clusters=2,
        min_visits=3,
        max_visits=4,
        hidden_size=12,
        latent_dim=4,
        attention_layers=2,
        seed=31,
    )
    batch = clinical_collate_fn([dataset[0], dataset[1], dataset[2], dataset[3]])
    model = TrailsSurvVaderModel(
        DataConfig(n_features=dataset.n_features),
        model_config,
    )
    model.set_feature_means(dataset.feature_means)
    output = model(
        times=batch["times"],
        x=batch["x"],
        mask=batch["mask"],
        delta_time=batch["delta_time"],
        sequence_lengths=batch["sequence_lengths"],
    )
    return model, output, batch


def test_fixed_loss_matches_legacy_weighted_sum() -> None:
    loss_config = LossConfig(
        weighting="fixed",
        reconstruction_weight=1.3,
        survival_weight=0.4,
        cluster_weight=0.07,
    )
    model_config = make_architecture_config().model_copy(update={"loss": loss_config})
    model, output, batch = make_loss_test_state(model_config)

    loss = model.compute_loss(output, batch, include_vade_kl=True)
    reconstruction = masked_mse(output.reconstruction, batch["x"], batch["mask"])
    survival = weibull_mixture_negative_log_likelihood(
        output.cluster_logits,
        output.weibull_shape,
        output.weibull_scale,
        batch["survival_time"],
        batch["event"],
    )
    cluster = vade_kl_loss(
        output.latent,
        output.latent_mean,
        output.latent_log_variance,
        output.cluster_logits,
        model.mixture_logits,
        model.mixture_means,
        model.mixture_log_variances,
    )
    expected = (
        loss_config.reconstruction_weight * reconstruction
        + loss_config.survival_weight * survival
        + loss_config.cluster_weight * cluster
    )

    assert torch.allclose(loss.loss, expected)
    assert torch.allclose(loss.reconstruction_loss, reconstruction)
    assert torch.allclose(loss.survival_loss, survival)
    assert torch.allclose(loss.vade_kl_loss, cluster)
    assert "reconstruction_log_variance" not in dict(loss.items())


def test_uncertainty_loss_trains_log_variance_parameters() -> None:
    model_config = make_architecture_config().model_copy(
        update={
            "loss": LossConfig(
                weighting="uncertainty",
                reconstruction_weight=1.0,
                survival_weight=0.2,
                cluster_weight=0.05,
            )
        }
    )
    model, output, batch = make_loss_test_state(model_config)

    loss = model.compute_loss(output, batch, include_vade_kl=True)
    loss.loss.backward()

    assert set(model.loss_log_variances.keys()) == {"reconstruction", "survival", "vade_kl"}
    assert model.loss_log_variances["reconstruction"].grad is not None
    assert model.loss_log_variances["survival"].grad is not None
    assert model.loss_log_variances["vade_kl"].grad is not None
    assert {"reconstruction_log_variance", "survival_log_variance", "vade_kl_log_variance"} <= set(
        dict(loss.items())
    )


def test_uncertainty_warmup_skips_vade_kl_term_and_log_variance() -> None:
    model_config = make_architecture_config().model_copy(
        update={
            "loss": LossConfig(
                weighting="uncertainty",
                reconstruction_weight=1.0,
                survival_weight=0.2,
                cluster_weight=0.05,
            )
        }
    )
    model, output, batch = make_loss_test_state(model_config)

    loss = model.compute_loss(output, batch, include_vade_kl=False)
    reconstruction_log_variance = model.loss_log_variances["reconstruction"]
    survival_log_variance = model.loss_log_variances["survival"]
    reconstruction = masked_mse(output.reconstruction, batch["x"], batch["mask"])
    survival = weibull_mixture_negative_log_likelihood(
        output.cluster_logits,
        output.weibull_shape,
        output.weibull_scale,
        batch["survival_time"],
        batch["event"],
    )
    expected = (
        0.5 * torch.exp(-reconstruction_log_variance) * reconstruction
        + 0.5 * reconstruction_log_variance
        + 0.5 * torch.exp(-survival_log_variance) * survival
        + 0.5 * survival_log_variance
    )

    assert torch.allclose(loss.loss, expected)
    assert torch.allclose(loss.vade_kl_loss, torch.zeros_like(loss.vade_kl_loss))

    loss.loss.backward()
    assert model.loss_log_variances["reconstruction"].grad is not None
    assert model.loss_log_variances["survival"].grad is not None
    assert model.loss_log_variances["vade_kl"].grad is None


def test_uncertainty_loss_requires_positive_initial_weights() -> None:
    with pytest.raises(ValidationError, match="requires all initial weights > 0"):
        LossConfig(weighting="uncertainty", survival_weight=0.0)

    fixed = LossConfig(weighting="fixed", survival_weight=0.0, cluster_weight=0.0)

    assert fixed.survival_weight == 0.0
    assert fixed.cluster_weight == 0.0


@pytest.mark.parametrize("survival_head_hidden_layers", [0, 2])
def test_grud_model_forward_shapes(survival_head_hidden_layers: int) -> None:
    assert_model_forward_shapes(
        make_architecture_config(survival_head_hidden_layers=survival_head_hidden_layers)
    )


@pytest.mark.parametrize(
    ("decoder_kind", "decoder_conditioning"),
    [
        ("gru", "initial_state"),
        ("lstm", "initial_state"),
        ("gru", "concat_time"),
        ("lstm", "concat_time"),
        ("transformer", "concat_time"),
    ],
)
def test_model_forward_shapes_for_decoder_architectures(
    decoder_kind: DecoderKind,
    decoder_conditioning: DecoderConditioning,
) -> None:
    assert_model_forward_shapes(
        make_architecture_config(
            decoder_kind=decoder_kind,
            decoder_conditioning=decoder_conditioning,
        )
    )


@pytest.mark.parametrize(
    ("encoder_input_kind", "encoder_mapping_kind"),
    [
        ("grud", "gru"),
        ("mtan", "gru"),
        ("mtan2", "gru"),
        ("grud", "lstm"),
        ("mtan", "transformer"),
        ("mtan2", "transformer"),
    ],
)
def test_model_forward_shapes_for_encoder_architectures(
    encoder_input_kind: EncoderInputKind,
    encoder_mapping_kind: EncoderMappingKind,
) -> None:
    assert_model_forward_shapes(
        make_architecture_config(
            encoder_input_kind=encoder_input_kind,
            encoder_mapping_kind=encoder_mapping_kind,
        )
    )


def test_transformer_decoder_rejects_initial_state_conditioning() -> None:
    with pytest.raises(ValidationError, match="Transformer decoder only supports"):
        DecoderConfig(kind="transformer", conditioning="initial_state")


def test_model_config_rejects_stale_flat_architecture_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ModelConfig.model_validate({"encoder_hidden_dim": 8})

from typing import Literal

import pytest
import torch
from pydantic import ValidationError

from trails.config import (
    DataConfig,
    DecoderConfig,
    EncoderConfig,
    EncoderInputConfig,
    EncoderMappingConfig,
    ModelConfig,
)
from trails.data import ClinicalTimeSeriesDataset, clinical_collate_fn, make_clinical_sample
from trails.model import SequencePool, TrailsSurvVaderModel
from trails_simulate import generate_clinical_time_series_dataset

EncoderInputKind = Literal["grud", "mtan"]
EncoderMappingKind = Literal["gru", "lstm", "transformer"]
DecoderKind = Literal["gru", "lstm", "transformer"]
DecoderConditioning = Literal["initial_state", "concat_time"]


def test_clinical_dataset_and_collate_shapes() -> None:
    dataset = generate_clinical_time_series_dataset(
        n_patients=6,
        n_clusters=2,
        min_visits=3,
        max_visits=5,
        hidden_size=12,
        latent_dim=4,
        attention_layers=2,
        seed=7,
    )
    sample = dataset[0]
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


def test_unlabeled_dataset_collates_without_cluster_labels() -> None:
    labeled = generate_clinical_time_series_dataset(
        n_patients=4,
        n_clusters=2,
        min_visits=3,
        max_visits=4,
        hidden_size=12,
        latent_dim=4,
        attention_layers=2,
        seed=19,
    )
    samples = [
        make_clinical_sample(
            times=sample.times,
            x=sample.x,
            mask=sample.mask,
            delta_time=sample.delta_time,
            survival_time=sample.survival_time,
            event=sample.event,
        )
        for sample in labeled
    ]
    unlabeled = ClinicalTimeSeriesDataset(samples, feature_names=labeled.feature_names)
    batch = clinical_collate_fn([unlabeled[0], unlabeled[1]])

    assert not unlabeled.has_cluster_labels
    assert "cluster_label" not in batch


def test_dataset_rejects_mixed_cluster_label_availability() -> None:
    labeled = generate_clinical_time_series_dataset(
        n_patients=4,
        n_clusters=2,
        min_visits=3,
        max_visits=4,
        hidden_size=12,
        latent_dim=4,
        attention_layers=2,
        seed=23,
    )
    unlabeled_sample = make_clinical_sample(
        times=labeled[1].times,
        x=labeled[1].x,
        mask=labeled[1].mask,
        delta_time=labeled[1].delta_time,
        survival_time=labeled[1].survival_time,
        event=labeled[1].event,
    )

    with pytest.raises(ValueError, match="cannot mix labeled and unlabeled"):
        ClinicalTimeSeriesDataset(
            [labeled[0], unlabeled_sample],
            feature_names=labeled.feature_names,
        )


def test_simulation_has_asynchronous_masks_and_valid_delta_time() -> None:
    dataset = generate_clinical_time_series_dataset(
        n_patients=64,
        n_clusters=2,
        min_visits=4,
        max_visits=5,
        hidden_size=12,
        latent_dim=4,
        attention_layers=2,
        censoring_rate=0.3,
        seed=11,
    )
    sequence_lengths = {int(sample.times.shape[0]) for sample in dataset}
    assert len(sequence_lengths) > 1

    partial_visit_exists = any(
        bool(
            torch.any((sample.mask.sum(dim=1) > 0) & (sample.mask.sum(dim=1) < dataset.n_features))
        )
        for sample in dataset
    )
    assert partial_visit_exists

    for sample in dataset:
        assert sample.times.max() <= sample.survival_time
        assert torch.all(sample.delta_time >= 0)
        for step in range(1, int(sample.times.shape[0])):
            missing_previously = sample.mask[step - 1] == 0
            assert torch.all(
                sample.delta_time[step][missing_previously]
                >= sample.delta_time[step - 1][missing_previously]
            )

    observed_values = torch.cat([sample.x[sample.mask > 0] for sample in dataset])
    assert observed_values.dtype.is_floating_point
    assert not torch.allclose(observed_values, observed_values.round())

    event_rate = torch.stack([sample.event for sample in dataset]).float().mean()
    assert 0.45 <= float(event_rate) <= 0.95


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
    dataset = generate_clinical_time_series_dataset(
        n_patients=4,
        n_clusters=2,
        min_visits=3,
        max_visits=4,
        hidden_size=12,
        latent_dim=4,
        attention_layers=2,
        seed=13,
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
        ("grud", "lstm"),
        ("mtan", "transformer"),
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

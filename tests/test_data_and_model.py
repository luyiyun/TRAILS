import torch

from trails.config import DataConfig, ModelConfig
from trails.data import clinical_collate_fn
from trails.model import TrailsSurvVaderModel
from trails_simulate import generate_clinical_time_series_dataset


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
    assert {"latent_z", "cluster_means", "survival_coefficients", "generation_params"} <= set(
        dataset.metadata
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


def test_grud_model_forward_shapes() -> None:
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
        ModelConfig(
            n_clusters=2,
            latent_dim=4,
            encoder_hidden_dim=8,
            decoder_hidden_dim=8,
        ),
    )
    model.set_feature_means(dataset.feature_means)
    output = model(
        batch["x"],
        batch["mask"],
        batch["delta_time"],
        batch["sequence_lengths"],
    )

    assert output.reconstruction.shape == batch["x"].shape
    assert output.cluster_logits.shape == (4, 2)
    assert output.weibull_shape.shape == (4, 2)
    assert torch.all(output.weibull_scale > 0)

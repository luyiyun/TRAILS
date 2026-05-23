from pathlib import Path

import torch

from trails.config import (
    DataConfig,
    DecoderConfig,
    EncoderConfig,
    EncoderInputConfig,
    EncoderMappingConfig,
    ModelConfig,
    TrailsConfig,
    TrainerConfig,
)
from trails.data import ClinicalTimeSeriesDataset, make_clinical_sample
from trails.estimator import TrailsEstimator
from trails_simulate import (
    ClinicalTimeSeriesDatasetGenerator,
    ClinicalTimeSeriesDatasetGeneratorConfig,
)


def tiny_config(n_features: int) -> TrailsConfig:
    return TrailsConfig(
        data=DataConfig(n_features=n_features),
        model=ModelConfig(
            n_clusters=2,
            latent_dim=4,
            encoder=EncoderConfig(
                input=EncoderInputConfig(hidden_dim=8),
                mapping=EncoderMappingConfig(hidden_dim=8),
            ),
            decoder=DecoderConfig(hidden_dim=8),
        ),
        trainer=TrainerConfig(
            max_epochs=1,
            warmup_epochs=1,
            batch_size=4,
            gmm_init_iters=2,
            valid_size=0.0,
        ),
        seed=13,
    )


def strip_cluster_labels(data: ClinicalTimeSeriesDataset) -> ClinicalTimeSeriesDataset:
    aligned_samples = [sample.to_aligned() for sample in data.samples]
    return ClinicalTimeSeriesDataset(
        [
            make_clinical_sample(
                times=sample.times,
                x=sample.x,
                mask=sample.mask,
                delta_time=sample.delta_time,
                survival_time=sample.survival_time,
                event=sample.event,
            )
            for sample in aligned_samples
        ],
        feature_names=data.feature_names,
        description=data.description,
        metadata=data.metadata,
    )


def simulate_dataset(seed: int) -> ClinicalTimeSeriesDataset:
    config = ClinicalTimeSeriesDatasetGeneratorConfig(
        n_clusters=2,
        min_visits=3,
        max_visits=5,
        hidden_size=12,
        latent_dim=4,
        attention_layers=2,
    )
    return ClinicalTimeSeriesDatasetGenerator(config, mechanism_seed=seed).simulate(
        n_patients=8,
        seed=seed,
    )


def test_estimator_fit_predict_test() -> None:
    data = simulate_dataset(seed=13)
    estimator = TrailsEstimator(tiny_config(data.n_features)).fit(data)
    predictions = estimator.predict(data)
    probabilities = estimator.predict_proba(data)
    metrics = estimator.test(data)

    assert predictions.shape == (8,)
    assert probabilities.shape == (8, 2)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(8), atol=1e-5)
    assert set(predictions.tolist()) <= {0, 1}
    assert not any(key.startswith("val_") for key in estimator.history[-1])
    assert metrics["loss"] > 0
    assert "cindex" in metrics
    assert "acc" in metrics
    assert "ari" in metrics
    assert "nmi" in metrics
    assert "vade_kl_loss" in metrics
    assert "cluster_empty_count" in metrics
    assert "cluster_min_fraction" in metrics
    assert "cluster_max_fraction" in metrics
    assert "cluster_entropy" in metrics
    assert 0.0 <= metrics["cluster_entropy"] <= 1.0


def test_estimator_latent_diagnostics_exports_labels_and_embeddings() -> None:
    data = simulate_dataset(seed=23)
    estimator = TrailsEstimator(tiny_config(data.n_features)).fit(data)
    diagnostics = estimator.latent_diagnostics(data)

    assert diagnostics["z"].shape == (8, 4)
    assert diagnostics["cluster_probabilities"].shape == (8, 2)
    assert diagnostics["pred_cluster"].shape == (8,)
    assert diagnostics["sample_index"].tolist() == list(range(8))
    assert "true_cluster" in diagnostics
    assert diagnostics["true_cluster"].shape == (8,)
    assert set(diagnostics["pred_cluster"].tolist()) <= {0, 1}
    assert torch.allclose(
        diagnostics["cluster_probabilities"].sum(dim=-1),
        torch.ones(8),
        atol=1e-5,
    )


def test_estimator_fit_without_internal_validation_skips_validation_metrics() -> None:
    data = simulate_dataset(seed=31)
    estimator = TrailsEstimator(tiny_config(data.n_features)).fit(data)

    assert "valid" not in estimator.history[-1]


def test_unlabeled_data_skips_cluster_metrics() -> None:
    labeled = simulate_dataset(seed=37)
    data = strip_cluster_labels(labeled)
    estimator = TrailsEstimator(tiny_config(data.n_features)).fit(data)
    metrics = estimator.test(data)

    assert "ari" not in metrics
    assert "acc" not in metrics
    assert "nmi" not in metrics
    assert "cluster_empty_count" in metrics
    assert "cluster_entropy" in metrics
    assert "valid" not in estimator.history[-1]


def test_estimator_save_load(tmp_path: Path) -> None:
    data = simulate_dataset(seed=17)
    estimator = TrailsEstimator(tiny_config(data.n_features)).fit(data)
    path = tmp_path / "trails.pt"
    estimator.save(path)
    loaded = TrailsEstimator.load(path)

    assert torch.equal(estimator.predict(data), loaded.predict(data))

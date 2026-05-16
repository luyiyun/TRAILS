from pathlib import Path

import torch

from trails.config import DataConfig, ModelConfig, TrailsConfig, TrainerConfig
from trails.data import ClinicalTimeSeriesDataset, make_clinical_sample
from trails.estimator import TrailsEstimator
from trails_simulate import generate_clinical_time_series_dataset


def tiny_config(n_features: int) -> TrailsConfig:
    return TrailsConfig(
        data=DataConfig(n_features=n_features),
        model=ModelConfig(
            n_clusters=2,
            encoder_hidden_dim=8,
            decoder_hidden_dim=8,
            latent_dim=4,
        ),
        trainer=TrainerConfig(max_epochs=1, warmup_epochs=1, batch_size=4, gmm_init_iters=2),
        seed=13,
    )


def strip_cluster_labels(data: ClinicalTimeSeriesDataset) -> ClinicalTimeSeriesDataset:
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
            for sample in data
        ],
        feature_names=data.feature_names,
        description=data.description,
        metadata=data.metadata,
    )


def test_estimator_fit_predict_test() -> None:
    data = generate_clinical_time_series_dataset(
        n_patients=8,
        n_clusters=2,
        min_visits=3,
        max_visits=5,
        hidden_size=12,
        latent_dim=4,
        attention_layers=2,
        seed=13,
    )
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
    assert "c_index" in metrics
    assert "ari" in metrics
    assert "nmi" in metrics
    assert "vade_kl_loss" in metrics


def test_estimator_fit_with_validation_reports_validation_cluster_metrics() -> None:
    data = generate_clinical_time_series_dataset(
        n_patients=8,
        n_clusters=2,
        min_visits=3,
        max_visits=5,
        hidden_size=12,
        latent_dim=4,
        attention_layers=2,
        seed=31,
    )
    estimator = TrailsEstimator(tiny_config(data.n_features)).fit(data, validation_data=data)

    assert "val_loss" in estimator.history[-1]
    assert "val_ari" in estimator.history[-1]
    assert "val_nmi" in estimator.history[-1]


def test_unlabeled_data_skips_cluster_metrics() -> None:
    labeled = generate_clinical_time_series_dataset(
        n_patients=8,
        n_clusters=2,
        min_visits=3,
        max_visits=5,
        hidden_size=12,
        latent_dim=4,
        attention_layers=2,
        seed=37,
    )
    data = strip_cluster_labels(labeled)
    estimator = TrailsEstimator(tiny_config(data.n_features)).fit(data, validation_data=data)
    metrics = estimator.test(data)

    assert "ari" not in metrics
    assert "nmi" not in metrics
    assert "val_ari" not in estimator.history[-1]
    assert "val_nmi" not in estimator.history[-1]


def test_estimator_save_load(tmp_path: Path) -> None:
    data = generate_clinical_time_series_dataset(
        n_patients=8,
        n_clusters=2,
        min_visits=3,
        max_visits=5,
        hidden_size=12,
        latent_dim=4,
        attention_layers=2,
        seed=17,
    )
    estimator = TrailsEstimator(tiny_config(data.n_features)).fit(data)
    path = tmp_path / "trails.pt"
    estimator.save(path)
    loaded = TrailsEstimator.load(path)

    assert torch.equal(estimator.predict(data), loaded.predict(data))

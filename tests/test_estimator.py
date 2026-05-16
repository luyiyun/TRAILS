from pathlib import Path

import torch

from trails.config import DataConfig, EstimatorConfig, ModelConfig, TrainerConfig
from trails.estimator import TrailsEstimator
from trails_simulate import generate_clinical_time_series_dataset


def tiny_config(n_features: int) -> EstimatorConfig:
    return EstimatorConfig(
        data=DataConfig(n_features=n_features),
        model=ModelConfig(
            n_clusters=2,
            encoder_hidden_dim=8,
            decoder_hidden_dim=8,
            latent_dim=4,
        ),
        trainer=TrainerConfig(max_epochs=1, batch_size=4),
        seed=13,
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
    metrics = estimator.test(data)

    assert predictions.shape == (8,)
    assert set(predictions.tolist()) <= {0, 1}
    assert metrics["loss"] > 0
    assert "c_index" in metrics


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

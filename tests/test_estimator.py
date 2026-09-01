import logging
from pathlib import Path
from typing import Literal

import pytest
import torch

from trails import (
    ClinicalTimeSeriesDataset,
    DataConfig,
    DecoderConfig,
    EncoderConfig,
    EncoderInputConfig,
    EncoderMappingConfig,
    ModelConfig,
    TrailsConfig,
    TrailsEstimator,
    TrailsPrediction,
    TrainerConfig,
)
from trails.data import make_clinical_sample
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
            batch_size=None,
            gmm_init_iters=2,
            valid_size=0.0,
        ),
        seed=13,
    )


def tiny_config_with_validation(n_features: int, valid_size: float = 0.25) -> TrailsConfig:
    config = tiny_config(n_features)
    return config.model_copy(
        update={"trainer": config.trainer.model_copy(update={"valid_size": valid_size})}
    )


def tiny_mtan_config(n_features: int, kind: Literal["mtan", "mtan2"] = "mtan") -> TrailsConfig:
    return TrailsConfig(
        data=DataConfig(n_features=n_features),
        model=ModelConfig(
            n_clusters=2,
            latent_dim=4,
            encoder=EncoderConfig(
                input=EncoderInputConfig(
                    kind=kind,
                    hidden_dim=4,
                    n_heads=2,
                    num_ref_points=5,
                ),
                mapping=EncoderMappingConfig(kind="gru", hidden_dim=8),
            ),
            decoder=DecoderConfig(hidden_dim=8),
        ),
        trainer=TrainerConfig(
            max_epochs=1,
            warmup_epochs=1,
            batch_size=None,
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
    prediction = estimator.predict(data)
    predictions = prediction.predict()
    probabilities = prediction.predict_proba()
    survival = prediction.survival([0.0, 7.0, 14.0])
    metrics = estimator.test(data)

    assert predictions.shape == (8,)
    assert probabilities.shape == (8, 2)
    assert prediction.weibull_shape.shape == (8,)
    assert prediction.weibull_scale.shape == (8,)
    risk_score = prediction.risk_score(28.0)
    assert risk_score.shape == (8,)
    assert torch.allclose(risk_score, 1.0 - prediction.survival([28.0]).squeeze(1))
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(8), atol=1e-5)
    assert survival.shape == (8, 3)
    assert torch.allclose(survival[:, 0], torch.ones(8), atol=1e-6)
    assert bool(((survival[:, :-1] - survival[:, 1:]) >= 0).all())
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


def test_estimator_uncertainty_weighting_can_disable_survival_loss() -> None:
    data = simulate_dataset(seed=19)
    config = tiny_config(data.n_features)
    config = config.model_copy(
        update={
            "model": config.model.model_copy(
                update={"loss": config.model.loss.model_copy(update={"survival_weight": 0.0})}
            )
        }
    )
    estimator = TrailsEstimator(config).fit(data)

    assert "survival" not in estimator.model.loss_log_variances
    assert estimator.history[-1]["train"]["survival_loss_weight"] == 0.0
    assert "survival_log_variance" not in estimator.history[-1]["train"]
    assert estimator.history[-1]["train"]["reconstruction_loss_weight"] > 0.0
    assert estimator.history[-1]["train"]["vade_kl_loss_weight"] > 0.0


@pytest.mark.parametrize("kind", ["mtan", "mtan2"])
def test_mtan_estimator_fit_and_predict(kind: Literal["mtan", "mtan2"]) -> None:
    data = simulate_dataset(seed=37)
    config = tiny_mtan_config(data.n_features, kind=kind)
    estimator = TrailsEstimator(config).fit(data)

    assert estimator.predict(data).predict().shape == (8,)


def test_estimator_latent_diagnostics_exports_labels_and_embeddings() -> None:
    data = simulate_dataset(seed=23)
    estimator = TrailsEstimator(tiny_config(data.n_features)).fit(data)
    diagnostics = estimator.predict(data).latent_diagnostics()

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


def test_fit_with_explicit_validation_data_warns_and_uses_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    data = simulate_dataset(seed=47)
    train_data, validation_data = data.split([0.75, 0.25], seed=11)
    config = tiny_config_with_validation(data.n_features)
    config = config.model_copy(
        update={
            "trainer": config.trainer.model_copy(update={"early_stopping_monitor": "survival_loss"})
        }
    )

    caplog.set_level(logging.WARNING, logger="trails.trainer")
    estimator = TrailsEstimator(config).fit(train_data, validation_data=validation_data)

    assert "Explicit validation_data was provided" in caplog.text
    assert "trainer.valid_size=0.25 is ignored" in caplog.text
    assert "valid" in estimator.history[-1]
    assert estimator.history[-1].get("best_monitor") == "valid/survival_loss"
    assert estimator.history[-1].get("best_monitor_value") == pytest.approx(
        estimator.history[-1]["valid"]["survival_loss"]
    )


def test_fit_internal_validation_split_remains_default_behavior() -> None:
    data = simulate_dataset(seed=53)
    estimator = TrailsEstimator(tiny_config_with_validation(data.n_features)).fit(data)

    assert "valid" in estimator.history[-1]


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
    prediction_path = tmp_path / "prediction.pt"
    estimator.predict(data).save(prediction_path)
    loaded_prediction = TrailsPrediction.load(prediction_path)

    assert torch.equal(estimator.predict(data).predict(), loaded.predict(data).predict())
    assert torch.equal(
        estimator.predict(data).risk_score(28.0),
        loaded_prediction.risk_score(28.0),
    )

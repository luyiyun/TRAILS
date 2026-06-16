import logging
import math
from pathlib import Path
from typing import Literal

import pytest
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
from trails.estimator import (
    TrailsEstimator,
    observed_time_range,
    score_k_selection_rows,
    selected_k_from_selection_metrics,
    selection_metrics_to_rows,
)
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


@pytest.mark.parametrize("kind", ["mtan", "mtan2"])
def test_mtan_estimator_sets_training_reference_time_grid(kind: Literal["mtan", "mtan2"]) -> None:
    data = simulate_dataset(seed=37)
    config = tiny_mtan_config(data.n_features, kind=kind)
    estimator = TrailsEstimator(config).fit(data)
    min_time, max_time = observed_time_range(data)
    reference_times = estimator.model.reference_times
    assert reference_times is not None
    expected = torch.linspace(
        min_time,
        max_time,
        config.model.encoder.input.num_ref_points,
        dtype=reference_times.dtype,
        device=reference_times.device,
    )

    assert torch.allclose(reference_times, expected)
    saved_reference_times = reference_times.clone()
    estimator.predict(data)
    prediction_reference_times = estimator.model.reference_times
    assert prediction_reference_times is not None
    assert torch.allclose(prediction_reference_times, saved_reference_times)


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


def test_estimator_selection_metrics_include_cindex_and_bic() -> None:
    data = simulate_dataset(seed=29)
    estimator = TrailsEstimator(tiny_config(data.n_features)).fit(data)
    metrics = estimator.selection_metrics(data)

    for name in ["cindex", "bic", "mean_nll", "n_parameters"]:
        assert name in metrics
        assert math.isfinite(metrics[name])
    assert metrics["n_parameters"] == pytest.approx(17.0)
    assert "cluster_empty_count" in metrics
    assert "cluster_entropy" in metrics


def test_select_n_clusters_scores_ranks_and_can_inherit_best(tmp_path: Path) -> None:
    data = simulate_dataset(seed=41)
    estimator = TrailsEstimator(tiny_config(data.n_features))
    result = estimator.select_n_clusters(
        data,
        candidate_clusters=(2, 3),
        valid_fraction=0.25,
        inherit_best=True,
        result_dir=tmp_path / "k_selection",
    )

    selected_k = selected_k_from_selection_metrics(result)
    rows = selection_metrics_to_rows(result)
    assert selected_k in {2, 3}
    assert estimator.config.model.n_clusters == selected_k
    assert estimator.history
    assert set(result["bic"]) == {"2", "3"}
    assert {int(row["n_clusters"]) for row in rows} == {2, 3}
    assert [int(row["rank"]) for row in rows] == [1, 2]
    for row in rows:
        assert 0.0 <= float(row["bic_norm"]) <= 1.0
        assert math.isfinite(float(row["selection_score"]))
    assert (tmp_path / "k_selection" / "selection_metrics.csv").exists()
    assert (tmp_path / "k_selection" / "selection_metrics.json").exists()
    for n_clusters in [2, 3]:
        candidate_dir = tmp_path / "k_selection" / f"k{n_clusters}"
        assert (candidate_dir / "model.pt").exists()
        assert (candidate_dir / "history.json").exists()
        assert (candidate_dir / "metrics.json").exists()
        assert (candidate_dir / "config.json").exists()


def test_select_n_clusters_reuses_one_validation_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = simulate_dataset(seed=43)
    config = tiny_config_with_validation(data.n_features)
    split_calls: list[tuple[int, tuple[float, ...], int]] = []
    original_split = ClinicalTimeSeriesDataset.split

    def tracking_split(
        self: ClinicalTimeSeriesDataset,
        fraction: list[float],
        seed: int = 0,
    ) -> list[ClinicalTimeSeriesDataset]:
        split_calls.append((len(self), tuple(fraction), seed))
        return original_split(self, fraction, seed)

    monkeypatch.setattr(ClinicalTimeSeriesDataset, "split", tracking_split)

    TrailsEstimator(config).select_n_clusters(data, candidate_clusters=(2, 3))

    assert split_calls == [(8, (0.75, 0.25), config.seed)]


def test_k_selection_bic_normalization_handles_equal_bic() -> None:
    rows = score_k_selection_rows(
        [
            {"n_clusters": 2, "cindex": 0.3, "bic": 10.0, "mean_nll": 1.0},
            {"n_clusters": 3, "cindex": 0.7, "bic": 10.0, "mean_nll": 1.1},
        ]
    )

    assert [float(row["bic_norm"]) for row in rows] == [0.0, 0.0]
    assert int(rows[0]["n_clusters"]) == 3
    assert int(rows[0]["rank"]) == 1


def test_fit_with_explicit_validation_data_warns_and_uses_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    data = simulate_dataset(seed=47)
    train_data, validation_data = data.split([0.75, 0.25], seed=11)
    config = tiny_config_with_validation(data.n_features)

    caplog.set_level(logging.WARNING, logger="trails.trainer")
    estimator = TrailsEstimator(config).fit(train_data, validation_data=validation_data)

    assert "Explicit validation_data was provided" in caplog.text
    assert "trainer.valid_size=0.25 is ignored" in caplog.text
    assert "valid" in estimator.history[-1]


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

    assert torch.equal(estimator.predict(data), loaded.predict(data))

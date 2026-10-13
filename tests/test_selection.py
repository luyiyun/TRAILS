from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from matplotlib import pyplot as plt
from pydantic import ValidationError

from trails import (
    ClinicalTimeSeriesDataset,
    ClusterNumberEvaluationResult,
    ClusterNumberEvaluator,
    ClusterNumberEvaluatorConfig,
    TrailsConfig,
    TrailsEstimator,
    TrailsPrediction,
)


def _dataset(*, events: bool = True) -> ClinicalTimeSeriesDataset:
    patients = pd.DataFrame(
        {
            "patient_id": [str(i) for i in range(16)],
            "survival_time": np.arange(1, 17, dtype=float),
            "event": np.arange(16) % 2 if events else np.zeros(16),
        }
    )
    observations = pd.DataFrame(
        {
            "patient_id": np.repeat(patients["patient_id"], 3),
            "time": np.tile([0.0, 1.0, 3.0], 16),
            "x1": np.linspace(0, 2, 48),
            "x2": np.sin(np.arange(48)),
        }
    )
    return ClinicalTimeSeriesDataset.from_dataframes(
        patients=patients, observations=observations, use_features=["x1", "x2"]
    )


def _config() -> TrailsConfig:
    return TrailsConfig.model_validate(
        {
            "data": {"n_features": 2},
            "model": {
                "latent_dim": 2,
                "encoder": {"input": {"hidden_dim": 4}, "mapping": {"hidden_dim": 4}},
                "decoder": {"hidden_dim": 4},
            },
            "trainer": {
                "device": "cpu",
                "max_epochs": 2,
                "warmup_epochs": 1,
                "gmm_init_iters": 2,
                "batch_size": 16,
                "valid_size": 0.25,
            },
        }
    )


def test_evaluator_config_supports_single_seed_and_json_round_trip() -> None:
    config = ClusterNumberEvaluatorConfig.model_validate({"candidates": (2, 3, 4), "seeds": 17})
    assert config.seeds == (17,)
    assert ClusterNumberEvaluatorConfig.model_validate_json(config.model_dump_json()) == config


@pytest.mark.parametrize(
    "kwargs",
    [
        {"candidates": ()},
        {"candidates": (1, 2)},
        {"candidates": (2, 2)},
        {"candidates": (2,), "seeds": ()},
        {"candidates": (2,), "seeds": (1, 1)},
        {"candidates": (2,), "selection_rule": "one_standard_error"},
    ],
)
def test_evaluator_config_rejects_invalid_settings(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ClusterNumberEvaluatorConfig.model_validate(kwargs)


@pytest.mark.parametrize(
    ("ks", "seeds", "validation"),
    [
        ((2,), (11,), False),
        ((2, 3), (11,), True),
        ((2, 3), (11, 12), True),
        ((2,), (11, 12), False),
    ],
)
def test_fit_reload_and_lazy_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ks: tuple[int, ...],
    seeds: tuple[int, ...],
    validation: bool,
) -> None:
    data, extra = _dataset(), _dataset(events=False)
    extra_path = tmp_path / "extra.pt"
    extra.save(extra_path)
    root = tmp_path / "fit"
    original_load = ClinicalTimeSeriesDataset.load
    loaded: list[Path] = []

    def delayed_load(path: str | Path) -> ClinicalTimeSeriesDataset:
        assert len(list(root.glob("models/seed-*/k*/model.pt"))) == len(ks) * len(seeds)
        loaded.append(Path(path))
        return original_load(path)

    monkeypatch.setattr(ClinicalTimeSeriesDataset, "load", delayed_load)
    evaluator = ClusterNumberEvaluator(
        ks, seeds=seeds, estimator_config=_config(), min_cluster_fraction=0.49
    )
    result = evaluator.fit(
        data,
        validation_data=data if validation else None,
        evaluation_data={"test": extra_path},
        result_dir=root,
    )
    assert isinstance(result, ClusterNumberEvaluationResult)
    assert loaded == [extra_path]
    assert [(row["seed"], row["n_clusters"]) for row in evaluator.candidates] == [
        (seed, k) for seed in seeds for k in ks
    ]
    assert evaluator.run_metrics is result.run_metrics
    assert len(result.run_metrics) == len(ks) * len(seeds) * (3 if validation else 2)
    assert result.run_metrics.loc[result.run_metrics["dataset"] == "test", "cindex"].isna().all()
    assert not {"selected_k", "seed_winners", "rank"} & result.to_payload().keys()
    assert "rank" not in result.run_metrics
    assert result.stability_pairs.empty is (not validation or len(seeds) == 1)
    assert bool(result.stability_reason) is (not validation or len(seeds) == 1)
    model = evaluator.load_estimator(seed=seeds[0], n_clusters=ks[0])
    assert isinstance(model, TrailsEstimator)
    assert model.config.trainer.valid_size == 0
    assert ("valid" in model.history[-1]) is validation
    pred = evaluator.load_prediction(seed=seeds[0], n_clusters=ks[0], dataset="test")
    assert isinstance(pred, TrailsPrediction)
    assert torch.equal(pred.predict_proba(), model.predict(extra).predict_proba())
    assert not evaluator.load_history(seed=seeds[0], n_clusters=ks[0]).empty
    with pytest.raises(FileExistsError):
        evaluator.fit(data, result_dir=root)

    moved = tmp_path / "moved"
    shutil.move(root, moved)
    restored = ClusterNumberEvaluator.load(moved)
    assert restored.config == evaluator.config
    assert restored.candidates == evaluator.candidates
    pd.testing.assert_frame_equal(restored.run_metrics, result.run_metrics, check_dtype=False)
    pd.testing.assert_frame_equal(restored.k_summary, result.k_summary, check_dtype=False)
    assert torch.equal(
        restored.load_prediction(seed=seeds[0], n_clusters=ks[0], dataset="test").predict_proba(),
        pred.predict_proba(),
    )
    restored.load_estimator(seed=seeds[0], n_clusters=ks[0])
    shutil.rmtree(moved / "models")
    aggregate = ClusterNumberEvaluator.load(moved)
    fig = aggregate.plot_metrics(dataset="validation" if validation else "train")
    assert len(fig.axes) == 5
    plt.close(fig)
    with pytest.raises(FileNotFoundError):
        aggregate.load_estimator(seed=seeds[0], n_clusters=ks[0])
    with pytest.raises(FileNotFoundError):
        aggregate.load_prediction(seed=seeds[0], n_clusters=ks[0], dataset="test")
    with pytest.raises(FileNotFoundError):
        aggregate.load_history(seed=seeds[0], n_clusters=ks[0])


def test_summarize_preserves_empty_clusters_and_dataset_scores() -> None:
    rows = pd.DataFrame(
        {
            "seed": [1, 1, 1, 1],
            "n_clusters": [2, 3, 2, 3],
            "dataset": ["train", "train", "test", "test"],
            "cindex": [0.7, 0.6, 0.5, 0.8],
            "latent_mixture_bic": [1.0, 2.0, 20.0, 10.0],
            "cluster_empty_count": [0.0, 1.0, 0.0, 1.0],
            "cluster_min_fraction": [0.3, 0.0, 0.3, 0.0],
        }
    )
    evaluator = ClusterNumberEvaluator((2, 3), seeds=1, min_cluster_fraction=0.05)
    result = evaluator.summarize_completed(
        rows, pd.DataFrame(columns=["n_clusters", "seed_a", "seed_b", "ari"])
    )
    assert result.run_metrics["latent_mixture_bic_normalized"].tolist() == [0, 1, 1, 0]
    np.testing.assert_allclose(
        result.run_metrics["selection_score"], np.hypot(rows["cindex"], [1, 0, 0, 1])
    )
    assert result.run_metrics["passes_occupancy"].tolist() == [True, False, True, False]
    assert not hasattr(result, "selected_k")
    figure = evaluator.plot_metrics(["cindex", "latent_mixture_bic"], dataset="test")
    np.testing.assert_allclose(
        np.asarray(figure.axes[0].lines[0].get_ydata(), dtype=float), [0.5, 0.8]
    )
    plt.close(figure)
    with pytest.raises(ValueError, match="缺少绘图指标"):
        evaluator.plot_metrics(["unknown"], dataset="test")

    with pytest.raises(ValueError, match="仅描述独立validation"):
        evaluator.plot_metrics(["mean_pairwise_ari"], dataset="test")

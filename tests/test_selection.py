from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest
import torch
from pydantic import ValidationError

import trails.selection as selection_module
from trails import (
    ClusterNumberSelectionResult,
    ClusterNumberSelector,
    ClusterNumberSelectorConfig,
    TrailsEstimator,
)


def test_selector_config_supports_single_seed_and_json_round_trip() -> None:
    config = ClusterNumberSelectorConfig.model_validate(
        {
            "candidates": (2, 3, 4),
            "seeds": 17,
            "estimator": {"model": {"n_clusters": 4}},
        }
    )

    assert config.seeds == (17,)
    assert config.estimator.model.n_clusters == 4
    assert ClusterNumberSelectorConfig.model_validate_json(config.model_dump_json()) == config


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"candidates": ()}, "at least one K"),
        ({"candidates": (1, 2)}, "greater than 1"),
        ({"candidates": (2, 2)}, "must be unique"),
        ({"candidates": (2,), "seeds": ()}, "at least one value"),
        ({"candidates": (2,), "seeds": (1, 1)}, "must be unique"),
        (
            {"candidates": (2,), "seeds": 1, "min_mean_pairwise_ari": 0.75},
            "requires at least two seeds",
        ),
    ],
)
def test_selector_config_rejects_invalid_settings(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        ClusterNumberSelectorConfig.model_validate(kwargs)


def test_selection_result_exposes_estimators_for_selected_k() -> None:
    config = ClusterNumberSelectorConfig(candidates=(2, 3), seeds=(11, 12))
    estimators = {
        (11, 2): TrailsEstimator(),
        (11, 3): TrailsEstimator(),
        (12, 3): TrailsEstimator(),
    }
    result = ClusterNumberSelectionResult(
        config=config,
        selected_k=3,
        run_metrics=pd.DataFrame(),
        stability_pairs=pd.DataFrame(),
        k_summary=pd.DataFrame(),
        seed_winners={11: 2, 12: 3},
        estimators=estimators,
    )

    assert result.selected_estimators == {11: estimators[(11, 3)], 12: estimators[(12, 3)]}


def test_selector_runs_shared_split_and_selects_across_seeds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_fit(estimator: TrailsEstimator, *args: Any, **kwargs: Any) -> TrailsEstimator:
        return estimator

    def fake_metrics(
        selector: ClusterNumberSelector, estimator: TrailsEstimator, data: object
    ) -> dict[str, float]:
        n_clusters = estimator.config.model.n_clusters
        return {
            "cindex": 0.8 if n_clusters == 2 else 0.5,
            "latent_mixture_bic": float(n_clusters),
            "cluster_min_fraction": 0.4,
            "cluster_empty_count": 0.0,
        }

    monkeypatch.setattr(selection_module.TrailsEstimator, "fit", fake_fit)
    monkeypatch.setattr(ClusterNumberSelector, "_calculate_candidate_metrics", fake_metrics)
    prediction = Mock()
    prediction.predict.return_value = torch.tensor([0, 1, 0])
    monkeypatch.setattr(
        selection_module.TrailsEstimator, "predict", lambda estimator, data: prediction
    )
    data: Any = Mock()
    data.split.return_value = [data, data]
    selector = ClusterNumberSelector(
        (2, 3),
        seeds=(11, 12),
        split_seed=42,
        selection_rule="one_standard_error",
        min_mean_pairwise_ari=0.75,
    )

    result = ClusterNumberSelector.from_config(selector.config).select(data)

    data.split.assert_called_once_with([0.8, 0.2], seed=42)
    assert result.selected_k == 2
    assert {"latent_mixture_bic_normalized", "selection_score", "rank"} <= set(
        result.run_metrics.columns
    )
    result.save(tmp_path)
    assert (tmp_path / "run_metrics.csv").exists()
    assert (tmp_path / "seed-11" / "k2" / "model.pt").exists()

    single_seed_result = ClusterNumberSelector((2, 3), seeds=11).select(data, validation_data=data)
    assert single_seed_result.selected_k == 2
    assert single_seed_result.stability_pairs.empty
    assert single_seed_result.seed_winners == {11: 2}

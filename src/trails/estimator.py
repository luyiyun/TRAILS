from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .config import EstimatorConfig
from .data import ClinicalTimeSeriesDataset, infer_data_config
from .model import TrailsSurvVaderModel
from .trainer import TrailsTrainer


class TrailsEstimator:
    def __init__(self, config: EstimatorConfig | None = None) -> None:
        self.config = config or EstimatorConfig()
        torch.manual_seed(self.config.seed)
        self.model = TrailsSurvVaderModel(self.config.data, self.config.model)
        self.trainer = TrailsTrainer(self.model, self.config.trainer)
        self.history: list[dict[str, float]] = []

    def fit(self, data: ClinicalTimeSeriesDataset) -> TrailsEstimator:
        self._validate_data_config(data)
        self.model.set_feature_means(data.feature_means)
        self.history = self.trainer.fit(data)
        return self

    def predict(self, data: ClinicalTimeSeriesDataset) -> Tensor:
        self._validate_data_config(data)
        return self.trainer.predict(data)

    def test(self, data: ClinicalTimeSeriesDataset) -> dict[str, float]:
        self._validate_data_config(data)
        return self.trainer.test(data)

    def save(self, path: str | Path) -> None:
        checkpoint = {
            "config": self.config.model_dump(mode="json"),
            "history": self.history,
            "model_state": self.model.state_dict(),
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, destination)

    @classmethod
    def load(cls, path: str | Path) -> TrailsEstimator:
        checkpoint: dict[str, Any] = torch.load(Path(path), map_location="cpu", weights_only=False)
        estimator = cls(EstimatorConfig.model_validate(checkpoint["config"]))
        estimator.model.load_state_dict(checkpoint["model_state"])
        estimator.history = list(checkpoint.get("history", []))
        return estimator

    def _validate_data_config(self, data: ClinicalTimeSeriesDataset) -> None:
        inferred = infer_data_config(data)
        if inferred != self.config.data:
            raise ValueError(
                "Data shape does not match estimator config: "
                f"expected {self.config.data}, got {inferred}."
            )

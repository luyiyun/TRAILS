from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .config import TrailsConfig
from .data import ClinicalTimeSeriesDataset, infer_data_config
from .diagnostics import LatentDiagnostics
from .model import TrailsSurvVaderModel
from .trainer import HistoryEntry, TrailsTrainer

HistoryCallback = Callable[[HistoryEntry], None]


class TrailsEstimator:
    def __init__(self, config: TrailsConfig | None = None) -> None:
        self.config = config or TrailsConfig()
        torch.manual_seed(self.config.seed)
        self.model = TrailsSurvVaderModel(self.config.data, self.config.model)
        trainer_config = self.config.trainer.model_copy(update={"seed": self.config.seed})
        self.trainer = TrailsTrainer(self.model, trainer_config)
        self.history: list[HistoryEntry] = []

    def fit(
        self,
        data: ClinicalTimeSeriesDataset,
        history_callback: HistoryCallback | None = None,
    ) -> TrailsEstimator:
        self._validate_data_config(data)
        self.model.set_feature_means(data.feature_means)
        if self.config.model.encoder.input.kind == "mtan":
            min_time, max_time = observed_time_range(data)
            self.model.set_reference_time_range(min_time, max_time)
        self.history = self.trainer.fit(
            data,
            history_callback=history_callback,
        )
        return self

    def predict(self, data: ClinicalTimeSeriesDataset) -> Tensor:
        self._validate_data_config(data)
        return self.trainer.predict(data)

    def predict_proba(self, data: ClinicalTimeSeriesDataset) -> Tensor:
        self._validate_data_config(data)
        return self.trainer.predict_proba(data)

    def predict_risk(self, data: ClinicalTimeSeriesDataset) -> Tensor:
        self._validate_data_config(data)
        return self.trainer.predict_risk(data)

    def test(self, data: ClinicalTimeSeriesDataset) -> dict[str, float]:
        self._validate_data_config(data)
        return self.trainer.test(data)

    def latent_diagnostics(self, data: ClinicalTimeSeriesDataset) -> LatentDiagnostics:
        self._validate_data_config(data)
        outputs, batch = self.trainer._collect_outputs(data)
        cluster_probabilities = outputs.cluster_probabilities.detach().cpu()
        diagnostics: LatentDiagnostics = {
            "z": outputs.latent_mean.detach().cpu(),
            "cluster_probabilities": cluster_probabilities,
            "pred_cluster": torch.argmax(cluster_probabilities, dim=-1).long(),
            "sample_index": torch.arange(len(data), dtype=torch.long),
        }
        if "cluster_label" in batch:
            diagnostics["true_cluster"] = batch["cluster_label"].detach().cpu().long()
        return diagnostics

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
        estimator = cls(TrailsConfig.model_validate(checkpoint["config"]))
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


def observed_time_range(data: ClinicalTimeSeriesDataset) -> tuple[float, float]:
    compact_data = data.with_return_kind("compact")
    min_time: float | None = None
    max_time: float | None = None
    for sample in compact_data.samples:
        observed_times = sample.times[sample.mask > 0].float()
        if observed_times.numel() == 0:
            continue
        sample_min = float(observed_times.min().item())
        sample_max = float(observed_times.max().item())
        min_time = sample_min if min_time is None else min(min_time, sample_min)
        max_time = sample_max if max_time is None else max(max_time, sample_max)

    if min_time is None or max_time is None:
        raise ValueError("mTAN reference time grid requires at least one observed time.")
    return min_time, max_time

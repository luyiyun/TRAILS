"""TRAILS无生存损失消融的共享baseline适配器。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Self

import numpy as np
import torch
from numpy.typing import NDArray

import trails
from trails.metrics import gaussian_log_prob

from .baseline_features import dataset_patient_ids
from .baselines import BaselineCapability, BaselinePrediction


class TrailsNoSurvivalBaseline:
    """以零生存损失拟合固定K的TRAILS，并提供潜空间BIC。"""

    capabilities: frozenset[BaselineCapability] = frozenset({"cluster"})

    def __init__(
        self,
        name: str,
        n_clusters: int,
        seed: int,
        model_config: trails.ModelConfig,
        trainer_config: trails.TrainerConfig,
    ) -> None:
        self.name, self.n_clusters, self.seed = name, n_clusters, seed
        self.model_config, self.trainer_config = model_config, trainer_config
        self.estimator: trails.TrailsEstimator | None = None

    def fit(
        self,
        train: trails.ClinicalTimeSeriesDataset,
        validation: trails.ClinicalTimeSeriesDataset,
    ) -> Self:
        model = self.model_config.model_copy(update={"n_clusters": self.n_clusters})
        trainer = self.trainer_config.model_copy(
            update={
                "batch_size": trails.resolve_batch_size(len(train), self.trainer_config.batch_size),
                "seed": self.seed,
            }
        )
        config = trails.TrailsConfig(
            data=trails.DataConfig(n_features=train.n_features),
            model=model,
            trainer=trainer,
            seed=self.seed,
        )
        self.estimator = trails.TrailsEstimator(config).fit(train, validation_data=validation)

        self.estimator.trainer.optimizer.state.clear()
        self.estimator.model.cpu()
        return self

    def predict_trails(self, data: trails.ClinicalTimeSeriesDataset) -> trails.TrailsPrediction:
        if self.estimator is None:
            raise RuntimeError("必须先拟合TRAILS-no-survival")
        self.estimator.model.to(self.estimator.config.trainer.device)
        prediction = self.estimator.predict(data)
        self.estimator.model.cpu()
        return prediction

    def k_selection_metrics(
        self,
        train: trails.ClinicalTimeSeriesDataset,
        validation: trails.ClinicalTimeSeriesDataset,
        *,
        prediction_times: NDArray[np.float64],
        risk_horizon: float,
    ) -> dict[str, float]:
        del validation, prediction_times, risk_horizon
        if self.estimator is None:
            raise RuntimeError("必须先拟合TRAILS-no-survival")
        latent = self.predict_trails(train).latent_representation
        model = self.estimator.model
        batch_size = self.estimator.config.trainer.batch_size
        if batch_size is None:
            raise RuntimeError("拟合后batch_size应已解析为整数")
        log_prior = torch.log_softmax(model.mixture_logits.detach(), dim=-1).unsqueeze(0)
        log_likelihood_sum = 0.0
        with torch.no_grad():
            for start in range(0, len(latent), batch_size):
                batch = latent[start : start + batch_size]
                component_log_prob = gaussian_log_prob(
                    batch.unsqueeze(1),
                    model.mixture_means.detach().unsqueeze(0),
                    model.mixture_log_variances.detach().unsqueeze(0),
                ).sum(dim=-1)
                log_likelihood_sum += float(
                    torch.logsumexp(log_prior + component_log_prob, dim=-1).sum().item()
                )
        latent_dim = self.estimator.config.model.latent_dim
        n_parameters = self.n_clusters * (2 * latent_dim) + (self.n_clusters - 1)
        bic = -2.0 * log_likelihood_sum + math.log(float(len(latent))) * n_parameters
        return {"bic": float(bic)}

    def predict(
        self,
        data: trails.ClinicalTimeSeriesDataset,
        *,
        prediction_times: NDArray[np.float64],
        risk_horizon: float,
    ) -> BaselinePrediction:
        del prediction_times, risk_horizon
        labels = self.predict_trails(data).predict().numpy().astype(np.int64, copy=False)
        return BaselinePrediction(
            method_name=self.name,
            patient_ids=dataset_patient_ids(data),
            cluster_labels=labels,
            n_clusters=self.n_clusters,
        )

    def save_model(self, path: Path) -> None:
        if self.estimator is None:
            raise RuntimeError("必须先拟合TRAILS-no-survival")
        self.estimator.save(path)

from __future__ import annotations

from pathlib import Path
from typing import Any, Self, cast

import joblib
import numpy as np
from numpy.typing import NDArray

from trails import ClinicalTimeSeriesDataset

from .baseline_features import dataset_patient_ids, dataset_survival_arrays
from .baseline_fpca import FPCAFeaturePipeline
from .baselines import BaselineCapability, BaselinePrediction


class DeepCoxMixturesBaseline:
    """在train-fitted FPCA特征上拟合官方Deep Cox Mixtures。"""

    capabilities: frozenset[BaselineCapability] = frozenset({"cluster", "survival"})

    def __init__(
        self,
        name: str,
        n_clusters: int,
        seed: int,
        n_components: int,
        grid_size: int,
        time_start: float,
        time_end: float,
        hidden_dims: tuple[int, ...],
        gamma: float,
        smoothing_factor: float,
        use_activation: bool,
        max_epochs: int,
        learning_rate: float,
        batch_size: int,
    ) -> None:
        self.name = name
        self.n_clusters = n_clusters
        self.seed = seed
        self.hidden_dims = hidden_dims
        self.gamma = gamma
        self.smoothing_factor = smoothing_factor
        self.use_activation = use_activation
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.features = FPCAFeaturePipeline(n_components, grid_size, time_start, time_end)
        self.model: Any | None = None

    def fit(
        self,
        train: ClinicalTimeSeriesDataset,
        validation: ClinicalTimeSeriesDataset,
    ) -> Self:
        """以train拟合FPCA和DCM，并只把validation用于官方早停选择。"""
        train_features = self.features.fit_transform(train)
        valid_features = self.features.transform(validation)
        train_event, train_time = dataset_survival_arrays(train)
        valid_event, valid_time = dataset_survival_arrays(validation)
        from auton_survival.models.dcm import DeepCoxMixtures

        self.model = DeepCoxMixtures(
            k=self.n_clusters,
            layers=list(self.hidden_dims),
            gamma=cast(Any, self.gamma),
            smoothing_factor=self.smoothing_factor,
            use_activation=self.use_activation,
            random_seed=self.seed,
        )
        self.model.fit(
            train_features,
            train_time,
            train_event.astype(np.float32),
            val_data=(valid_features, valid_time, valid_event.astype(np.float32)),
            iters=self.max_epochs,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
        )
        return self

    def predict(
        self,
        data: ClinicalTimeSeriesDataset,
        *,
        prediction_times: NDArray[np.float64],
        risk_horizon: float,
    ) -> BaselinePrediction:
        """输出潜在成分标签、固定时间风险及患者级生存曲线。"""
        if self.model is None:
            raise RuntimeError("DeepCoxMixturesBaseline必须先拟合")
        features = self.features.transform(data)
        probabilities = np.asarray(self.model.predict_latent_z(features), dtype=np.float64)
        expected_shape = (len(data), self.n_clusters)
        if probabilities.shape != expected_shape or not np.isfinite(probabilities).all():
            raise ValueError("官方DCM返回了形状错误或非有限的潜在成分概率")

        combined_times = np.unique(np.concatenate((prediction_times, np.asarray([risk_horizon]))))
        combined_survival = np.asarray(
            self.model.predict_survival(features, combined_times.tolist()),
            dtype=np.float64,
        )
        prediction_indices = np.searchsorted(combined_times, prediction_times)
        horizon_index = int(np.searchsorted(combined_times, risk_horizon))
        return BaselinePrediction(
            method_name=self.name,
            patient_ids=dataset_patient_ids(data),
            cluster_labels=np.asarray(probabilities.argmax(axis=1), dtype=np.int64),
            n_clusters=self.n_clusters,
            risk_score=np.clip(1.0 - combined_survival[:, horizon_index], 0.0, 1.0),
            risk_horizon=risk_horizon,
            survival_times=prediction_times.copy(),
            survival_probabilities=np.clip(combined_survival[:, prediction_indices], 0.0, 1.0),
        )

    def save_model(self, path: Path) -> None:
        """保存FPCA变换和官方DCM拟合状态。"""
        if self.model is None:
            raise RuntimeError("DeepCoxMixturesBaseline必须先拟合")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from trails.data import AlignedClinicalSample, ClinicalTimeSeriesDataset

from .config import BaselineMethod
from .evaluation import PredictionPayload, prediction_payload_from_dataset


class BaseBaseline(ABC):
    name: BaselineMethod

    def __init__(self, *, n_clusters: int, random_state: int, kmeans_iters: int = 50) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans_iters = kmeans_iters
        self.feature_pipeline: Pipeline | None = None
        self.cluster_model: KMeans | None = None
        self.cluster_risk: NDArray[np.float64] | None = None

    def fit(self, data: ClinicalTimeSeriesDataset) -> BaseBaseline:
        raw_features = self.extract_features(data)
        features = self.fit_features(raw_features)
        model_features = self.fit_model_features(features, data)
        self.cluster_model = KMeans(
            n_clusters=self.n_clusters,
            max_iter=self.kmeans_iters,
            n_init="auto",
            random_state=self.random_state,
        ).fit(model_features)
        train_cluster = self.cluster_model.predict(model_features)
        target = self.survival_risk_target(data)
        self.cluster_risk = np.array(
            [
                target[train_cluster == cluster].mean()
                if np.any(train_cluster == cluster)
                else target.mean()
                for cluster in range(self.n_clusters)
            ],
            dtype=np.float64,
        )
        return self

    def fit_predict(self, data: ClinicalTimeSeriesDataset) -> PredictionPayload:
        return self.fit(data).predict(data)

    def predict(self, data: ClinicalTimeSeriesDataset) -> PredictionPayload:
        if self.feature_pipeline is None or self.cluster_model is None or self.cluster_risk is None:
            raise RuntimeError(f"{self.__class__.__name__} must be fitted before predict().")
        raw_features = self.extract_features(data)
        features = self.feature_pipeline.transform(raw_features)
        model_features = self.predict_model_features(features)
        pred_cluster = self.cluster_model.predict(model_features)
        risk_score = self.cluster_risk[pred_cluster]
        return prediction_payload_from_dataset(
            data,
            pred_cluster=torch.as_tensor(pred_cluster, dtype=torch.long),
            risk_score=torch.as_tensor(risk_score, dtype=torch.float32),
        )

    def fit_features(self, raw_features: NDArray[np.float64]) -> NDArray[np.float64]:
        self.feature_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )
        return self.feature_pipeline.fit_transform(raw_features)

    def fit_model_features(
        self,
        features: NDArray[np.float64],
        data: ClinicalTimeSeriesDataset,
    ) -> NDArray[np.float64]:
        return features

    def predict_model_features(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        return features

    def extract_features(self, data: ClinicalTimeSeriesDataset) -> NDArray[np.float64]:
        aligned_data = data.with_return_kind("aligned")
        rows = [
            self.sample_features(aligned_data.samples[index].to_aligned())
            for index in range(len(aligned_data))
        ]
        return np.asarray(rows, dtype=np.float64)

    def sample_features(self, sample: AlignedClinicalSample) -> list[float]:
        times = sample.times.detach().cpu().numpy()
        values = sample.x.detach().cpu().numpy()
        mask = sample.mask.detach().cpu().numpy()
        n_visits, n_features = values.shape
        row: list[float] = []

        for reducer in ("mean", "last", "slope", "observed_fraction"):
            for feature_index in range(n_features):
                observed = np.flatnonzero(mask[:, feature_index] > 0)
                if reducer == "observed_fraction":
                    row.append(float(observed.size / max(1, n_visits)))
                elif observed.size == 0:
                    row.append(float("nan"))
                elif reducer == "mean":
                    row.append(float(values[observed, feature_index].mean()))
                elif reducer == "last":
                    row.append(float(values[observed[-1], feature_index]))
                else:
                    first = int(observed[0])
                    last = int(observed[-1])
                    span = float(times[last] - times[first])
                    slope = (
                        0.0
                        if span <= 1e-6
                        else float(
                            (values[last, feature_index] - values[first, feature_index]) / span
                        )
                    )
                    row.append(slope)

        row.append(float(n_visits))
        row.append(float(times[-1] - times[0]) if n_visits > 1 else 0.0)
        return row

    def survival_risk_target(self, data: ClinicalTimeSeriesDataset) -> NDArray[np.float64]:
        survival_time = np.asarray(
            [float(data[index].survival_time) for index in range(len(data))],
            dtype=np.float64,
        )
        event = np.asarray(
            [float(data[index].event) for index in range(len(data))], dtype=np.float64
        )
        return event - np.log(np.clip(survival_time, 1e-4, None))


class SummaryKMeansBaseline(BaseBaseline):
    name: BaselineMethod = "summary_kmeans"


class RiskStratifiedKMeansBaseline(BaseBaseline):
    name: BaselineMethod = "risk_stratified_kmeans"

    def __init__(
        self,
        *,
        n_clusters: int,
        random_state: int,
        ridge_alpha: float,
        risk_feature_weight: float,
        kmeans_iters: int = 50,
    ) -> None:
        super().__init__(
            n_clusters=n_clusters,
            random_state=random_state,
            kmeans_iters=kmeans_iters,
        )
        self.ridge_alpha = ridge_alpha
        self.risk_feature_weight = risk_feature_weight
        self.risk_model: Ridge | None = None
        self.risk_scaler: StandardScaler | None = None

    def fit_model_features(
        self,
        features: NDArray[np.float64],
        data: ClinicalTimeSeriesDataset,
    ) -> NDArray[np.float64]:
        target = self.survival_risk_target(data)
        self.risk_model = Ridge(alpha=self.ridge_alpha).fit(features, target)
        train_risk = self.risk_model.predict(features).reshape(-1, 1)
        self.risk_scaler = StandardScaler()
        risk_feature = (
            np.asarray(self.risk_scaler.fit_transform(train_risk), dtype=np.float64)
            * self.risk_feature_weight
        )
        return np.hstack([features, risk_feature])

    def predict_model_features(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.risk_model is None or self.risk_scaler is None:
            raise RuntimeError(f"{self.__class__.__name__} must be fitted before predict().")
        risk = self.risk_model.predict(features).reshape(-1, 1)
        risk_feature = (
            np.asarray(self.risk_scaler.transform(risk), dtype=np.float64)
            * self.risk_feature_weight
        )
        return np.hstack([features, risk_feature])


BASELINE_REGISTRY: Mapping[BaselineMethod, type[BaseBaseline]] = {
    SummaryKMeansBaseline.name: SummaryKMeansBaseline,
    RiskStratifiedKMeansBaseline.name: RiskStratifiedKMeansBaseline,
}


def make_baseline(
    method: BaselineMethod,
    *,
    n_clusters: int,
    random_state: int,
    kmeans_iters: int,
    ridge_alpha: float,
    risk_feature_weight: float,
) -> BaseBaseline:
    baseline_cls = BASELINE_REGISTRY[method]
    kwargs: dict[str, Any] = {
        "n_clusters": n_clusters,
        "random_state": random_state,
        "kmeans_iters": kmeans_iters,
    }
    if issubclass(baseline_cls, RiskStratifiedKMeansBaseline):
        kwargs.update(
            {
                "ridge_alpha": ridge_alpha,
                "risk_feature_weight": risk_feature_weight,
            }
        )
    return baseline_cls(**kwargs)

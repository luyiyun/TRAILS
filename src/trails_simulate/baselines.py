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


class FPCAKMeansBaseline(BaseBaseline):
    name: BaselineMethod = "fpca_kmeans"

    def __init__(
        self,
        *,
        n_clusters: int,
        random_state: int,
        fpca_components: int,
        fpca_grid_size: int,
        kmeans_iters: int = 50,
    ) -> None:
        super().__init__(
            n_clusters=n_clusters,
            random_state=random_state,
            kmeans_iters=kmeans_iters,
        )
        self.fpca_components = fpca_components
        self.fpca_grid_size = fpca_grid_size
        self.fpca_imputers: list[SimpleImputer] = []
        self.fpca_models: list[Any] = []
        self.reference_grid = np.linspace(0.0, 1.0, fpca_grid_size)

    def fit(self, data: ClinicalTimeSeriesDataset) -> BaseBaseline:
        raw_features = self.fit_fpca_features(data)
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

    def predict(self, data: ClinicalTimeSeriesDataset) -> PredictionPayload:
        if self.feature_pipeline is None or self.cluster_model is None or self.cluster_risk is None:
            raise RuntimeError(f"{self.__class__.__name__} must be fitted before predict().")
        raw_features = self.transform_fpca_features(data)
        features = self.feature_pipeline.transform(raw_features)
        model_features = self.predict_model_features(features)
        pred_cluster = self.cluster_model.predict(model_features)
        risk_score = self.cluster_risk[pred_cluster]
        return prediction_payload_from_dataset(
            data,
            pred_cluster=torch.as_tensor(pred_cluster, dtype=torch.long),
            risk_score=torch.as_tensor(risk_score, dtype=torch.float32),
        )

    def fit_fpca_features(self, data: ClinicalTimeSeriesDataset) -> NDArray[np.float64]:
        fdata_grid_cls, fpca_cls = load_skfda_fpca()
        self.fpca_imputers = []
        self.fpca_models = []
        score_blocks = []
        for matrix in self.interpolated_feature_matrices(data):
            imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
            imputed = np.asarray(imputer.fit_transform(matrix), dtype=np.float64)
            n_components = min(self.fpca_components, imputed.shape[0], imputed.shape[1])
            fpca_model = fpca_cls(n_components=n_components)
            scores = np.asarray(
                fpca_model.fit_transform(
                    fdata_grid_cls(data_matrix=imputed, grid_points=self.reference_grid)
                ),
                dtype=np.float64,
            )
            self.fpca_imputers.append(imputer)
            self.fpca_models.append(fpca_model)
            score_blocks.append(scores)
        return np.hstack(score_blocks)

    def transform_fpca_features(self, data: ClinicalTimeSeriesDataset) -> NDArray[np.float64]:
        if not self.fpca_imputers or not self.fpca_models:
            raise RuntimeError(f"{self.__class__.__name__} must be fitted before predict().")
        fdata_grid_cls, _fpca_cls = load_skfda_fpca()
        score_blocks = []
        matrices = self.interpolated_feature_matrices(data)
        for matrix, imputer, fpca_model in zip(
            matrices,
            self.fpca_imputers,
            self.fpca_models,
            strict=True,
        ):
            imputed = np.asarray(imputer.transform(matrix), dtype=np.float64)
            scores = np.asarray(
                fpca_model.transform(
                    fdata_grid_cls(data_matrix=imputed, grid_points=self.reference_grid)
                ),
                dtype=np.float64,
            )
            score_blocks.append(scores)
        return np.hstack(score_blocks)

    def interpolated_feature_matrices(
        self,
        data: ClinicalTimeSeriesDataset,
    ) -> list[NDArray[np.float64]]:
        aligned_data = data.with_return_kind("aligned")
        matrices = [
            np.full((len(aligned_data), self.fpca_grid_size), np.nan, dtype=np.float64)
            for _feature in range(aligned_data.n_features)
        ]
        for sample_index in range(len(aligned_data)):
            sample = aligned_data.samples[sample_index].to_aligned()
            times = sample.times.detach().cpu().numpy()
            values = sample.x.detach().cpu().numpy()
            mask = sample.mask.detach().cpu().numpy()
            for feature_index in range(aligned_data.n_features):
                matrices[feature_index][sample_index] = interpolate_feature_to_grid(
                    times=times,
                    values=values[:, feature_index],
                    mask=mask[:, feature_index],
                    reference_grid=self.reference_grid,
                )
        return matrices


BASELINE_REGISTRY: Mapping[BaselineMethod, type[BaseBaseline]] = {
    SummaryKMeansBaseline.name: SummaryKMeansBaseline,
    RiskStratifiedKMeansBaseline.name: RiskStratifiedKMeansBaseline,
    FPCAKMeansBaseline.name: FPCAKMeansBaseline,
}


def make_baseline(
    method: BaselineMethod,
    *,
    n_clusters: int,
    random_state: int,
    kmeans_iters: int,
    ridge_alpha: float,
    risk_feature_weight: float,
    fpca_components: int = 3,
    fpca_grid_size: int = 16,
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
    if issubclass(baseline_cls, FPCAKMeansBaseline):
        kwargs.update(
            {
                "fpca_components": fpca_components,
                "fpca_grid_size": fpca_grid_size,
            }
        )
    return baseline_cls(**kwargs)


def load_skfda_fpca() -> tuple[Any, Any]:
    try:
        from skfda import FDataGrid
        from skfda.preprocessing.dim_reduction import FPCA
    except ImportError as error:
        raise RuntimeError(
            "baseline method fpca_kmeans requires scikit-fda. Run `uv sync` after "
            "updating project dependencies."
        ) from error
    return FDataGrid, FPCA


def interpolate_feature_to_grid(
    *,
    times: NDArray[np.float64],
    values: NDArray[np.float64],
    mask: NDArray[np.float64],
    reference_grid: NDArray[np.float64],
) -> NDArray[np.float64]:
    observed = np.flatnonzero(mask > 0)
    if observed.size == 0:
        return np.full(reference_grid.shape, np.nan, dtype=np.float64)
    observed_times = times[observed].astype(np.float64)
    observed_values = values[observed].astype(np.float64)
    if observed.size == 1:
        return np.full(reference_grid.shape, float(observed_values[0]), dtype=np.float64)
    span = float(observed_times[-1] - observed_times[0])
    if span <= 1e-8:
        return np.full(reference_grid.shape, float(observed_values[-1]), dtype=np.float64)
    normalized_times = (observed_times - observed_times[0]) / span
    return np.asarray(
        np.interp(
            reference_grid,
            normalized_times,
            observed_values,
            left=float(observed_values[0]),
            right=float(observed_values[-1]),
        ),
        dtype=np.float64,
    )

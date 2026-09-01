"""不同数据工作流共享的传统聚类与患者级生存预测基线。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self, cast

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

from trails import ClinicalTimeSeriesDataset

from .baseline_features import (
    SummaryFeaturePipeline,
    dataset_patient_ids,
    dataset_survival_arrays,
)
from .baselines import BaselineCapability, BaselinePrediction


class SummaryKMeansBaseline:
    """在训练集标准化轨迹摘要上拟合KMeans。"""

    capabilities: frozenset[BaselineCapability] = frozenset({"cluster"})

    def __init__(
        self,
        name: str,
        n_clusters: int,
        seed: int,
        kmeans_iters: int,
    ) -> None:
        self.name = name
        self.n_clusters = n_clusters
        self.seed = seed
        self.kmeans_iters = kmeans_iters
        self.features = SummaryFeaturePipeline()
        self.model: KMeans | None = None

    def fit(
        self,
        train: ClinicalTimeSeriesDataset,
        validation: ClinicalTimeSeriesDataset,
    ) -> Self:
        """仅用train拟合特征管线和KMeans；validation保持冻结。"""
        del validation
        train_features = self.features.fit_transform(train)
        self.model = KMeans(
            n_clusters=self.n_clusters,
            n_init="auto",
            max_iter=self.kmeans_iters,
            random_state=self.seed,
        ).fit(train_features)
        return self

    def predict(
        self,
        data: ClinicalTimeSeriesDataset,
        *,
        prediction_times: NDArray[np.float64],
        risk_horizon: float,
    ) -> BaselinePrediction:
        """返回一个冻结划分的簇标签，不构造派生生存风险。"""
        del prediction_times, risk_horizon
        if self.model is None:
            raise RuntimeError("SummaryKMeansBaseline必须先拟合")
        labels = np.asarray(self.model.predict(self.features.transform(data)), dtype=np.int64)
        return BaselinePrediction(
            method_name=self.name,
            patient_ids=dataset_patient_ids(data),
            cluster_labels=labels,
            n_clusters=self.n_clusters,
        )

    def save_model(self, path: Path) -> None:
        """保存训练集特征管线、KMeans和复现参数。"""
        if self.model is None:
            raise RuntimeError("SummaryKMeansBaseline必须先拟合")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "method": self.name,
                "seed": self.seed,
                "n_clusters": self.n_clusters,
                "features": self.features,
                "model": self.model,
            },
            path,
        )


class SummarySurvivalBaseline:
    """在训练集标准化轨迹摘要上拟合患者级生存模型。"""

    capabilities: frozenset[BaselineCapability] = frozenset({"survival"})

    def __init__(self, name: str, model: Any) -> None:
        self.name = name
        self.features = SummaryFeaturePipeline()
        self.model = model
        self.fitted = False

    def fit(
        self,
        train: ClinicalTimeSeriesDataset,
        validation: ClinicalTimeSeriesDataset,
    ) -> Self:
        """仅用train拟合摘要变换与生存模型。"""
        del validation
        features = self.features.fit_transform(train)
        event, time = dataset_survival_arrays(train)
        self.model.fit(features, Surv.from_arrays(event, time))
        self.fitted = True
        return self

    def predict(
        self,
        data: ClinicalTimeSeriesDataset,
        *,
        prediction_times: NDArray[np.float64],
        risk_horizon: float,
    ) -> BaselinePrediction:
        """输出固定时间窗事件概率和患者级生存曲线。"""
        if not self.fitted:
            raise RuntimeError(f"{self.__class__.__name__}必须先拟合")
        features = self.features.transform(data)
        functions = cast(Any, self.model.predict_survival_function(features))
        probabilities = np.asarray(
            [function(prediction_times) for function in functions],
            dtype=np.float64,
        )
        horizon_survival = np.asarray(
            [float(function(risk_horizon)) for function in functions],
            dtype=np.float64,
        )
        return BaselinePrediction(
            method_name=self.name,
            patient_ids=dataset_patient_ids(data),
            risk_score=np.clip(1.0 - horizon_survival, 0.0, 1.0),
            risk_horizon=risk_horizon,
            survival_times=prediction_times.copy(),
            survival_probabilities=np.clip(probabilities, 0.0, 1.0),
        )

    def save_model(self, path: Path) -> None:
        """保存摘要变换与已拟合生存模型。"""
        if not self.fitted:
            raise RuntimeError(f"{self.__class__.__name__}必须先拟合")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)


class CoxPHBaseline(SummarySurvivalBaseline):
    """带L2稳定项的Cox比例风险基线。"""

    def __init__(self, name: str, alpha: float) -> None:
        super().__init__(name, CoxPHSurvivalAnalysis(alpha=cast(Any, alpha)))


class RandomSurvivalForestBaseline(SummarySurvivalBaseline):
    """基于轨迹摘要的随机生存森林基线。"""

    def __init__(
        self,
        name: str,
        seed: int,
        n_estimators: int,
        min_samples_split: int,
        min_samples_leaf: int,
        max_features: str | int | float | None,
        n_jobs: int,
    ) -> None:
        model = RandomSurvivalForest(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=cast(Any, max_features),
            n_jobs=n_jobs,
            random_state=seed,
        )
        super().__init__(name, model)

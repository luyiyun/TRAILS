"""不同数据工作流共享的Cox风险增强聚类基线。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self, cast

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

from trails import ClinicalTimeSeriesDataset

from .baseline_features import (
    SummaryFeaturePipeline,
    dataset_patient_ids,
    dataset_survival_arrays,
)
from .baselines import BaselineCapability, BaselinePrediction


class CoxRiskKMeansBaseline:
    """将删失感知Cox线性预测量加入轨迹摘要后拟合KMeans。"""

    capabilities: frozenset[BaselineCapability] = frozenset({"cluster"})

    def __init__(
        self,
        name: str,
        n_clusters: int,
        seed: int,
        kmeans_iters: int,
        cox_alpha: float,
        risk_feature_weight: float,
    ) -> None:
        self.name = name
        self.n_clusters = n_clusters
        self.seed = seed
        self.kmeans_iters = kmeans_iters
        self.risk_feature_weight = risk_feature_weight
        self.features = SummaryFeaturePipeline()
        self.risk_model = CoxPHSurvivalAnalysis(alpha=cast(Any, cox_alpha))
        self.risk_scaler = StandardScaler()
        self.model: KMeans | None = None

    def fit(
        self,
        train: ClinicalTimeSeriesDataset,
        validation: ClinicalTimeSeriesDataset,
    ) -> Self:
        """仅用train拟合摘要、Cox风险维度和KMeans。"""
        del validation
        features = self.features.fit_transform(train)
        event, time = dataset_survival_arrays(train)
        self.risk_model.fit(features, Surv.from_arrays(event, time))
        risk = np.asarray(self.risk_model.predict(features), dtype=np.float64).reshape(-1, 1)
        scaled_risk = self.risk_scaler.fit_transform(risk) * self.risk_feature_weight
        self.model = KMeans(
            n_clusters=self.n_clusters,
            n_init="auto",
            max_iter=self.kmeans_iters,
            random_state=self.seed,
        ).fit(np.hstack((features, scaled_risk)))
        return self

    def predict(
        self,
        data: ClinicalTimeSeriesDataset,
        *,
        prediction_times: NDArray[np.float64],
        risk_horizon: float,
    ) -> BaselinePrediction:
        """转换冻结划分并返回簇标签，不暴露内部Cox风险为预测结局。"""
        del prediction_times, risk_horizon
        if self.model is None:
            raise RuntimeError("CoxRiskKMeansBaseline必须先拟合")
        features = self.features.transform(data)
        risk = np.asarray(self.risk_model.predict(features), dtype=np.float64).reshape(-1, 1)
        scaled_risk = (
            np.asarray(self.risk_scaler.transform(risk), dtype=np.float64)
            * self.risk_feature_weight
        )
        labels = np.asarray(
            self.model.predict(np.hstack((features, scaled_risk))),
            dtype=np.int64,
        )
        return BaselinePrediction(
            method_name=self.name,
            patient_ids=dataset_patient_ids(data),
            cluster_labels=labels,
            n_clusters=self.n_clusters,
        )

    def save_model(self, path: Path) -> None:
        """保存摘要、Cox、风险标准化与KMeans参数。"""
        if self.model is None:
            raise RuntimeError("CoxRiskKMeansBaseline必须先拟合")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

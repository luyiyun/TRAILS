"""不同数据工作流共享的单变量与多变量FPCA聚类基线。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from trails import ClinicalTimeSeriesDataset

from .baseline_features import dataset_patient_ids
from .baselines import BaselineCapability, BaselinePrediction


class UFPCAFeaturePipeline:
    """仅由训练集拟合的逐变量FPCA特征变换。"""

    def __init__(
        self,
        n_components: int,
        grid_size: int,
        time_start: float,
        time_end: float,
    ) -> None:
        if time_end <= time_start:
            raise ValueError("FPCA时间窗终点必须晚于起点")
        self.n_components = n_components
        self.reference_grid = np.linspace(time_start, time_end, grid_size)
        self.imputers: list[SimpleImputer] = []
        self.fpca_models: list[Any] = []
        self.score_scaler: StandardScaler | None = None

    def fit_transform(self, data: ClinicalTimeSeriesDataset) -> NDArray[np.float64]:
        """拟合插补、FPCA和标准化，并返回训练集特征。"""
        scores = self._fit_scores(data)
        self.score_scaler = StandardScaler()
        return np.asarray(self.score_scaler.fit_transform(scores), dtype=np.float64)

    def transform(self, data: ClinicalTimeSeriesDataset) -> NDArray[np.float64]:
        """使用训练集参数转换冻结划分。"""
        if self.score_scaler is None:
            raise RuntimeError("UFPCAFeaturePipeline必须先拟合")
        return np.asarray(
            self.score_scaler.transform(self._transform_scores(data)),
            dtype=np.float64,
        )

    def _fit_scores(self, data: ClinicalTimeSeriesDataset) -> NDArray[np.float64]:
        """按变量拟合插补器与FPCA并拼接患者得分。"""
        fdata_grid_cls, fpca_cls = load_skfda_fpca()
        self.imputers = []
        self.fpca_models = []
        score_blocks: list[NDArray[np.float64]] = []
        for matrix in self._interpolated_matrices(data):
            imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
            imputed = np.asarray(imputer.fit_transform(matrix), dtype=np.float64)
            n_components = min(self.n_components, imputed.shape[0], imputed.shape[1])
            fpca_model = fpca_cls(n_components=n_components)
            scores = np.asarray(
                fpca_model.fit_transform(
                    fdata_grid_cls(data_matrix=imputed, grid_points=self.reference_grid)
                ),
                dtype=np.float64,
            )
            self.imputers.append(imputer)
            self.fpca_models.append(fpca_model)
            score_blocks.append(scores)
        return np.hstack(score_blocks)

    def _transform_scores(self, data: ClinicalTimeSeriesDataset) -> NDArray[np.float64]:
        """使用训练集插补器与FPCA转换冻结划分。"""
        if not self.imputers or not self.fpca_models:
            raise RuntimeError("UFPCAFeaturePipeline必须先拟合")
        fdata_grid_cls, _ = load_skfda_fpca()
        matrices = self._interpolated_matrices(data)
        if len(matrices) != len(self.fpca_models):
            raise ValueError("冻结划分特征数与FPCA训练集不一致")
        score_blocks: list[NDArray[np.float64]] = []
        for matrix, imputer, fpca_model in zip(
            matrices, self.imputers, self.fpca_models, strict=True
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

    def _interpolated_matrices(
        self,
        data: ClinicalTimeSeriesDataset,
    ) -> list[NDArray[np.float64]]:
        """在同一绝对时间网格插值，边界使用最近观测值延伸。"""
        aligned = data.with_return_kind("aligned")
        matrices = [
            np.full((len(aligned), len(self.reference_grid)), np.nan, dtype=np.float64)
            for _ in range(aligned.n_features)
        ]
        start, end = float(self.reference_grid[0]), float(self.reference_grid[-1])
        for patient_index, raw_sample in enumerate(aligned.samples):
            sample = raw_sample.to_aligned()
            times = sample.times.detach().cpu().numpy()
            values = sample.x.detach().cpu().numpy()
            mask = sample.mask.detach().cpu().numpy()
            in_window = (times >= start) & (times <= end)
            for feature_index in range(aligned.n_features):
                observed = np.flatnonzero((mask[:, feature_index] > 0) & in_window)
                if observed.size == 0:
                    continue
                observed_values = values[observed, feature_index]
                matrices[feature_index][patient_index] = np.interp(
                    self.reference_grid,
                    times[observed],
                    observed_values,
                    left=float(observed_values[0]),
                    right=float(observed_values[-1]),
                )
        return matrices


class UFPCAKMeansBaseline(UFPCAFeaturePipeline):
    """在固定全局时间网格的逐变量FPCA得分上拟合KMeans。"""

    capabilities: frozenset[BaselineCapability] = frozenset({"cluster"})

    def __init__(
        self,
        name: str,
        n_clusters: int,
        seed: int,
        kmeans_iters: int,
        n_components: int,
        grid_size: int,
        time_start: float,
        time_end: float,
    ) -> None:
        super().__init__(n_components, grid_size, time_start, time_end)
        self.name = name
        self.n_clusters = n_clusters
        self.seed = seed
        self.kmeans_iters = kmeans_iters
        self.model: KMeans | None = None

    def fit(
        self,
        train: ClinicalTimeSeriesDataset,
        validation: ClinicalTimeSeriesDataset,
    ) -> Self:
        """仅用train拟合FPCA变换和KMeans。"""
        del validation
        self.model = KMeans(
            n_clusters=self.n_clusters,
            n_init="auto",
            max_iter=self.kmeans_iters,
            random_state=self.seed,
        ).fit(self.fit_transform(train))
        return self

    def predict(
        self,
        data: ClinicalTimeSeriesDataset,
        *,
        prediction_times: NDArray[np.float64],
        risk_horizon: float,
    ) -> BaselinePrediction:
        """使用训练集FPCA与KMeans返回冻结划分簇标签。"""
        del prediction_times, risk_horizon
        if self.model is None:
            raise RuntimeError("UFPCAKMeansBaseline必须先拟合")
        labels = np.asarray(self.model.predict(self.transform(data)), dtype=np.int64)
        return BaselinePrediction(
            method_name=self.name,
            patient_ids=dataset_patient_ids(data),
            cluster_labels=labels,
            n_clusters=self.n_clusters,
        )

    def save_model(self, path: Path) -> None:
        """保存FPCA、KMeans及全部训练集变换参数。"""
        if self.model is None:
            raise RuntimeError("UFPCAKMeansBaseline必须先拟合")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)


class MFPCAFeaturePipeline:
    """用FDApy从训练集拟合不规则多变量FPCA特征。"""

    def __init__(
        self,
        n_components: int,
        grid_size: int,
        time_start: float,
        time_end: float,
    ) -> None:
        if time_end <= time_start:
            raise ValueError("MFPCA时间窗终点必须晚于起点")
        self.n_components = n_components
        self.reference_grid = np.linspace(time_start, time_end, grid_size)
        self.empty_curves: NDArray[np.float64] | None = None
        self.model: Any | None = None
        self.score_scaler: StandardScaler | None = None

    def fit_transform(self, data: ClinicalTimeSeriesDataset) -> NDArray[np.float64]:
        """拟合训练集缺失曲线、MFPCA和得分标准化。"""
        functional_data = self._functional_data(data, fit=True)
        mfpca_cls, dense_argvals_cls, *_ = load_fdapy_mfpca()
        n_components = min(self.n_components, len(data), len(self.reference_grid))
        expansions = [
            {"method": "UFPCA", "n_components": n_components, "method_smoothing": "LP"}
            for _ in range(data.n_features)
        ]
        model = mfpca_cls(n_components=n_components, univariate_expansions=expansions)
        points = [
            dense_argvals_cls({"input_dim_0": self.reference_grid}) for _ in range(data.n_features)
        ]
        model.fit(functional_data, points=points, method_smoothing="LP")
        self.model = model
        scores = np.asarray(
            model.transform(functional_data, method="NumInt", method_smoothing="LP"),
            dtype=np.float64,
        )
        self.score_scaler = StandardScaler()
        return np.asarray(self.score_scaler.fit_transform(scores), dtype=np.float64)

    def transform(self, data: ClinicalTimeSeriesDataset) -> NDArray[np.float64]:
        """使用训练集MFPCA与标准化参数转换冻结划分。"""
        if self.model is None or self.score_scaler is None:
            raise RuntimeError("MFPCAFeaturePipeline必须先拟合")
        scores = np.asarray(
            self.model.transform(
                self._functional_data(data, fit=False),
                method="NumInt",
                method_smoothing="LP",
            ),
            dtype=np.float64,
        )
        return np.asarray(self.score_scaler.transform(scores), dtype=np.float64)

    def _functional_data(self, data: ClinicalTimeSeriesDataset, *, fit: bool) -> Any:
        """保留患者实际观测格点，仅为整条缺失曲线使用train均值曲线。"""
        dense_argvals_cls, irregular_argvals_cls, irregular_values_cls = load_fdapy_mfpca()[1:4]
        irregular_data_cls, multivariate_data_cls = load_fdapy_mfpca()[4:]
        aligned = data.with_return_kind("aligned")
        grid_size = len(self.reference_grid)
        argvals: list[dict[int, Any]] = [dict() for _ in range(aligned.n_features)]
        values: list[dict[int, NDArray[np.float64]]] = [dict() for _ in range(aligned.n_features)]
        bin_sums = np.zeros((aligned.n_features, grid_size), dtype=np.float64)
        bin_counts = np.zeros((aligned.n_features, grid_size), dtype=np.int64)
        missing: list[tuple[int, int]] = []
        start, end = float(self.reference_grid[0]), float(self.reference_grid[-1])

        for patient_index, raw_sample in enumerate(aligned.samples):
            sample = raw_sample.to_aligned()
            times = sample.times.detach().cpu().numpy()
            sample_values = sample.x.detach().cpu().numpy()
            mask = sample.mask.detach().cpu().numpy()
            in_window = (times >= start) & (times <= end)
            for feature_index in range(aligned.n_features):
                observed = np.flatnonzero((mask[:, feature_index] > 0) & in_window)
                if observed.size == 0:
                    missing.append((feature_index, patient_index))
                    continue
                bins = np.rint((times[observed] - start) / (end - start) * (grid_size - 1)).astype(
                    int
                )
                unique_bins = np.unique(bins)
                binned = np.asarray(
                    [
                        sample_values[observed[bins == index], feature_index].mean()
                        for index in unique_bins
                    ],
                    dtype=np.float64,
                )
                argvals[feature_index][patient_index] = dense_argvals_cls(
                    {"input_dim_0": self.reference_grid[unique_bins]}
                )
                values[feature_index][patient_index] = binned
                if fit:
                    bin_sums[feature_index, unique_bins] += binned
                    bin_counts[feature_index, unique_bins] += 1

        if fit:
            curves = np.empty_like(bin_sums)
            for feature_index in range(aligned.n_features):
                observed_bins = np.flatnonzero(bin_counts[feature_index] > 0)
                if observed_bins.size == 0:
                    raise ValueError(f"训练集特征{feature_index}在MFPCA时间窗内无观测")
                means = (
                    bin_sums[feature_index, observed_bins]
                    / bin_counts[feature_index, observed_bins]
                )
                curves[feature_index] = np.interp(np.arange(grid_size), observed_bins, means)
            self.empty_curves = curves
        if self.empty_curves is None or self.empty_curves.shape[0] != aligned.n_features:
            raise ValueError("冻结划分特征数与MFPCA训练集不一致")
        for feature_index, patient_index in missing:
            argvals[feature_index][patient_index] = dense_argvals_cls(
                {"input_dim_0": self.reference_grid}
            )
            values[feature_index][patient_index] = self.empty_curves[feature_index].copy()

        components = [
            irregular_data_cls(
                irregular_argvals_cls(dict(sorted(component_argvals.items()))),
                irregular_values_cls(dict(sorted(component_values.items()))),
            )
            for component_argvals, component_values in zip(argvals, values, strict=True)
        ]
        return multivariate_data_cls(components)


class MFPCAKMeansBaseline(MFPCAFeaturePipeline):
    """在FDApy联合多变量FPCA得分上拟合KMeans。"""

    capabilities: frozenset[BaselineCapability] = frozenset({"cluster"})

    def __init__(
        self,
        name: str,
        n_clusters: int,
        seed: int,
        kmeans_iters: int,
        n_components: int,
        grid_size: int,
        time_start: float,
        time_end: float,
    ) -> None:
        super().__init__(n_components, grid_size, time_start, time_end)
        self.name = name
        self.n_clusters = n_clusters
        self.seed = seed
        self.kmeans_iters = kmeans_iters
        self.kmeans: KMeans | None = None

    def fit(
        self,
        train: ClinicalTimeSeriesDataset,
        validation: ClinicalTimeSeriesDataset,
    ) -> Self:
        """仅用train拟合MFPCA变换和KMeans。"""
        del validation
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init="auto",
            max_iter=self.kmeans_iters,
            random_state=self.seed,
        ).fit(self.fit_transform(train))
        return self

    def predict(
        self,
        data: ClinicalTimeSeriesDataset,
        *,
        prediction_times: NDArray[np.float64],
        risk_horizon: float,
    ) -> BaselinePrediction:
        """使用训练集MFPCA与KMeans返回冻结划分簇标签。"""
        del prediction_times, risk_horizon
        if self.kmeans is None:
            raise RuntimeError("MFPCAKMeansBaseline必须先拟合")
        labels = np.asarray(self.kmeans.predict(self.transform(data)), dtype=np.int64)
        return BaselinePrediction(
            method_name=self.name,
            patient_ids=dataset_patient_ids(data),
            cluster_labels=labels,
            n_clusters=self.n_clusters,
        )

    def save_model(self, path: Path) -> None:
        """保存MFPCA、KMeans及全部训练集变换参数。"""
        if self.kmeans is None:
            raise RuntimeError("MFPCAKMeansBaseline必须先拟合")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)


def load_skfda_fpca() -> tuple[Any, Any]:
    """延迟导入scikit-fda对象，避免共享模块加载时产生额外开销。"""
    from skfda import FDataGrid
    from skfda.preprocessing.dim_reduction import FPCA

    return FDataGrid, FPCA


def load_fdapy_mfpca() -> tuple[Any, Any, Any, Any, Any, Any]:
    """延迟导入FDApy对象，避免未运行MFPCA时加载该依赖。"""
    from FDApy.preprocessing.dim_reduction.mfpca import MFPCA
    from FDApy.representation.argvals import DenseArgvals, IrregularArgvals
    from FDApy.representation.functional_data import (
        IrregularFunctionalData,
        MultivariateFunctionalData,
    )
    from FDApy.representation.values import IrregularValues

    return (
        MFPCA,
        DenseArgvals,
        IrregularArgvals,
        IrregularValues,
        IrregularFunctionalData,
        MultivariateFunctionalData,
    )

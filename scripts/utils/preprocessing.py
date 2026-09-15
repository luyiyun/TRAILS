"""支持多特征矩阵和结构性NaN的独立sklearn预处理步骤。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data  # type: ignore


def _matrix(estimator: BaseEstimator, X: ArrayLike, *, reset: bool) -> NDArray[np.float64]:
    values = np.asarray(
        validate_data(
            estimator,
            cast(Any, X),
            reset=reset,
            dtype="float64",
            ensure_all_finite="allow-nan",
            copy=True,
        ),
        dtype=np.float64,
    )
    if reset and (empty := np.isnan(values).all(axis=0)).any():
        names = np.asarray(getattr(estimator, "feature_names_in_", np.arange(values.shape[1])))
        raise ValueError(f"train存在全缺失特征：{names[empty].tolist()}")
    return values


class Winsorizer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """按列拟合训练分位点并截尾，忽略NaN拟合且保留缺失位置。"""

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X: ArrayLike, y: Any = None) -> Self:
        if not 0 <= self.lower_quantile < self.upper_quantile <= 1:
            raise ValueError("截尾分位点必须满足 0 <= lower < upper <= 1")
        values = _matrix(self, X, reset=True)
        self.lower_, self.upper_ = np.nanquantile(
            values, [self.lower_quantile, self.upper_quantile], axis=0
        )
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, ["lower_", "upper_"])
        return np.clip(_matrix(self, X, reset=False), self.lower_, self.upper_)


class AutoLog1pTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """根据训练观测的偏度改善逐列选择log1p，仅保存变换选择。"""

    def __init__(self, skew_threshold: float = 1.0) -> None:
        self.skew_threshold = skew_threshold

    def fit(self, X: ArrayLike, y: Any = None) -> Self:
        if not np.isfinite(self.skew_threshold):
            raise ValueError("skew_threshold须为有限数值")
        values = _matrix(self, X, reset=True)
        eligible = ((~np.isnan(values)).sum(axis=0) >= 3) & (np.nanmin(values, axis=0) >= 0)
        self.use_log_ = np.zeros(values.shape[1], dtype=bool)
        if not eligible.any():
            return self
        candidates = values[:, eligible]
        # 沿用groupby.skew的校正Fisher–Pearson算法，NaN不计入观测数或偏度。
        groups = np.zeros(len(values), dtype=int)
        raw_skewness = pd.DataFrame(candidates).groupby(groups).skew().to_numpy()[0]
        log_skewness = pd.DataFrame(np.log1p(candidates)).groupby(groups).skew().to_numpy()[0]
        self.use_log_[eligible] = (
            np.isfinite(raw_skewness)
            & np.isfinite(log_skewness)
            & (raw_skewness > self.skew_threshold)
            & (np.abs(log_skewness) < np.abs(raw_skewness))
        )
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, "use_log_")
        values = _matrix(self, X, reset=False)
        negative = self.use_log_ & (values < 0).any(axis=0)
        if negative.any():
            names = np.asarray(getattr(self, "feature_names_in_", np.arange(values.shape[1])))
            raise ValueError(f"train选定log1p的指标出现负值：{names[negative].tolist()}")
        values[:, self.use_log_] = np.log1p(values[:, self.use_log_])
        return values


class RobustScalerWithStdFallback(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """逐列以中位数居中，IQR严格为零时回退总体标准差，保留NaN。"""

    def fit(self, X: ArrayLike, y: Any = None) -> Self:
        values = _matrix(self, X, reset=True)
        self.center_ = np.nanmedian(values, axis=0)
        lower, upper = np.nanquantile(values, [0.25, 0.75], axis=0)
        self.scale_ = upper - lower
        fallback = self.scale_ == 0
        if fallback.any():
            self.scale_[fallback] = pd.DataFrame(values[:, fallback]).std(ddof=0).to_numpy()
        invalid = ~np.isfinite(self.center_) | ~np.isfinite(self.scale_) | (self.scale_ <= 0)
        if invalid.any():
            names = np.asarray(getattr(self, "feature_names_in_", np.arange(values.shape[1])))
            raise ValueError(f"train预处理参数非有限或尺度非正：{names[invalid].tolist()}")
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, ["center_", "scale_"])
        return (_matrix(self, X, reset=False) - self.center_) / self.scale_


def save_scaler(
    path: Path,
    scaler: BaseEstimator,
    log_transformer: AutoLog1pTransformer | Literal["passthrough"],
    feature_order: list[str],
) -> None:
    # 仅导出还原数值所需参数：先按需log1p，再减center、除scale。
    center, scale = np.zeros(len(feature_order)), np.ones(len(feature_order))
    if isinstance(scaler, StandardScaler):
        center, scale = scaler.mean_, scaler.scale_
    elif isinstance(scaler, MinMaxScaler):
        center, scale = -scaler.min_ / scaler.scale_, 1.0 / scaler.scale_
    elif isinstance(scaler, RobustScalerWithStdFallback):
        center, scale = scaler.center_, scaler.scale_
    if not np.isfinite(np.r_[center, scale]).all() or (np.asarray(scale) <= 0).any():
        raise ValueError("导出的预处理参数须有限且尺度为正")
    parameters = pd.DataFrame(
        {
            "feature": feature_order,
            "transform": np.where(log_transformer.use_log_, "log1p", "none")
            if isinstance(log_transformer, AutoLog1pTransformer)
            else "none",
            "center": center,
            "scale": scale,
        }
    )
    parameters.to_csv(path, index=False)

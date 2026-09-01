"""传统基线共享的纵向摘要特征。"""

from __future__ import annotations

from itertools import product

import numpy as np
from numpy.typing import NDArray
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from trails import AlignedClinicalSample, ClinicalTimeSeriesDataset


class SummaryFeaturePipeline:
    """提取并按训练集拟合标准化的患者级轨迹摘要。"""

    def __init__(self) -> None:
        self.pipeline: Pipeline | None = None

    def fit_transform(self, data: ClinicalTimeSeriesDataset) -> NDArray[np.float64]:
        """从训练集拟合缺失填补和标准化参数并返回特征。"""
        self.pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean", keep_empty_features=True)),
                ("scaler", StandardScaler()),
            ]
        )
        return np.asarray(self.pipeline.fit_transform(self.extract(data)), dtype=np.float64)

    def transform(self, data: ClinicalTimeSeriesDataset) -> NDArray[np.float64]:
        """使用训练集参数转换一个冻结划分。"""
        if self.pipeline is None:
            raise RuntimeError("SummaryFeaturePipeline必须先拟合")
        return np.asarray(self.pipeline.transform(self.extract(data)), dtype=np.float64)

    @staticmethod
    def extract(data: ClinicalTimeSeriesDataset) -> NDArray[np.float64]:
        """按患者生成每变量mean、last、slope和观测比例。"""
        aligned = data.with_return_kind("aligned")
        rows = [
            SummaryFeaturePipeline._sample_features(aligned.samples[index].to_aligned())
            for index in range(len(aligned))
        ]
        return np.asarray(rows, dtype=np.float64)

    @staticmethod
    def _sample_features(sample: AlignedClinicalSample) -> list[float]:
        """提取单名患者的轨迹摘要及总体观察长度。"""
        times = sample.times.detach().cpu().numpy()
        values = sample.x.detach().cpu().numpy()
        mask = sample.mask.detach().cpu().numpy()
        n_visits, n_features = values.shape
        row: list[float] = []

        for reducer, feature_index in product(
            ("mean", "last", "slope", "observed_fraction"), range(n_features)
        ):
            observed = np.flatnonzero(mask[:, feature_index] > 0)
            if reducer == "observed_fraction":
                row.append(float(observed.size / n_visits))
            elif observed.size == 0:
                row.append(float("nan"))
            elif reducer == "mean":
                row.append(float(values[observed, feature_index].mean()))
            elif reducer == "last":
                row.append(float(values[observed[-1], feature_index]))
            else:
                first, last = int(observed[0]), int(observed[-1])
                span = float(times[last] - times[first])
                slope = (
                    0.0
                    if span <= 1e-6
                    else float((values[last, feature_index] - values[first, feature_index]) / span)
                )
                row.append(slope)

        row.extend((float(n_visits), float(times[-1] - times[0]) if n_visits > 1 else 0.0))
        return row


def dataset_patient_ids(data: ClinicalTimeSeriesDataset) -> tuple[str, ...]:
    """读取并校验冻结dataset中的患者顺序。"""
    raw_ids = data.metadata.get("patient_ids")
    if not isinstance(raw_ids, list) or len(raw_ids) != len(data):
        raise ValueError("dataset metadata缺少与样本数一致的patient_ids")
    return tuple(str(patient_id) for patient_id in raw_ids)


def dataset_survival_arrays(
    data: ClinicalTimeSeriesDataset,
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """按dataset顺序读取事件指示与正随访时间。"""
    event = np.asarray([bool(data[index].event.item()) for index in range(len(data))])
    time = np.asarray(
        [float(data[index].survival_time.item()) for index in range(len(data))],
        dtype=np.float64,
    )
    if np.any(time <= 0.0) or not np.isfinite(time).all():
        raise ValueError("dataset包含非正或非有限生存时间")
    return event, time

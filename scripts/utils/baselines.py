"""不同数据工作流共享的基线预测产物与方法协议。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, Self

import numpy as np
from numpy.typing import NDArray

from trails import ClinicalTimeSeriesDataset

BaselineCapability = Literal["cluster", "survival"]


@dataclass(frozen=True)
class BaselinePrediction:
    """保存基线在一个冻结划分上的有序患者级预测。"""

    method_name: str
    patient_ids: tuple[str, ...]
    cluster_labels: NDArray[np.int64] | None = None
    n_clusters: int | None = None
    risk_score: NDArray[np.float64] | None = None
    risk_horizon: float | None = None
    survival_times: NDArray[np.float64] | None = None
    survival_probabilities: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        """在写盘前固定评价脚本依赖的形状与数值语义。"""
        if not self.method_name:
            raise ValueError("method_name不能为空")
        n_patients = len(self.patient_ids)
        if n_patients == 0 or len(set(self.patient_ids)) != n_patients:
            raise ValueError("patient_ids必须非空且唯一")

        has_clusters = self.cluster_labels is not None
        if has_clusters != (self.n_clusters is not None):
            raise ValueError("cluster_labels与n_clusters必须同时提供")
        if self.cluster_labels is not None and self.n_clusters is not None:
            labels = self.cluster_labels
            if labels.shape != (n_patients,) or not np.issubdtype(labels.dtype, np.integer):
                raise ValueError("cluster_labels必须是一维整数数组并与患者数一致")
            if self.n_clusters < 2 or np.any(labels < 0) or np.any(labels >= self.n_clusters):
                raise ValueError("cluster_labels必须位于配置的簇编号范围内")

        survival_fields = (
            self.risk_score,
            self.risk_horizon,
            self.survival_times,
            self.survival_probabilities,
        )
        has_survival = all(value is not None for value in survival_fields)
        if any(value is not None for value in survival_fields) and not has_survival:
            raise ValueError("患者级生存预测字段必须成套提供")
        if has_survival:
            risk = self.risk_score
            times = self.survival_times
            probabilities = self.survival_probabilities
            assert risk is not None and times is not None and probabilities is not None
            if risk.shape != (n_patients,) or not np.isfinite(risk).all():
                raise ValueError("risk_score必须有限且与患者数一致")
            if np.any((risk < 0.0) | (risk > 1.0)):
                raise ValueError("risk_score必须表示固定时间窗事件概率")
            if self.risk_horizon is None or self.risk_horizon <= 0.0:
                raise ValueError("risk_horizon必须为正数")
            if times.ndim != 1 or len(times) == 0 or not np.isfinite(times).all():
                raise ValueError("survival_times必须是一维有限非空数组")
            if np.any(times <= 0.0) or np.any(np.diff(times) <= 0.0):
                raise ValueError("survival_times必须为正且严格递增")
            if probabilities.shape != (n_patients, len(times)):
                raise ValueError("survival_probabilities形状必须为患者数×时间点数")
            if not np.isfinite(probabilities).all() or np.any(
                (probabilities < 0.0) | (probabilities > 1.0)
            ):
                raise ValueError("survival_probabilities必须位于[0, 1]")
            if np.any(np.diff(probabilities, axis=1) > 1e-6):
                raise ValueError("每位患者的生存概率必须随时间单调不增")
        if not has_clusters and not has_survival:
            raise ValueError("基线预测至少需要cluster或survival能力")

    @property
    def capabilities(self) -> frozenset[BaselineCapability]:
        """返回该预测实际包含的评价能力。"""
        capabilities: set[BaselineCapability] = set()
        if self.cluster_labels is not None:
            capabilities.add("cluster")
        if self.risk_score is not None:
            capabilities.add("survival")
        return frozenset(capabilities)

    def save(self, path: Path) -> None:
        """以无pickle的压缩NumPy格式保存患者级预测。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "format_version": np.asarray(1, dtype=np.int64),
            "method_name": np.asarray(self.method_name, dtype=np.str_),
            "patient_ids": np.asarray(self.patient_ids, dtype=np.str_),
        }
        for name in (
            "cluster_labels",
            "n_clusters",
            "risk_score",
            "risk_horizon",
            "survival_times",
            "survival_probabilities",
        ):
            value = getattr(self, name)
            if value is not None:
                payload[name] = np.asarray(value)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: Path) -> BaselinePrediction:
        """读取预测并重新执行完整契约校验。"""
        with np.load(path, allow_pickle=False) as payload:
            if int(payload["format_version"].item()) != 1:
                raise ValueError(f"不支持的基线预测格式：{path}")
            files = set(payload.files)

            def optional_array(name: str, dtype: Any) -> NDArray[Any] | None:
                return np.asarray(payload[name], dtype=dtype) if name in files else None

            return cls(
                method_name=str(payload["method_name"].item()),
                patient_ids=tuple(str(value) for value in payload["patient_ids"].tolist()),
                cluster_labels=optional_array("cluster_labels", np.int64),
                n_clusters=(int(payload["n_clusters"].item()) if "n_clusters" in files else None),
                risk_score=optional_array("risk_score", np.float64),
                risk_horizon=(
                    float(payload["risk_horizon"].item()) if "risk_horizon" in files else None
                ),
                survival_times=optional_array("survival_times", np.float64),
                survival_probabilities=optional_array("survival_probabilities", np.float64),
            )


class BaselineMethod(Protocol):
    """工作流编排器调用不同基线实现时依赖的最小接口。"""

    capabilities: frozenset[BaselineCapability]

    def fit(
        self,
        train: ClinicalTimeSeriesDataset,
        validation: ClinicalTimeSeriesDataset,
    ) -> Self: ...

    def predict(
        self,
        data: ClinicalTimeSeriesDataset,
        *,
        prediction_times: NDArray[np.float64],
        risk_horizon: float,
    ) -> BaselinePrediction: ...

    def save_model(self, path: Path) -> None: ...

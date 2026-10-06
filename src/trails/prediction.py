"""TRAILS 单次模型推理的结构化预测结果。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .config import RiskMethod, SurvivalKind
from .diagnostics import LatentDiagnostics
from .survival import SURVIVAL_FORMAT_VERSION, mixture_survival, survival_risk


@dataclass(frozen=True)
class TrailsPrediction:
    """保存潜空间、簇后验及生存 log 参数，不重复执行模型前向推理。

    生存参数形状为 (N,K,2)，末维为 log(scale)、log(shape)。
    常规簇概率为 p(c|z)，可选结局后验仅用于回顾性分析。
    """

    latent_representation: Tensor
    cluster_probabilities: Tensor
    survival_log_params: Tensor
    survival_kind: SurvivalKind = "weibull"
    true_cluster: Tensor | None = None
    survival_conditioned_cluster_probabilities: Tensor | None = None

    def __post_init__(self) -> None:
        if self.survival_kind == "cox":
            raise NotImplementedError("Cluster-specific Cox survival is not implemented.")
        expected = (*self.cluster_probabilities.shape, 2)
        if self.survival_log_params.shape != expected:
            raise ValueError(f"survival_log_params must have shape {expected} (format 3).")

    @property
    def weibull_shape(self) -> Tensor:
        return self.survival_log_params[..., 1].exp()

    @property
    def weibull_scale(self) -> Tensor:
        return self.survival_log_params[..., 0].exp()

    def predict(self) -> Tensor:
        """返回每位患者后验概率最大的簇标签。"""
        return torch.argmax(self.cluster_probabilities, dim=-1).long()

    def predict_proba(self) -> Tensor:
        """返回每位患者对全部簇的后验概率。"""
        return self.cluster_probabilities

    def risk_score(self, horizon: float | None = None, *, method: RiskMethod = "auto") -> Tensor:
        """越大越危险；默认使用混合生存曲线的时间窗事件概率。"""
        return survival_risk(
            self.survival_log_params, self.cluster_probabilities, horizon, method=method
        )

    def survival(self, times: Sequence[float] | Tensor) -> Tensor:
        """返回不依赖结局的混合生存概率，形状为 (患者数, 时间点数)。"""
        time_grid = torch.as_tensor(times, device=self.survival_log_params.device).detach()
        if time_grid.ndim != 1 or time_grid.numel() == 0:
            raise ValueError("times must be a non-empty one-dimensional grid.")
        if not bool(torch.isfinite(time_grid).all()) or bool((time_grid < 0).any()):
            raise ValueError("times must contain finite non-negative values.")
        if time_grid.numel() > 1 and not bool((time_grid[1:] > time_grid[:-1]).all()):
            raise ValueError("times must be strictly increasing.")
        return mixture_survival(self.survival_log_params, self.cluster_probabilities, time_grid)

    def median_survival_time(self) -> Tensor:
        """数值求解混合曲线达到 0.5 的时间，不平均各分量参数。"""
        return torch.exp(-self.risk_score(method="median_survival"))

    def latent_diagnostics(self) -> LatentDiagnostics:
        """返回与现有潜空间诊断产物兼容的张量映射。"""
        diagnostics: LatentDiagnostics = {
            "z": self.latent_representation,
            "cluster_probabilities": self.cluster_probabilities,
            "pred_cluster": self.predict(),
            "sample_index": torch.arange(len(self.cluster_probabilities), dtype=torch.long),
        }
        if self.true_cluster is not None:
            diagnostics["true_cluster"] = self.true_cluster
        return diagnostics

    def save(self, path: str | Path) -> None:
        """保存分量 log 参数、类型及可选回顾性后验。"""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "survival_format_version": SURVIVAL_FORMAT_VERSION,
                "latent_representation": self.latent_representation,
                "cluster_probabilities": self.cluster_probabilities,
                "survival_log_params": self.survival_log_params,
                "survival_kind": self.survival_kind,
                "true_cluster": self.true_cluster,
                "survival_conditioned_cluster_probabilities": (
                    self.survival_conditioned_cluster_probabilities
                ),
            },
            destination,
        )

    @classmethod
    def load(cls, path: str | Path) -> TrailsPrediction:
        """读取版本 3；旧患者特异生存头产物须由原版本代码分析。"""
        payload: dict[str, Any] = torch.load(Path(path), map_location="cpu", weights_only=True)
        if payload.pop("survival_format_version", None) != SURVIVAL_FORMAT_VERSION:
            raise ValueError(
                "Unsupported survival format; load legacy artifacts with the original code."
            )
        return cls(**payload)

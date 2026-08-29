"""TRAILS 单次模型推理的结构化预测结果。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .diagnostics import LatentDiagnostics


@dataclass(frozen=True)
class TrailsPrediction:
    """保存潜空间、簇后验和 Weibull 混合分布参数。

    同一个对象可以派生簇标签、簇概率、风险分数和任意时间网格上的
    生存曲线，避免为每种下游产物重复执行模型前向推理。

    属性：
        latent_representation: 形状为 ``(n_samples, latent_dim)`` 的确定性潜表示。
        cluster_probabilities: 形状为 ``(n_samples, n_clusters)`` 的后验簇概率。
        weibull_shape: 每位患者、每个簇的 Weibull 形状参数。
        weibull_scale: 每位患者、每个簇的 Weibull 尺度参数。
        true_cluster: 数据集提供的可选参考簇标签。
    """

    latent_representation: Tensor
    cluster_probabilities: Tensor
    weibull_shape: Tensor
    weibull_scale: Tensor
    true_cluster: Tensor | None = None

    def predict(self) -> Tensor:
        """返回每位患者后验概率最大的簇标签。"""
        return torch.argmax(self.cluster_probabilities, dim=-1).long()

    def predict_proba(self) -> Tensor:
        """返回每位患者对全部簇的后验概率。"""
        return self.cluster_probabilities

    def risk_score(self) -> Tensor:
        """返回负后验加权 Weibull 尺度作为连续生存风险分数。"""
        expected_scale = torch.sum(
            self.cluster_probabilities * self.weibull_scale,
            dim=-1,
        )
        return -expected_scale

    def survival(self, times: Sequence[float] | Tensor) -> Tensor:
        """返回后验加权 Weibull 混合生存曲线。

        参数：
            times: 非空、有限、非负且严格递增的一维时间网格。

        返回：
            形状为 ``(n_samples, n_times)`` 的生存概率张量。
        """
        time_grid = (
            times.detach().cpu().float()
            if isinstance(times, Tensor)
            else torch.tensor(times, dtype=torch.float32)
        )
        if time_grid.ndim != 1 or time_grid.numel() == 0:
            raise ValueError("times must be a non-empty one-dimensional grid.")
        if not bool(torch.isfinite(time_grid).all()) or bool((time_grid < 0).any()):
            raise ValueError("times must contain finite non-negative values.")
        if time_grid.numel() > 1 and not bool((time_grid[1:] > time_grid[:-1]).all()):
            raise ValueError("times must be strictly increasing.")
        shape = self.weibull_shape.unsqueeze(1)
        scale = self.weibull_scale.unsqueeze(1)
        component_survival = torch.exp(-torch.pow(time_grid.reshape(1, -1, 1) / scale, shape))
        return torch.sum(self.cluster_probabilities.unsqueeze(1) * component_survival, dim=-1)

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
        """将完整预测参数保存到 PyTorch 载荷。"""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "latent_representation": self.latent_representation,
                "cluster_probabilities": self.cluster_probabilities,
                "weibull_shape": self.weibull_shape,
                "weibull_scale": self.weibull_scale,
                "true_cluster": self.true_cluster,
            },
            destination,
        )

    @classmethod
    def load(cls, path: str | Path) -> TrailsPrediction:
        """在 CPU 上读取 :meth:`save` 保存的完整预测参数。"""
        payload: dict[str, Any] = torch.load(
            Path(path),
            map_location="cpu",
            weights_only=True,
        )
        return cls(
            latent_representation=payload["latent_representation"],
            cluster_probabilities=payload["cluster_probabilities"],
            weibull_shape=payload["weibull_shape"],
            weibull_scale=payload["weibull_scale"],
            true_cluster=payload.get("true_cluster"),
        )

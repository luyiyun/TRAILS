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
from .survival import SURVIVAL_FORMAT_VERSION, CoxBaseline, survival_risk


@dataclass(frozen=True)
class TrailsPrediction:
    """保存潜空间、簇后验及生存 log 参数，不重复执行模型前向推理。

    Weibull 的两列依次为 log(scale)、log(shape)；Cox 的一列为 log-risk。
    Cox 基线仅从实际训练子集估计，与预测一同保存。
    """

    latent_representation: Tensor
    cluster_probabilities: Tensor
    survival_log_params: Tensor
    survival_kind: SurvivalKind = "weibull"
    true_cluster: Tensor | None = None
    cox_baseline: CoxBaseline | None = None

    @property
    def weibull_shape(self) -> Tensor:
        if self.survival_kind != "weibull":
            raise ValueError("Cox predictions do not have Weibull shape parameters.")
        return self.survival_log_params[:, 1].exp()

    @property
    def weibull_scale(self) -> Tensor:
        if self.survival_kind != "weibull":
            raise ValueError("Cox predictions do not have Weibull scale parameters.")
        return self.survival_log_params[:, 0].exp()

    def predict(self) -> Tensor:
        """返回每位患者后验概率最大的簇标签。"""
        return torch.argmax(self.cluster_probabilities, dim=-1).long()

    def predict_proba(self) -> Tensor:
        """返回每位患者对全部簇的后验概率。"""
        return self.cluster_probabilities

    def risk_score(self, horizon: float | None = None, *, method: RiskMethod = "auto") -> Tensor:
        """越大越危险；auto 对 Cox 返回 log-risk，对 Weibull 返回时间窗事件概率。"""
        if self.survival_kind == "cox" and method == "event_probability":
            if horizon is None or not 0 < horizon < float("inf"):
                raise ValueError("event_probability requires a positive horizon.")
            return 1.0 - self.survival([horizon])[:, 0]
        return survival_risk(self.survival_log_params, self.survival_kind, horizon, method=method)

    def survival(self, times: Sequence[float] | Tensor) -> Tensor:
        """返回 (患者数, 时间点数) 生存概率；Cox 尾部延续最后基线值，不作参数外推。"""
        time_grid = torch.as_tensor(times).detach().cpu().double()
        if time_grid.ndim != 1 or time_grid.numel() == 0:
            raise ValueError("times must be a non-empty one-dimensional grid.")
        if not bool(torch.isfinite(time_grid).all()) or bool((time_grid < 0).any()):
            raise ValueError("times must contain finite non-negative values.")
        if time_grid.numel() > 1 and not bool((time_grid[1:] > time_grid[:-1]).all()):
            raise ValueError("times must be strictly increasing.")
        if self.survival_kind == "cox":
            if self.cox_baseline is None:
                raise ValueError("Cox survival probabilities require a fitted training baseline.")
            return self.cox_baseline.predict(self.survival_log_params, time_grid)
        log_scale, log_shape = self.survival_log_params.unbind(-1)
        # 直接用 log 参数，避免先 exp 再 log；S(0)=1 显式保留。
        grid = time_grid.to(log_scale.dtype)
        log_hazard = log_shape.exp().unsqueeze(1) * (
            grid.log().unsqueeze(0) - log_scale.unsqueeze(1)
        )
        return torch.exp(-torch.exp(log_hazard))

    def median_survival_time(self) -> Tensor:
        """返回中位时间；Cox 在训练支持内未达到 0.5 时记为 NaN。"""
        if self.survival_kind == "weibull":
            return torch.exp(-self.risk_score(method="median_survival"))
        if self.cox_baseline is None:
            raise ValueError("Cox median survival requires a fitted training baseline.")
        curves = self.survival(self.cox_baseline.time)
        reached = curves <= 0.5
        index = reached.long().argmax(dim=1)
        return torch.where(reached.any(dim=1), self.cox_baseline.time[index], torch.nan)

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
        """保存 log 参数、类型及可选训练基线。"""
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
                "cox_baseline": self.cox_baseline.payload()
                if self.cox_baseline is not None
                else None,
            },
            destination,
        )

    @classmethod
    def load(cls, path: str | Path) -> TrailsPrediction:
        """读取新格式；旧 softplus 参数化产物须由原版本代码分析。"""
        payload: dict[str, Any] = torch.load(Path(path), map_location="cpu", weights_only=True)
        if payload.pop("survival_format_version", None) != SURVIVAL_FORMAT_VERSION:
            raise ValueError(
                "Unsupported survival format; load legacy artifacts with the original code."
            )
        baseline = payload.pop("cox_baseline")
        return cls(
            **payload, cox_baseline=CoxBaseline(**baseline) if baseline is not None else None
        )

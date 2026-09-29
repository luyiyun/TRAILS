"""生存头的 log 参数适配及训练集 Cox 基线；似然计算委托 TorchSurv。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor
from torchsurv.loss.cox import baseline_survival_function, neg_partial_log_likelihood
from torchsurv.loss.weibull import neg_log_likelihood_weibull

from .config import RiskMethod, SurvivalKind

SURVIVAL_FORMAT_VERSION = 2

# TorchSurv 0.2 的通用 eager 校验拒绝全删失批次，但 Weibull 完整似然在此仍有效。
# 使用库支持的脚本化 loss 路径，保留原公式及全删失患者的梯度，不补造事件。
_weibull_nll = torch.jit.script(neg_log_likelihood_weibull)


def survival_loss(log_params: Tensor, event: Tensor, time: Tensor, kind: SurvivalKind) -> Tensor:
    """Weibull 按患者、Cox 按事件归一化；调用方保证 Cox 风险集有效。"""
    event = event.reshape(-1).bool()
    time = time.reshape(-1)
    if not bool(torch.isfinite(log_params).all()):
        raise ValueError("Survival log parameters must be finite.")
    if kind == "weibull":
        return _weibull_nll(log_params, event, time.clamp_min(1e-4), reduction="mean")
    # Cox 部分似然只依赖时间顺序。TorchSurv 0.2 的无并列死亡分支
    # 按行构造风险集，故调用时把同刻删失排在死亡之后；真实时间仍用于基线。
    _, ranks = torch.unique(time, sorted=True, return_inverse=True)
    cox_time = (2 * ranks + 1 + (~event).long()).float()
    log_hazard = log_params.reshape(-1).double()
    log_hazard = log_hazard - log_hazard.detach().mean()
    return (
        neg_partial_log_likelihood(
            log_hazard, event, cox_time, ties_method="efron", reduction="sum"
        )
        / event.sum()
    )


def survival_risk(
    log_params: Tensor,
    kind: SurvivalKind,
    horizon: float | None = None,
    *,
    method: RiskMethod = "auto",
) -> Tensor:
    """无需基线的排序分数；Cox 的绝对概率由预测对象提供。"""
    if method == "auto":
        method = "log_hazard" if kind == "cox" else "event_probability"
    if kind == "cox":
        if method != "log_hazard":
            raise ValueError(
                "Cox probabilities require a fitted baseline; median_survival is Weibull-only."
            )
        return log_params.reshape(-1)
    if method == "median_survival":
        log_scale, log_shape = log_params.double().unbind(-1)
        return -(log_scale + math.log(math.log(2.0)) * torch.exp(-log_shape))
    if method != "event_probability":
        raise ValueError("Weibull supports event_probability or median_survival risk.")
    if horizon is None or not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("event_probability requires a positive horizon.")
    log_scale, log_shape = log_params.unbind(-1)
    cumulative_hazard = torch.exp(torch.exp(log_shape) * (math.log(horizon) - log_scale))
    return -torch.expm1(-cumulative_hazard)


@dataclass(frozen=True)
class CoxBaseline:
    """最佳模型在实际训练子集上拟合的 Breslow 基线（CPU float64）。"""

    time: Tensor
    survival: Tensor
    log_risk_center: Tensor

    @classmethod
    def fit(cls, log_params: Tensor, event: Tensor, time: Tensor) -> CoxBaseline:
        event, time = event.detach().cpu().bool(), time.detach().cpu().double()
        log_risk = log_params.detach().cpu().double().reshape(-1)
        center = log_risk.mean()
        baseline = cast(
            dict[str, Tensor], baseline_survival_function(log_risk - center, event, time)
        )
        return cls(baseline["time"].double(), baseline["baseline_survival"].double(), center)

    def predict(self, log_params: Tensor, times: Tensor) -> Tensor:
        """右连续阶梯；首个观察时间之前 S=1，尾部延续最后估计值。"""
        index = torch.searchsorted(self.time, times.double(), right=True)
        baseline = torch.cat((self.survival.new_ones(1), self.survival))[index]
        relative_risk = torch.exp(log_params.double().reshape(-1) - self.log_risk_center)
        return baseline.unsqueeze(0).pow(relative_risk.unsqueeze(1))

    def payload(self) -> dict[str, Tensor]:
        return {
            "time": self.time,
            "survival": self.survival,
            "log_risk_center": self.log_risk_center,
        }

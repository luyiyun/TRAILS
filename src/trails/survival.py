"""簇特异 Weibull 似然、结局后验和混合生存预测。"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torchsurv.loss.weibull import _cumulative_hazard, log_hazard

from .config import RiskMethod

SURVIVAL_FORMAT_VERSION = 3


def component_log_likelihood(log_params: Tensor, event: Tensor, time: Tensor) -> Tensor:
    """返回 (B,K) 删失对数似然；TorchSurv 的标量 loss 不提供未归约输出。"""
    if not bool(torch.isfinite(log_params).all()):
        raise ValueError("Survival log parameters must be finite.")
    batch_size, n_clusters, _ = log_params.shape
    params = log_params.reshape(-1, 2)
    times = time.reshape(-1).clamp_min(1e-4).repeat_interleave(n_clusters)
    events = event.reshape(-1).bool().repeat_interleave(n_clusters)
    # 私有累积风险接口集中在此适配锁定的 TorchSurv 版本，保留其数值截断。
    likelihood = events * log_hazard(params, times, respective_times=True) - _cumulative_hazard(
        params, times, respective_times=True
    )
    return likelihood.reshape(batch_size, n_clusters)


def survival_log_posterior(cluster_logits: Tensor, log_likelihood: Tensor) -> Tensor:
    """p(c|z,t,delta)；使用原始似然，不缩放任务权重，也不截断梯度。"""
    return torch.log_softmax(cluster_logits + log_likelihood, dim=-1)


def _mixture_survival_at_log_time(
    log_params: Tensor, probabilities: Tensor, log_time: Tensor
) -> Tensor:
    """log_time 为 (N,T) 或可广播的 (1,T)，避免中位数搜索中的时间溢出。"""
    log_scale, log_shape = log_params.unbind(-1)
    log_cumulative_hazard = log_shape.exp().unsqueeze(-1) * (
        log_time.unsqueeze(-2) - log_scale.unsqueeze(-1)
    )
    component_survival = torch.exp(-torch.exp(log_cumulative_hazard))
    return (probabilities.unsqueeze(-1) * component_survival).sum(dim=-2)


def mixture_survival(log_params: Tensor, probabilities: Tensor, times: Tensor) -> Tensor:
    """按 p(c|z) 混合各簇生存曲线，返回 (N,T)，包括 S(0)=1。"""
    times = times.to(device=log_params.device, dtype=log_params.dtype)
    curves = _mixture_survival_at_log_time(log_params, probabilities, times.log().unsqueeze(0))
    return torch.where(times.unsqueeze(0) == 0, torch.ones_like(curves), curves)


def mixture_log_median(log_params: Tensor, probabilities: Tensor) -> Tensor:
    """在分量中位数构成的 log 时间区间内向量化二分求混合中位数。"""
    params = log_params.double()
    weights = probabilities.double()
    log_scale, log_shape = params.unbind(-1)
    component_medians = log_scale + math.log(math.log(2.0)) * torch.exp(-log_shape)
    lower, upper = component_medians.amin(-1), component_medians.amax(-1)
    for _ in range(64):
        middle = lower + (upper - lower) * 0.5
        above = (
            _mixture_survival_at_log_time(params, weights, middle.unsqueeze(-1)).squeeze(-1) > 0.5
        )
        lower = torch.where(above, middle, lower)
        upper = torch.where(above, upper, middle)
    return lower + (upper - lower) * 0.5


def survival_risk(
    log_params: Tensor,
    probabilities: Tensor,
    horizon: float | None = None,
    *,
    method: RiskMethod = "auto",
) -> Tensor:
    """统一混合风险入口：固定时间事件概率，或负 log 中位生存时间。"""
    if method == "median_survival":
        return -mixture_log_median(log_params, probabilities)
    if method not in {"auto", "event_probability"}:
        raise ValueError("Weibull supports event_probability or median_survival risk.")
    if horizon is None or not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("event_probability requires a positive horizon.")
    return 1.0 - mixture_survival(log_params, probabilities, log_params.new_tensor([horizon]))[:, 0]

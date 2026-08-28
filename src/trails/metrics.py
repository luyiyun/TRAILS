"""TRAILS 的重建、生存、聚类损失与评价指标。"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy
from scipy.stats.contingency import crosstab
from sksurv.exceptions import NoComparablePairException
from sksurv.metrics import concordance_index_censored
from torch import Tensor
from torchmetrics import Metric


def masked_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """计算掩码加权的重建平方误差。

    误差在全部维度求和后除以 ``prediction`` 第一维的大小，而不是除以观测
    位置数，以保持当前 ELBO 实现的样本级归一化约定。

    参数：
        prediction: 模型重建值。
        target: 与预测同形状的真实值。
        mask: 与预测同形状的观测权重或指示矩阵。

    返回：
        标量重建损失张量。
    """
    # NOTE: 严格按照ELBO的计算公式，后面的部分是x的对数联合似然函数，其实就是单个
    # x的似然函数之和
    observed = prediction.shape[0]
    # observed = mask.sum().clamp_min(1.0)
    return torch.sum(((prediction - target) ** 2) * mask) / observed


def vade_kl_loss(
    latent: Tensor,
    latent_mean: Tensor,
    latent_log_variance: Tensor,
    cluster_logits: Tensor,
    mixture_logits: Tensor,
    mixture_means: Tensor,
    mixture_log_variances: Tensor,
) -> Tensor:
    """计算 VaDE 后验与高斯混合先验之间的变分 KL 项。

    该项联合考虑 ``q(z|x)``、簇责任度 ``q(c|x)``、混合比例和各高斯分量，
    将聚类结构引入 ELBO。

    参数：
        latent: 从患者后验采样的潜变量。
        latent_mean: 患者后验均值。
        latent_log_variance: 患者后验对数方差。
        cluster_logits: 各患者的后验簇 logits。
        mixture_logits: 高斯混合先验的分量 logits。
        mixture_means: 各混合分量的潜空间均值。
        mixture_log_variances: 各混合分量的潜空间对数方差。

    返回：
        在患者维度取平均的标量 VaDE KL 损失。
    """
    # VaDE KL: q(z|x) 与 MoG prior p(z,c) 的变分距离，聚类结构由该项进入 ELBO。
    log_q_z = gaussian_log_prob(latent, latent_mean, latent_log_variance).sum(dim=-1)
    log_p_z_given_c = gaussian_log_prob(
        latent.unsqueeze(1),
        mixture_means.unsqueeze(0),
        mixture_log_variances.unsqueeze(0),
    ).sum(dim=-1)
    log_p_c = torch.log_softmax(mixture_logits, dim=-1).unsqueeze(0)
    log_q_c = torch.log_softmax(cluster_logits, dim=-1)
    responsibilities = torch.softmax(cluster_logits, dim=-1)
    expected_cluster_kl = torch.sum(
        responsibilities * (log_q_c - log_p_c - log_p_z_given_c),
        dim=-1,
    )
    return torch.mean(log_q_z + expected_cluster_kl)


def gaussian_log_prob(value: Tensor, mean: Tensor, log_variance: Tensor) -> Tensor:
    """逐元素计算对角高斯分布的对数概率。"""
    clamped_log_variance = log_variance
    variance = torch.exp(clamped_log_variance)
    log_two_pi = torch.tensor(math.log(2.0 * math.pi), device=value.device, dtype=value.dtype)
    return -0.5 * (log_two_pi + clamped_log_variance + (value - mean).pow(2) / variance)


def weibull_mixture_negative_log_likelihood(
    cluster_logits: Tensor,
    weibull_shape: Tensor,
    weibull_scale: Tensor,
    survival_time: Tensor,
    event: Tensor,
) -> Tensor:
    """计算由簇后验加权的 Weibull 混合生存负对数似然。

    已发生事件的样本使用密度项，删失样本使用生存函数项，再通过
    ``cluster_logits`` 对各簇分量进行对数空间混合。

    参数：
        cluster_logits: 形状为 ``(batch, n_clusters)`` 的簇 logits。
        weibull_shape: 各患者、各簇的正 Weibull 形状参数。
        weibull_scale: 各患者、各簇的正 Weibull 尺度参数。
        survival_time: 每位患者的随访或事件时间。
        event: 每位患者的事件指示。

    返回：
        批次平均的标量负对数似然。
    """
    time = survival_time.clamp_min(1e-4).unsqueeze(-1)
    event_indicator = event.unsqueeze(-1)
    log_time = torch.log(time)
    log_shape = torch.log(weibull_shape)
    log_scale = torch.log(weibull_scale)
    scaled_time = torch.pow(time / weibull_scale, weibull_shape)
    log_density = (
        log_shape - log_scale + (weibull_shape - 1.0) * (log_time - log_scale) - scaled_time
    )
    log_survival = -scaled_time
    log_component = event_indicator * log_density + (1.0 - event_indicator) * log_survival
    log_mixture = torch.logsumexp(torch.log_softmax(cluster_logits, dim=-1) + log_component, dim=-1)
    return -log_mixture.mean()


def concordance_index(risk_score: Tensor, survival_time: Tensor, event: Tensor) -> float:
    """计算考虑删失的 Harrell C-index。

    风险分数越大表示预测风险越高。少于两个样本、没有事件或不存在可比较样本
    对时返回 ``0.0``。

    参数：
        risk_score: 每位患者的连续风险分数。
        survival_time: 每位患者的随访或事件时间。
        event: 每位患者的事件指示。

    返回：
        ``[0, 1]`` 范围内的 concordance 指标；不可计算时为 ``0.0``。
    """
    risk = risk_score.detach().cpu().reshape(-1).double().numpy()
    time = survival_time.detach().cpu().reshape(-1).double().numpy()
    event_indicator = event.detach().cpu().reshape(-1).bool().numpy()
    if len(time) < 2 or not event_indicator.any():
        return 0.0
    try:
        result = concordance_index_censored(event_indicator, time, risk, tied_tol=1e-8)
    except NoComparablePairException:
        return 0.0
    comparable = int(result[1] + result[2] + result[3])
    return float(result[0]) if comparable > 0 else 0.0


def cluster_assignment_diagnostics(pred_cluster: Tensor, *, n_clusters: int) -> dict[str, float]:
    """汇总预测簇占用、极端簇比例和归一化熵。

    返回字典包含空簇数、最小/最大簇比例和以 ``n_clusters`` 为底的归一化熵。

    参数：
        pred_cluster: 每位患者的整数预测簇标签。
        n_clusters: 预期簇总数，包括可能的空簇。

    返回：
        键为 ``cluster_empty_count``、``cluster_min_fraction``、
        ``cluster_max_fraction`` 和 ``cluster_entropy`` 的诊断字典。
    """
    assignments = pred_cluster.detach().cpu().long()
    counts = torch.bincount(assignments, minlength=n_clusters).float()
    fractions = counts / counts.sum().clamp_min(1.0)
    normalized_entropy = (
        float(np.clip(entropy(counts.numpy(), base=n_clusters), 0.0, 1.0))
        if n_clusters > 1 and counts.sum() > 0
        else 0.0
    )
    return {
        "cluster_empty_count": float(torch.sum(counts == 0).item()),
        "cluster_min_fraction": float(fractions.min().item()),
        "cluster_max_fraction": float(fractions.max().item()),
        "cluster_entropy": normalized_entropy,
    }


class Cindex(Metric):
    """跨批次累积风险、时间和事件并计算 C-index 的 TorchMetrics 指标。"""

    def __init__(self, **kwargs):
        """初始化可在分布式环境中拼接的指标状态。"""
        super().__init__(**kwargs)
        self.add_state("risk", default=[], dist_reduce_fx="cat")
        self.add_state("time", default=[], dist_reduce_fx="cat")
        self.add_state("event", default=[], dist_reduce_fx="cat")

        self.risk: list
        self.time: list
        self.event: list

    def update(self, risk: Tensor, time: Tensor, event: Tensor) -> None:
        """追加一个批次的风险分数、生存时间和事件指示。"""
        self.risk.append(risk.detach())
        self.time.append(time.detach())
        self.event.append(event.detach())

    def compute(self) -> Tensor:
        """拼接全部批次并返回标量 C-index 张量。"""
        risk = torch.cat(self.risk, dim=0).squeeze()
        time = torch.cat(self.time, dim=0)
        event = torch.cat(self.event, dim=0)

        return risk.new_tensor(concordance_index(risk, time, event))


def cluster_accuracy(pred_cluster: Tensor, true_cluster: Tensor) -> float:
    """计算对簇标签排列不敏感的聚类准确率。

    使用 Hungarian 匹配寻找预测簇与真实簇之间的最佳一一对应；空输入返回
    ``0.0``。

    参数：
        pred_cluster: 预测簇标签。
        true_cluster: 参考簇标签。

    返回：
        最佳标签匹配下的正确样本比例。
    """
    prediction = pred_cluster.detach().cpu().long().reshape(-1).numpy()
    target = true_cluster.detach().cpu().long().reshape(-1).numpy()
    n_samples = prediction.shape[0]
    if n_samples == 0:
        return 0.0

    contingency: np.ndarray = crosstab(prediction, target).count  # type: ignore
    row_indices, column_indices = linear_sum_assignment(contingency, maximize=True)
    return float(contingency[row_indices, column_indices].sum() / n_samples)


class ClusteringAccuracy(Metric):
    """跨批次累积预测与真值并计算标签排列不变准确率的指标。"""

    def __init__(self, **kwargs):
        """初始化可在分布式环境中拼接的簇标签状态。"""
        super().__init__(**kwargs)
        self.add_state("pred_cluster", default=[], dist_reduce_fx="cat")
        self.add_state("true_cluster", default=[], dist_reduce_fx="cat")

        self.pred_cluster: list
        self.true_cluster: list

    def update(self, pred_cluster: Tensor, true_cluster: Tensor) -> None:
        """追加一个批次的预测簇和参考簇标签。"""
        self.pred_cluster.append(pred_cluster.detach())
        self.true_cluster.append(true_cluster.detach())

    def compute(self) -> Tensor:
        """拼接全部批次并返回标量聚类准确率张量。"""
        pred_cluster = torch.cat(self.pred_cluster, dim=0)
        true_cluster = torch.cat(self.true_cluster, dim=0)
        return pred_cluster.new_tensor(
            cluster_accuracy(pred_cluster, true_cluster), dtype=torch.float32
        )

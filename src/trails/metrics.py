"""TRAILS 的重建、生存、聚类损失与评价指标。"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy
from scipy.stats.contingency import crosstab
from torch import Tensor
from torchmetrics import Metric
from torchsurv.metrics.auc import Auc
from torchsurv.metrics.brier_score import BrierScore
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.stats.ipcw import get_ipcw
from torchsurv.stats.kaplan_meier import KaplanMeierEstimator


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


def concordance_index(
    risk_score: Tensor,
    survival_time: Tensor,
    event: Tensor,
    *,
    weight: Tensor | None = None,
    tau: float | None = None,
) -> Tensor:
    """由 TorchSurv 计算 Harrell/Uno C-index；保留训练入口无比较对时的 0 约定。"""
    risk = risk_score.reshape(-1)
    time = survival_time.reshape(-1)
    events = event.bool().reshape(-1)
    eligible = events if tau is None else events & (time < tau)
    if len(time) < 2 or not bool(eligible.any()):
        return risk.new_zeros(())
    comparable = time[eligible].min() < time.max() or (
        bool((~events).any()) and time[eligible].min() <= time[~events].max()
    )
    if not comparable:
        return risk.new_zeros(())
    return ConcordanceIndex(tied_tol=1e-8)(
        risk,
        events,
        time,
        weight=weight,
        tmax=None if tau is None else time.new_tensor(tau),
        instate=False,
    )


class SurvivalMetrics:
    """训练集 IPCW 适配；输入须位于同一设备，结果保留设备，不实现指标或 KM 公式。"""

    def __init__(self, train_event: Tensor, train_time: Tensor) -> None:
        self.event = train_event.detach().bool()
        self.time = train_time.detach()
        self.max_time = self.time.max()

    def supported(self, times: Tensor) -> Tensor:
        """零删失概率和超出训练随访范围均不能作为有效 IPCW 支持。"""
        grid = times
        result = (grid <= self.max_time) & (grid >= 0)
        if bool(result.any()):
            result[result.clone()] = get_ipcw(self.event, self.time, grid[result]) > 0
        return result

    def _weights(self, times: Tensor) -> Tensor:
        if not bool(self.supported(times).all()):
            raise ValueError(
                "Training censoring distribution does not support the requested times."
            )
        return get_ipcw(self.event, self.time, times).to(times.dtype)

    def _event_weights(self, event: Tensor, time: Tensor, tau: float | None = None) -> Tensor:
        # 删失者不作为病例；Uno 截断后的患者仍保留为比较对照。
        selected = event if tau is None else event & (time < tau)
        weights = torch.zeros_like(time)
        if bool(selected.any()):
            weights[selected] = self._weights(time[selected])
        return weights

    def cindex(
        self,
        risk: Tensor,
        event: Tensor,
        time: Tensor,
        *,
        tau: float,
    ) -> Tensor:
        events, times = event.bool(), time
        return concordance_index(
            risk, times, events, weight=self._event_weights(events, times, tau), tau=tau
        )

    def auc(
        self,
        risk: Tensor,
        event: Tensor,
        time: Tensor,
        grid: Tensor,
    ) -> tuple[Tensor, Tensor]:
        events, times, points = event.bool(), time, grid
        # 库内部部分临时张量未显式指定设备，使用局部默认设备避免 CPU/GPU 混用。
        with torch.device(risk.device):
            metric = Auc(tied_tol=1e-8)
            values = metric(
                risk,
                events,
                times,
                auc_type="cumulative",
                new_time=points,
                weight=self._event_weights(events, times),
                weight_new_time=self._weights(points),
            )
            if risk.device.type != "cpu" and len(points) > 1:
                # TorchSurv 0.2 integral() 的 KM 默认在 CPU；仅补设备，积分仍调用库原实现。
                km = KaplanMeierEstimator(device=str(risk.device))
                km(events, times)
                return values, metric._integrate_cumulative(km.predict(points), points[-1])
            return values, metric.integral()

    def brier(
        self,
        probabilities: Tensor,
        event: Tensor,
        time: Tensor,
        grid: Tensor,
    ) -> tuple[Tensor, Tensor]:
        events, times, points = event.bool(), time, grid
        if times.max() > self.max_time:
            raise ValueError("Evaluation follow-up exceeds training censoring support.")
        metric = BrierScore()
        values = metric(
            probabilities,
            events,
            times,
            new_time=points,
            weight=self._event_weights(events, times),
            weight_new_time=self._weights(points),
        )
        return values, metric.integral()


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

        return concordance_index(risk, time, event)


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

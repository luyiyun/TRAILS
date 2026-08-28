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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("risk", default=[], dist_reduce_fx="cat")
        self.add_state("time", default=[], dist_reduce_fx="cat")
        self.add_state("event", default=[], dist_reduce_fx="cat")

        self.risk: list
        self.time: list
        self.event: list

    def update(self, risk: Tensor, time: Tensor, event: Tensor) -> None:
        self.risk.append(risk.detach())
        self.time.append(time.detach())
        self.event.append(event.detach())

    def compute(self) -> Tensor:
        risk = torch.cat(self.risk, dim=0).squeeze()
        time = torch.cat(self.time, dim=0)
        event = torch.cat(self.event, dim=0)

        return risk.new_tensor(concordance_index(risk, time, event))


def cluster_accuracy(pred_cluster: Tensor, true_cluster: Tensor) -> float:
    prediction = pred_cluster.detach().cpu().long().reshape(-1).numpy()
    target = true_cluster.detach().cpu().long().reshape(-1).numpy()
    n_samples = prediction.shape[0]
    if n_samples == 0:
        return 0.0

    contingency: np.ndarray = crosstab(prediction, target).count  # type: ignore
    row_indices, column_indices = linear_sum_assignment(contingency, maximize=True)
    return float(contingency[row_indices, column_indices].sum() / n_samples)


class ClusteringAccuracy(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("pred_cluster", default=[], dist_reduce_fx="cat")
        self.add_state("true_cluster", default=[], dist_reduce_fx="cat")

        self.pred_cluster: list
        self.true_cluster: list

    def update(self, pred_cluster: Tensor, true_cluster: Tensor) -> None:
        self.pred_cluster.append(pred_cluster.detach())
        self.true_cluster.append(true_cluster.detach())

    def compute(self) -> Tensor:
        pred_cluster = torch.cat(self.pred_cluster, dim=0)
        true_cluster = torch.cat(self.true_cluster, dim=0)
        return pred_cluster.new_tensor(
            cluster_accuracy(pred_cluster, true_cluster), dtype=torch.float32
        )

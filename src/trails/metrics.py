from __future__ import annotations

import math

import torch
from torch import Tensor
from torchmetrics import Metric


def masked_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    observed = mask.sum().clamp_min(1.0)
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
    comparable = 0
    concordant = 0.0
    n_samples = int(survival_time.shape[0])
    for i in range(n_samples):
        for j in range(n_samples):
            if survival_time[i] < survival_time[j] and event[i] > 0:
                comparable += 1
                if risk_score[i] > risk_score[j]:
                    concordant += 1.0
                elif risk_score[i] == risk_score[j]:
                    concordant += 0.5
    if comparable == 0:
        return 0.0
    return concordant / comparable


def cluster_assignment_diagnostics(pred_cluster: Tensor, *, n_clusters: int) -> dict[str, float]:
    assignments = pred_cluster.detach().cpu().long()
    counts = torch.bincount(assignments, minlength=n_clusters).float()
    fractions = counts / counts.sum().clamp_min(1.0)
    nonzero_fractions = fractions[fractions > 0]
    entropy = -torch.sum(nonzero_fractions * torch.log(nonzero_fractions))
    max_entropy = torch.log(torch.tensor(float(n_clusters))).clamp_min(1e-8)
    normalized_entropy = torch.clamp(entropy / max_entropy, min=0.0, max=1.0)
    return {
        "cluster_empty_count": float(torch.sum(counts == 0).item()),
        "cluster_min_fraction": float(fractions.min().item()),
        "cluster_max_fraction": float(fractions.max().item()),
        "cluster_entropy": float(normalized_entropy.item()),
    }


class Cindex(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tied_tol = 1e-8

        self.add_state("risk", default=[], dist_reduce_fx="cat")
        self.add_state("time", default=[], dist_reduce_fx="cat")
        self.add_state("event", default=[], dist_reduce_fx="cat")

        self.risk: list
        self.time: list
        self.event: list

    def update(self, risk: Tensor, time: Tensor, event: Tensor) -> None:
        self.risk.append(risk)
        self.time.append(time)
        self.event.append(event)

    def compute(self) -> Tensor:
        risk = torch.cat(self.risk, dim=0).squeeze()
        time = torch.cat(self.time, dim=0)
        event = torch.cat(self.event, dim=0)

        comparable = (time < time[:, None]) & (event > 0)
        concordant = (risk > risk[:, None] + self.tied_tol) & comparable
        tied = ((risk - risk[:, None]).abs() <= self.tied_tol) & comparable

        return (concordant.sum() + 0.5 * tied.sum()) / comparable.sum().clamp_min(1.0)

from __future__ import annotations

import math

import torch
from torch import Tensor
from torchmetrics.clustering import AdjustedRandScore, NormalizedMutualInfoScore


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
    clamped_log_variance = log_variance.clamp(min=-12.0, max=12.0)
    variance = torch.exp(clamped_log_variance)
    log_two_pi = torch.tensor(math.log(2.0 * math.pi), device=value.device, dtype=value.dtype)
    return -0.5 * (log_two_pi + clamped_log_variance + (value - mean).pow(2) / variance)


class ClusterMetricAccumulator:
    def __init__(self) -> None:
        self.adjusted_rand = AdjustedRandScore()
        self.normalized_mutual_info = NormalizedMutualInfoScore()

    def reset(self) -> None:
        self.adjusted_rand.reset()
        self.normalized_mutual_info.reset()

    def update(self, predictions: Tensor, target: Tensor) -> None:
        self.adjusted_rand.update(predictions, target)
        self.normalized_mutual_info.update(predictions, target)

    def compute(self) -> dict[str, float]:
        return {
            "ari": float(self.adjusted_rand.compute().detach().cpu()),
            "nmi": float(self.normalized_mutual_info.compute().detach().cpu()),
        }

    def to(self, device: str | torch.device) -> None:
        self.adjusted_rand.to(device)
        self.normalized_mutual_info.to(device)


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

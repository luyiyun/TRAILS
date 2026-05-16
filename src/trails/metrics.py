from __future__ import annotations

import torch
from torch import Tensor


def masked_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    observed = mask.sum().clamp_min(1.0)
    return torch.sum(((prediction - target) ** 2) * mask) / observed


def cluster_balance_loss(cluster_logits: Tensor) -> Tensor:
    probabilities = torch.softmax(cluster_logits, dim=-1)
    mean_probabilities = probabilities.mean(dim=0).clamp_min(1e-8)
    n_clusters = torch.tensor(
        float(probabilities.shape[-1]),
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    return torch.sum(mean_probabilities * (torch.log(mean_probabilities) + torch.log(n_clusters)))


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

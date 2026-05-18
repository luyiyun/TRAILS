from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import DataConfig, ModelConfig


@dataclass(frozen=True)
class TrailsModelOutput:
    reconstruction: Tensor
    latent_mean: Tensor
    latent_log_variance: Tensor
    latent: Tensor
    cluster_logits: Tensor
    cluster_probabilities: Tensor
    weibull_shape: Tensor
    weibull_scale: Tensor


class GRUDCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_decay = nn.Linear(input_size, input_size)
        self.hidden_decay = nn.Linear(input_size, hidden_size)
        self.gru_cell = nn.GRUCell(input_size * 2, hidden_size)

    def forward(
        self,
        x_t: Tensor,
        mask_t: Tensor,
        delta_t: Tensor,
        hidden: Tensor,
        last_observed: Tensor,
        feature_means: Tensor,
    ) -> tuple[Tensor, Tensor]:
        gamma_x = torch.exp(-torch.relu(self.input_decay(delta_t)))
        gamma_h = torch.exp(-torch.relu(self.hidden_decay(delta_t)))
        mean = feature_means.unsqueeze(0).expand_as(x_t)
        decayed_input = gamma_x * last_observed + (1.0 - gamma_x) * mean
        x_hat = mask_t * x_t + (1.0 - mask_t) * decayed_input
        decayed_hidden = gamma_h * hidden
        next_hidden = self.gru_cell(torch.cat([x_hat, mask_t], dim=-1), decayed_hidden)
        next_last_observed = mask_t * x_t + (1.0 - mask_t) * last_observed
        return next_hidden, next_last_observed


class SequencePool(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden_sequence: Tensor, sequence_lengths: Tensor) -> Tensor:
        # SeqPool：只在有效访问上归一化时间权重，再汇总为病人级表示。
        weights = self.attention_weights(hidden_sequence, sequence_lengths)
        return torch.sum(weights.unsqueeze(-1) * hidden_sequence, dim=1)

    def attention_weights(self, hidden_sequence: Tensor, sequence_lengths: Tensor) -> Tensor:
        _batch_size, max_length, _hidden_size = hidden_sequence.shape
        steps = torch.arange(max_length, device=hidden_sequence.device).unsqueeze(0)
        active = steps < sequence_lengths.to(hidden_sequence.device).unsqueeze(1)
        logits = self.score(hidden_sequence).squeeze(-1)
        logits = logits.masked_fill(~active, torch.finfo(logits.dtype).min)
        return torch.softmax(logits, dim=-1)


class GRUDEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = GRUDCell(input_size, hidden_size)
        self.seq_pool = SequencePool(hidden_size)

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        delta_time: Tensor,
        sequence_lengths: Tensor,
        feature_means: Tensor,
    ) -> Tensor:
        batch_size, max_length, _n_features = x.shape
        hidden = x.new_zeros(batch_size, self.hidden_size)
        last_observed = feature_means.unsqueeze(0).expand(batch_size, self.input_size)
        hidden_states: list[Tensor] = []

        for step in range(max_length):
            next_hidden, next_last_observed = self.cell(
                x[:, step],
                mask[:, step],
                delta_time[:, step],
                hidden,
                last_observed,
                feature_means,
            )
            active = (step < sequence_lengths).to(dtype=x.dtype, device=x.device).unsqueeze(-1)
            hidden = active * next_hidden + (1.0 - active) * hidden
            last_observed = active * next_last_observed + (1.0 - active) * last_observed
            hidden_states.append(hidden)

        hidden_sequence = torch.stack(hidden_states, dim=1)
        return self.seq_pool(hidden_sequence, sequence_lengths)


class TrailsSurvVaderModel(nn.Module):
    def __init__(self, data_config: DataConfig, model_config: ModelConfig) -> None:
        super().__init__()
        self.data_config = data_config
        self.model_config = model_config
        self.register_buffer("_feature_means", torch.zeros(data_config.n_features))

        decoder_dropout = model_config.dropout if model_config.n_layers > 1 else 0.0
        self.encoder = GRUDEncoder(data_config.n_features, model_config.encoder_hidden_dim)
        self.latent_mean = nn.Linear(model_config.encoder_hidden_dim, model_config.latent_dim)
        self.latent_log_variance = nn.Linear(
            model_config.encoder_hidden_dim,
            model_config.latent_dim,
        )
        self.decoder_initial = nn.Linear(
            model_config.latent_dim,
            model_config.decoder_hidden_dim * model_config.n_layers,
        )
        self.decoder = nn.GRU(
            input_size=1,
            hidden_size=model_config.decoder_hidden_dim,
            num_layers=model_config.n_layers,
            batch_first=True,
            dropout=decoder_dropout,
        )
        self.reconstruction_head = nn.Linear(
            model_config.decoder_hidden_dim, data_config.n_features
        )
        # VaDE 聚类先验：c ~ Cat(pi), z | c ~ Normal(mu_c, var_c)。
        self.mixture_logits = nn.Parameter(torch.zeros(model_config.n_clusters))
        self.mixture_means = nn.Parameter(
            torch.randn(model_config.n_clusters, model_config.latent_dim) * 0.01
        )
        self.mixture_log_variances = nn.Parameter(
            torch.zeros(model_config.n_clusters, model_config.latent_dim)
        )
        self.survival_head = build_survival_head(model_config)

    def set_feature_means(self, feature_means: Tensor) -> None:
        if feature_means.shape != self.feature_means.shape:
            raise ValueError(
                f"feature_means must have shape {tuple(self.feature_means.shape)}, "
                f"got {tuple(feature_means.shape)}."
            )
        self.feature_means.copy_(feature_means.to(self.feature_means.device))

    @property
    def feature_means(self) -> Tensor:
        return cast(Tensor, self._buffers["_feature_means"])

    def set_mixture_parameters(
        self,
        prior_probabilities: Tensor,
        means: Tensor,
        variances: Tensor,
    ) -> None:
        expected_prior_shape = (self.model_config.n_clusters,)
        expected_component_shape = (self.model_config.n_clusters, self.model_config.latent_dim)
        if prior_probabilities.shape != expected_prior_shape:
            raise ValueError(
                "prior_probabilities must have shape "
                f"{expected_prior_shape}, got {tuple(prior_probabilities.shape)}."
            )
        if means.shape != expected_component_shape:
            raise ValueError(
                f"means must have shape {expected_component_shape}, got {tuple(means.shape)}."
            )
        if variances.shape != expected_component_shape:
            raise ValueError(
                "variances must have shape "
                f"{expected_component_shape}, got {tuple(variances.shape)}."
            )
        with torch.no_grad():
            self.mixture_logits.copy_(
                torch.log(prior_probabilities.to(self.mixture_logits.device).clamp_min(1e-6))
            )
            self.mixture_means.copy_(means.to(self.mixture_means.device))
            self.mixture_log_variances.copy_(
                torch.log(variances.to(self.mixture_log_variances.device).clamp_min(1e-6))
            )

    def forward(
        self,
        *,
        times: Tensor,
        x: Tensor,
        mask: Tensor,
        delta_time: Tensor,
        sequence_lengths: Tensor,
    ) -> TrailsModelOutput:
        hidden = self.encoder(
            x=x,
            mask=mask,
            delta_time=delta_time,
            sequence_lengths=sequence_lengths,
            feature_means=self.feature_means,
        )
        latent_mean = self.latent_mean(hidden)
        latent_log_variance = self.latent_log_variance(hidden).clamp(min=-8.0, max=8.0)
        latent = self._sample_latent(latent_mean, latent_log_variance)
        reconstruction = self._decode(latent, times)
        cluster_logits = self._cluster_logits(latent)
        cluster_probabilities = torch.softmax(cluster_logits, dim=-1)
        survival_raw = self.survival_head(latent).reshape(-1, self.model_config.n_clusters, 2)
        weibull_params = F.softplus(survival_raw) + 1e-3
        return TrailsModelOutput(
            reconstruction=reconstruction,
            latent_mean=latent_mean,
            latent_log_variance=latent_log_variance,
            latent=latent,
            cluster_logits=cluster_logits,
            cluster_probabilities=cluster_probabilities,
            weibull_shape=weibull_params[..., 0],
            weibull_scale=weibull_params[..., 1],
        )

    def _decode(self, latent: Tensor, times: Tensor) -> Tensor:
        batch_size = latent.shape[0]
        initial = self.decoder_initial(latent).reshape(
            self.model_config.n_layers,
            batch_size,
            self.model_config.decoder_hidden_dim,
        )
        decoder_input = times.unsqueeze(-1)
        decoded, _hidden = self.decoder(decoder_input, initial.contiguous())
        return self.reconstruction_head(decoded)

    def _sample_latent(self, mean: Tensor, log_variance: Tensor) -> Tensor:
        if not self.training:
            return mean
        noise = torch.randn_like(mean)
        return mean + noise * torch.exp(0.5 * log_variance)

    def _cluster_logits(self, latent: Tensor) -> Tensor:
        log_prior = torch.log_softmax(self.mixture_logits, dim=-1)
        return log_prior.unsqueeze(0) + self._component_log_prob(latent)

    def _component_log_prob(self, latent: Tensor) -> Tensor:
        centered = latent.unsqueeze(1) - self.mixture_means.unsqueeze(0)
        log_variance = self.mixture_log_variances.unsqueeze(0).clamp(min=-12.0, max=12.0)
        variance = torch.exp(log_variance)
        log_density = -0.5 * (
            torch.log(torch.tensor(2.0 * torch.pi, device=latent.device, dtype=latent.dtype))
            + log_variance
            + centered.pow(2) / variance
        )
        return log_density.sum(dim=-1)


def build_survival_head(model_config: ModelConfig) -> nn.Sequential:
    layers: list[nn.Module] = []
    for _layer in range(model_config.survival_head_hidden_layers):
        layers.append(nn.Linear(model_config.latent_dim, model_config.latent_dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(model_config.latent_dim, model_config.n_clusters * 2))
    return nn.Sequential(*layers)

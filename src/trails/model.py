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


class GRUDEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = GRUDCell(input_size, hidden_size)

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

        return hidden


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
            input_size=data_config.n_features,
            hidden_size=model_config.decoder_hidden_dim,
            num_layers=model_config.n_layers,
            batch_first=True,
            dropout=decoder_dropout,
        )
        self.reconstruction_head = nn.Linear(
            model_config.decoder_hidden_dim, data_config.n_features
        )
        self.cluster_head = nn.Linear(model_config.latent_dim, model_config.n_clusters)
        self.survival_head = nn.Linear(model_config.latent_dim, model_config.n_clusters * 2)

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

    def forward(
        self,
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
        reconstruction = self._decode(latent, x.shape[1], x)
        cluster_logits = self.cluster_head(latent)
        survival_raw = self.survival_head(latent).reshape(-1, self.model_config.n_clusters, 2)
        weibull_shape = F.softplus(survival_raw[..., 0]) + 1e-3
        weibull_scale = F.softplus(survival_raw[..., 1]) + 1e-3
        return TrailsModelOutput(
            reconstruction=reconstruction,
            latent_mean=latent_mean,
            latent_log_variance=latent_log_variance,
            latent=latent,
            cluster_logits=cluster_logits,
            weibull_shape=weibull_shape,
            weibull_scale=weibull_scale,
        )

    def _decode(self, latent: Tensor, max_length: int, reference: Tensor) -> Tensor:
        batch_size = latent.shape[0]
        initial = self.decoder_initial(latent).reshape(
            self.model_config.n_layers,
            batch_size,
            self.model_config.decoder_hidden_dim,
        )
        decoder_input = reference.new_zeros(batch_size, max_length, self.data_config.n_features)
        decoded, _hidden = self.decoder(decoder_input, initial.contiguous())
        return self.reconstruction_head(decoded)

    def _sample_latent(self, mean: Tensor, log_variance: Tensor) -> Tensor:
        if not self.training:
            return mean
        noise = torch.randn_like(mean)
        return mean + noise * torch.exp(0.5 * log_variance)

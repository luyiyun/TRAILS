from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import DataConfig, DecoderConfig, EncoderConfig, EncoderMappingConfig, ModelConfig


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


def sequence_padding_mask(
    sequence_lengths: Tensor,
    max_length: int,
    device: torch.device,
) -> Tensor:
    lengths = sequence_lengths.to(device)
    steps = torch.arange(max_length, device=device).unsqueeze(0)
    return steps >= lengths.unsqueeze(1)


def active_sequence_mask(
    sequence_lengths: Tensor,
    max_length: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    return (
        (~sequence_padding_mask(sequence_lengths, max_length, device)).to(dtype=dtype).unsqueeze(-1)
    )


def visit_time_features(times: Tensor) -> Tensor:
    nonnegative_times = times.clamp_min(0.0)
    return torch.stack(
        [
            times,
            torch.log1p(nonnegative_times),
            torch.sin(times),
            torch.cos(times),
        ],
        dim=-1,
    )


class GRUDInputLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = GRUDCell(input_size, hidden_size)

    def forward(
        self,
        *,
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

        return torch.stack(hidden_states, dim=1)


class MTANInputLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.value_projection = nn.Linear(input_size * 3, hidden_size)
        self.time_projection = nn.Linear(4, hidden_size)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        *,
        times: Tensor,
        x: Tensor,
        mask: Tensor,
        delta_time: Tensor,
        sequence_lengths: Tensor,
        feature_means: Tensor,
    ) -> Tensor:
        mean = feature_means.view(1, 1, -1).expand_as(x)
        imputed = mask * x + (1.0 - mask) * mean
        values = self.value_projection(torch.cat([imputed, mask, delta_time], dim=-1))
        time_embedding = self.time_projection(visit_time_features(times))
        padding_mask = sequence_padding_mask(sequence_lengths, x.shape[1], x.device)
        # mTAN-like 输入层：用观测值/缺失模式作 value，用真实访问时间构造 query/key。
        attended, _weights = self.attention(
            query=time_embedding,
            key=time_embedding,
            value=values,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        attended = self.attention_norm(attended + values)
        encoded = self.ffn_norm(attended + self.ffn(attended))
        return encoded * active_sequence_mask(
            sequence_lengths,
            x.shape[1],
            dtype=x.dtype,
            device=x.device,
        )


class RecurrentMappingLayer(nn.Module):
    def __init__(
        self,
        *,
        kind: str,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if n_layers > 1 else 0.0
        rnn_cls = nn.GRU if kind == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )

    def forward(self, sequence: Tensor, times: Tensor, sequence_lengths: Tensor) -> Tensor:
        del times
        encoded, _hidden = self.rnn(sequence)
        return encoded * active_sequence_mask(
            sequence_lengths,
            sequence.shape[1],
            dtype=sequence.dtype,
            device=sequence.device,
        )


class TransformerMappingLayer(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.time_projection = nn.Linear(4, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, sequence: Tensor, times: Tensor, sequence_lengths: Tensor) -> Tensor:
        padding_mask = sequence_padding_mask(sequence_lengths, sequence.shape[1], sequence.device)
        tokens = self.input_projection(sequence) + self.time_projection(visit_time_features(times))
        encoded = self.transformer(tokens, src_key_padding_mask=padding_mask)
        return encoded * active_sequence_mask(
            sequence_lengths,
            sequence.shape[1],
            dtype=sequence.dtype,
            device=sequence.device,
        )


class TrailsEncoder(nn.Module):
    def __init__(
        self,
        data_config: DataConfig,
        encoder_config: EncoderConfig,
        *,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder_config = encoder_config
        if encoder_config.input.kind == "grud":
            self.input_layer = GRUDInputLayer(
                data_config.n_features,
                encoder_config.input.hidden_dim,
            )
        else:
            self.input_layer = MTANInputLayer(
                data_config.n_features,
                encoder_config.input.hidden_dim,
                encoder_config.input.n_heads,
                dropout,
            )
        self.mapping = build_encoder_mapping(
            encoder_config.mapping,
            input_dim=encoder_config.input.hidden_dim,
            dropout=dropout,
        )
        self.seq_pool = SequencePool(encoder_config.mapping.hidden_dim)

    def forward(
        self,
        *,
        times: Tensor,
        x: Tensor,
        mask: Tensor,
        delta_time: Tensor,
        sequence_lengths: Tensor,
        feature_means: Tensor,
    ) -> Tensor:
        if self.encoder_config.input.kind == "grud":
            input_sequence = self.input_layer(
                x=x,
                mask=mask,
                delta_time=delta_time,
                sequence_lengths=sequence_lengths,
                feature_means=feature_means,
            )
        else:
            input_sequence = self.input_layer(
                times=times,
                x=x,
                mask=mask,
                delta_time=delta_time,
                sequence_lengths=sequence_lengths,
                feature_means=feature_means,
            )
        mapped_sequence = self.mapping(input_sequence, times, sequence_lengths)
        return self.seq_pool(mapped_sequence, sequence_lengths)


def build_encoder_mapping(
    mapping_config: EncoderMappingConfig,
    *,
    input_dim: int,
    dropout: float,
) -> nn.Module:
    if mapping_config.kind in {"gru", "lstm"}:
        return RecurrentMappingLayer(
            kind=mapping_config.kind,
            input_dim=input_dim,
            hidden_dim=mapping_config.hidden_dim,
            n_layers=mapping_config.n_layers,
            dropout=dropout,
        )
    return TransformerMappingLayer(
        input_dim=input_dim,
        hidden_dim=mapping_config.hidden_dim,
        n_layers=mapping_config.n_layers,
        n_heads=mapping_config.n_heads,
        dropout=dropout,
    )


class RecurrentDecoder(nn.Module):
    def __init__(
        self,
        *,
        data_config: DataConfig,
        decoder_config: DecoderConfig,
        latent_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.decoder_config = decoder_config
        recurrent_dropout = dropout if decoder_config.n_layers > 1 else 0.0
        input_size = 1 if decoder_config.conditioning == "initial_state" else latent_dim + 1
        rnn_cls = nn.GRU if decoder_config.kind == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=decoder_config.hidden_dim,
            num_layers=decoder_config.n_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        if decoder_config.conditioning == "initial_state":
            self.initial_hidden = nn.Linear(
                latent_dim,
                decoder_config.hidden_dim * decoder_config.n_layers,
            )
            self.initial_cell = (
                nn.Linear(latent_dim, decoder_config.hidden_dim * decoder_config.n_layers)
                if decoder_config.kind == "lstm"
                else None
            )
        else:
            self.initial_hidden = None
            self.initial_cell = None
        self.reconstruction_head = nn.Linear(decoder_config.hidden_dim, data_config.n_features)

    def forward(self, latent: Tensor, times: Tensor, sequence_lengths: Tensor) -> Tensor:
        del sequence_lengths
        if self.decoder_config.conditioning == "initial_state":
            decoded = self._decode_with_initial_state(latent, times)
        else:
            decoded = self._decode_with_concat_time(latent, times)
        return self.reconstruction_head(decoded)

    def _decode_with_initial_state(self, latent: Tensor, times: Tensor) -> Tensor:
        if self.initial_hidden is None:
            raise RuntimeError("initial hidden layer is required for initial_state decoding.")
        batch_size = latent.shape[0]
        hidden = self.initial_hidden(latent).reshape(
            self.decoder_config.n_layers,
            batch_size,
            self.decoder_config.hidden_dim,
        )
        decoder_input = times.unsqueeze(-1)
        if self.decoder_config.kind == "lstm":
            if self.initial_cell is None:
                raise RuntimeError("initial cell layer is required for LSTM decoding.")
            cell = self.initial_cell(latent).reshape(
                self.decoder_config.n_layers,
                batch_size,
                self.decoder_config.hidden_dim,
            )
            decoded, _state = self.rnn(decoder_input, (hidden.contiguous(), cell.contiguous()))
        else:
            decoded, _state = self.rnn(decoder_input, hidden.contiguous())
        return decoded

    def _decode_with_concat_time(self, latent: Tensor, times: Tensor) -> Tensor:
        repeated_latent = latent.unsqueeze(1).expand(-1, times.shape[1], -1)
        decoder_input = torch.cat([times.unsqueeze(-1), repeated_latent], dim=-1)
        decoded, _state = self.rnn(decoder_input)
        return decoded


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        *,
        data_config: DataConfig,
        decoder_config: DecoderConfig,
        latent_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(latent_dim + 1, decoder_config.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=decoder_config.hidden_dim,
            nhead=decoder_config.n_heads,
            dim_feedforward=decoder_config.hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=decoder_config.n_layers)
        self.reconstruction_head = nn.Linear(decoder_config.hidden_dim, data_config.n_features)

    def forward(self, latent: Tensor, times: Tensor, sequence_lengths: Tensor) -> Tensor:
        repeated_latent = latent.unsqueeze(1).expand(-1, times.shape[1], -1)
        decoder_input = torch.cat([times.unsqueeze(-1), repeated_latent], dim=-1)
        padding_mask = sequence_padding_mask(sequence_lengths, times.shape[1], times.device)
        decoded = self.transformer(
            self.input_projection(decoder_input),
            src_key_padding_mask=padding_mask,
        )
        return self.reconstruction_head(decoded)


def build_decoder(
    *,
    data_config: DataConfig,
    decoder_config: DecoderConfig,
    latent_dim: int,
    dropout: float,
) -> nn.Module:
    if decoder_config.kind in {"gru", "lstm"}:
        return RecurrentDecoder(
            data_config=data_config,
            decoder_config=decoder_config,
            latent_dim=latent_dim,
            dropout=dropout,
        )
    return TransformerDecoder(
        data_config=data_config,
        decoder_config=decoder_config,
        latent_dim=latent_dim,
        dropout=dropout,
    )


class TrailsSurvVaderModel(nn.Module):
    def __init__(self, data_config: DataConfig, model_config: ModelConfig) -> None:
        super().__init__()
        self.data_config = data_config
        self.model_config = model_config
        self.register_buffer("_feature_means", torch.zeros(data_config.n_features))

        self.encoder = TrailsEncoder(
            data_config,
            model_config.encoder,
            dropout=model_config.dropout,
        )
        self.latent_mean = nn.Linear(
            model_config.encoder.mapping.hidden_dim,
            model_config.latent_dim,
        )
        self.latent_log_variance = nn.Linear(
            model_config.encoder.mapping.hidden_dim,
            model_config.latent_dim,
        )
        self.decoder = build_decoder(
            data_config=data_config,
            decoder_config=model_config.decoder,
            latent_dim=model_config.latent_dim,
            dropout=model_config.dropout,
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
            times=times,
            x=x,
            mask=mask,
            delta_time=delta_time,
            sequence_lengths=sequence_lengths,
            feature_means=self.feature_means,
        )
        latent_mean = self.latent_mean(hidden)
        latent_log_variance = self.latent_log_variance(hidden).clamp(min=-8.0, max=8.0)
        latent = self._sample_latent(latent_mean, latent_log_variance)
        reconstruction = self.decoder(latent, times, sequence_lengths)
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

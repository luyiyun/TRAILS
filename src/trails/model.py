from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import DataConfig, DecoderConfig, EncoderConfig, EncoderInputConfig, ModelConfig
from .data import Batch
from .metrics import (
    masked_mse,
    vade_kl_loss,
    weibull_mixture_negative_log_likelihood,
)


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


@dataclass(frozen=True)
class TrailsLossBreakdown:
    loss: Tensor
    reconstruction_loss: Tensor
    survival_loss: Tensor
    vade_kl_loss: Tensor
    reconstruction_loss_weight: Tensor
    survival_loss_weight: Tensor
    vade_kl_loss_weight: Tensor
    reconstruction_log_variance: Tensor | None = None
    survival_log_variance: Tensor | None = None
    vade_kl_log_variance: Tensor | None = None

    def items(self) -> tuple[tuple[str, Tensor], ...]:
        values: list[tuple[str, Tensor | None]] = [
            ("loss", self.loss),
            ("reconstruction_loss", self.reconstruction_loss),
            ("survival_loss", self.survival_loss),
            ("vade_kl_loss", self.vade_kl_loss),
            ("reconstruction_loss_weight", self.reconstruction_loss_weight),
            ("survival_loss_weight", self.survival_loss_weight),
            ("vade_kl_loss_weight", self.vade_kl_loss_weight),
            ("reconstruction_log_variance", self.reconstruction_log_variance),
            ("survival_log_variance", self.survival_log_variance),
            ("vade_kl_log_variance", self.vade_kl_log_variance),
        ]
        return tuple((name, value) for name, value in values if value is not None)


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
    def __init__(
        self,
        input_size: int,
        config: EncoderInputConfig,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = config.hidden_dim
        self.num_ref_points = config.num_ref_points
        time_embedding_dim = config.time_embedding_dim or config.hidden_dim
        self.feature_embedding = nn.Embedding(input_size, config.hidden_dim)
        self.time_embedding = TimeEmbedding(
            embedding_dim=time_embedding_dim,
            learn_embedding=config.learn_time_embedding,
            frequency=config.time_embedding_frequency,
        )
        self.attention = MultiTimeAttention(
            input_dim=config.hidden_dim + 2,
            hidden_dim=config.hidden_dim,
            time_embedding_dim=time_embedding_dim,
            n_heads=config.n_heads,
            dropout=dropout,
        )
        self.attention_norm = nn.LayerNorm(config.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(config.hidden_dim)
        self.reference_times = nn.Buffer(
            torch.linspace(0.0, 1.0, config.num_ref_points, dtype=torch.float32)
        )

    def forward(
        self,
        *,
        times: Tensor,
        x: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, max_length, n_features = x.shape
        if n_features != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, got {n_features}.")
        if times.ndim == 2:
            times = times.unsqueeze(-1).expand_as(x)

        token_mask = mask.reshape(batch_size, max_length * n_features) > 0
        max_observed_time = (times * mask).reshape(batch_size, -1).max(dim=1).values.clamp_min(1e-6)
        normalized_times = times / max_observed_time.view(batch_size, 1, 1)
        key_times = normalized_times.reshape(batch_size, max_length * n_features)
        query_times = (
            self.reference_times.to(device=x.device, dtype=x.dtype)
            .unsqueeze(0)
            .expand(
                batch_size,
                -1,
            )
        )

        feature_index = torch.arange(n_features, device=x.device).repeat(max_length)
        feature_tokens = (
            self.feature_embedding(feature_index).unsqueeze(0).expand(batch_size, -1, -1)
        )
        values = torch.cat(
            [
                x.reshape(batch_size, max_length * n_features, 1),
                mask.reshape(batch_size, max_length * n_features, 1),
                feature_tokens,
            ],
            dim=-1,
        )
        key_embedding = self.time_embedding(key_times)
        query_embedding = self.time_embedding(query_times)
        attended = self.attention(
            query=query_embedding,
            key=key_embedding,
            value=values,
            mask=token_mask,
        )
        encoded = self.attention_norm(attended)
        encoded = self.ffn_norm(encoded + self.ffn(encoded))
        sequence_lengths = torch.full(
            (batch_size,),
            self.num_ref_points,
            dtype=torch.long,
            device=x.device,
        )
        return encoded, query_times, sequence_lengths


class TimeEmbedding(nn.Module):
    def __init__(self, *, embedding_dim: int, learn_embedding: bool, frequency: float) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.learn_embedding = learn_embedding
        self.frequency = frequency
        if learn_embedding:
            self.linear = nn.Linear(1, 1)
            self.periodic = nn.Linear(1, embedding_dim - 1) if embedding_dim > 1 else None
        else:
            self.linear = None
            self.periodic = None

    def forward(self, times: Tensor) -> Tensor:
        if self.learn_embedding:
            expanded = times.unsqueeze(-1)
            linear = self._required_linear(expanded)
            if self.embedding_dim == 1:
                return linear
            periodic = self._required_periodic(expanded)
            return torch.cat([linear, torch.sin(periodic)], dim=-1)
        return self.fixed_time_embedding(times)

    def fixed_time_embedding(self, times: Tensor) -> Tensor:
        position = 48.0 * times.unsqueeze(-1)
        embedding = times.new_zeros(*times.shape, self.embedding_dim)
        div_term = torch.exp(
            torch.arange(0, self.embedding_dim, 2, device=times.device, dtype=times.dtype)
            * -(math.log(self.frequency) / self.embedding_dim)
        )
        embedding[..., 0::2] = torch.sin(position * div_term)
        if self.embedding_dim > 1:
            embedding[..., 1::2] = torch.cos(position * div_term[: embedding[..., 1::2].shape[-1]])
        return embedding

    def _required_linear(self, times: Tensor) -> Tensor:
        if self.linear is None:
            raise RuntimeError("learned time embedding linear layer is not initialized.")
        return self.linear(times)

    def _required_periodic(self, times: Tensor) -> Tensor:
        if self.periodic is None:
            raise RuntimeError("learned time embedding periodic layer is not initialized.")
        return self.periodic(times)


class MultiTimeAttention(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        time_embedding_dim: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if time_embedding_dim % n_heads != 0:
            raise ValueError("time_embedding_dim must be divisible by n_heads.")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_embedding_dim = time_embedding_dim
        self.n_heads = n_heads
        self.head_dim = time_embedding_dim // n_heads
        self.query_projection = nn.Linear(time_embedding_dim, time_embedding_dim)
        self.key_projection = nn.Linear(time_embedding_dim, time_embedding_dim)
        self.output_projection = nn.Linear(input_dim * n_heads, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        *,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor,
    ) -> Tensor:
        batch_size, query_length, _embed_dim = query.shape
        key_length = int(key.shape[1])
        query_heads = self.query_projection(query).view(
            batch_size,
            query_length,
            self.n_heads,
            self.head_dim,
        )
        key_heads = self.key_projection(key).view(
            batch_size,
            key_length,
            self.n_heads,
            self.head_dim,
        )
        query_heads = query_heads.transpose(1, 2)
        key_heads = key_heads.transpose(1, 2)
        scores = torch.matmul(query_heads, key_heads.transpose(-2, -1)) / math.sqrt(
            float(self.head_dim)
        )
        attention = masked_softmax(scores, mask[:, None, None, :])
        attention = self.dropout(attention)
        value_heads = value.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        attended = torch.matmul(attention, value_heads)
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                query_length,
                self.n_heads * self.input_dim,
            )
        )
        return self.output_projection(attended)


def masked_softmax(scores: Tensor, mask: Tensor) -> Tensor:
    mask_float = mask.to(dtype=scores.dtype)
    masked_scores = scores.masked_fill(~mask, -1e9)
    shifted = masked_scores - masked_scores.amax(dim=-1, keepdim=True)
    weights = torch.exp(shifted) * mask_float
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)


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
                encoder_config.input,
                dropout,
            )

        mapping_config = encoder_config.mapping
        if mapping_config.kind in {"gru", "lstm"}:
            self.mapping = RecurrentMappingLayer(
                kind=mapping_config.kind,
                input_dim=encoder_config.input.hidden_dim,
                hidden_dim=mapping_config.hidden_dim,
                n_layers=mapping_config.n_layers,
                dropout=dropout,
            )
        else:
            self.mapping = TransformerMappingLayer(
                input_dim=encoder_config.input.hidden_dim,
                hidden_dim=mapping_config.hidden_dim,
                n_layers=mapping_config.n_layers,
                n_heads=mapping_config.n_heads,
                dropout=dropout,
            )

        self.seq_pool = SequencePool(encoder_config.mapping.hidden_dim)

    def forward(
        self,
        *,
        times: Tensor,
        x: Tensor,
        mask: Tensor,
        delta_time: Tensor | None,
        sequence_lengths: Tensor | None,
        feature_means: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.encoder_config.input.kind == "grud":
            if delta_time is None or sequence_lengths is None:
                raise ValueError("GRUD encoder requires delta_time and sequence_lengths.")
            input_sequence = self.input_layer(
                x=x,
                mask=mask,
                delta_time=delta_time,
                sequence_lengths=sequence_lengths,
                feature_means=feature_means,
            )
            mapping_times = times
            mapping_lengths = sequence_lengths
        else:
            input_sequence, mapping_times, mapping_lengths = self.input_layer(
                times=times,
                x=x,
                mask=mask,
            )
        mapped_sequence = self.mapping(input_sequence, mapping_times, mapping_lengths)
        return self.seq_pool(mapped_sequence, mapping_lengths), mapping_times, mapping_lengths


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

        decoder_config = model_config.decoder
        if decoder_config.kind in {"gru", "lstm"}:
            self.decoder = RecurrentDecoder(
                data_config=data_config,
                decoder_config=decoder_config,
                latent_dim=model_config.latent_dim,
                dropout=model_config.dropout,
            )
        else:
            self.decoder = TransformerDecoder(
                data_config=data_config,
                decoder_config=decoder_config,
                latent_dim=model_config.latent_dim,
                dropout=model_config.dropout,
            )

        # VaDE 聚类先验：c ~ Cat(pi), z | c ~ Normal(mu_c, var_c)。
        if model_config.mixture_logits_trained:
            self.mixture_logits = nn.Parameter(torch.zeros(model_config.n_clusters))
        else:
            self.mixture_logits = nn.Buffer(torch.zeros(model_config.n_clusters))
        self.mixture_means = nn.Parameter(
            torch.randn(model_config.n_clusters, model_config.latent_dim) * 0.01
        )
        self.mixture_log_variances = nn.Parameter(
            torch.zeros(model_config.n_clusters, model_config.latent_dim)
        )
        self.survival_head = build_survival_head(model_config)
        self.loss_log_variances = nn.ParameterDict()
        if model_config.loss.weighting == "uncertainty":
            for name, initial_weight in {
                "reconstruction": model_config.loss.reconstruction_weight,
                "survival": model_config.loss.survival_weight,
                "vade_kl": model_config.loss.cluster_weight,
            }.items():
                initial_log_variance = -math.log(2.0 * initial_weight)
                self.loss_log_variances[name] = nn.Parameter(
                    torch.tensor(initial_log_variance, dtype=torch.float32)
                )

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
        delta_time: Tensor | None = None,
        sequence_lengths: Tensor | None = None,
        feature_lengths: Tensor | None = None,
    ) -> TrailsModelOutput:
        hidden, encoder_times, encoder_sequence_lengths = self.encoder(
            times=times,
            x=x,
            mask=mask,
            delta_time=delta_time,
            sequence_lengths=sequence_lengths,
            feature_means=self.feature_means,
        )
        latent_mean = self.latent_mean(hidden)
        latent_log_variance = self.latent_log_variance(hidden)
        latent = self._sample_latent(latent_mean, latent_log_variance)
        reconstruction = self._decode_reconstruction(
            latent,
            batch_times=times,
            batch_mask=mask,
            aligned_sequence_lengths=sequence_lengths,
            encoder_times=encoder_times,
            encoder_sequence_lengths=encoder_sequence_lengths,
            feature_lengths=feature_lengths,
        )
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

    def _decode_reconstruction(
        self,
        latent: Tensor,
        *,
        batch_times: Tensor,
        batch_mask: Tensor,
        aligned_sequence_lengths: Tensor | None,
        encoder_times: Tensor,
        encoder_sequence_lengths: Tensor,
        feature_lengths: Tensor | None,
    ) -> Tensor:
        if feature_lengths is None:
            if aligned_sequence_lengths is None:
                raise ValueError("Aligned reconstruction requires sequence_lengths.")
            return self.decoder(latent, batch_times, aligned_sequence_lengths)

        del encoder_times, encoder_sequence_lengths, feature_lengths
        batch_size, max_length, n_features = batch_times.shape
        flat_times = batch_times.reshape(batch_size, max_length * n_features)
        flat_lengths = torch.full(
            (batch_size,),
            max_length * n_features,
            dtype=torch.long,
            device=batch_times.device,
        )
        flat_reconstruction = self.decoder(latent, flat_times, flat_lengths)
        feature_index = torch.arange(n_features, device=batch_times.device).repeat(max_length)
        gathered = flat_reconstruction.gather(
            dim=-1,
            index=feature_index.view(1, max_length * n_features, 1).expand(batch_size, -1, 1),
        )
        return gathered.reshape(batch_size, max_length, n_features) * (batch_mask > 0).to(
            dtype=gathered.dtype
        )

    def compute_loss(
        self,
        output: TrailsModelOutput,
        batch: Batch,
        *,
        include_vade_kl: bool,
    ) -> TrailsLossBreakdown:
        reconstruction = masked_mse(output.reconstruction, batch["x"], batch["mask"])
        survival = weibull_mixture_negative_log_likelihood(
            output.cluster_logits,
            output.weibull_shape,
            output.weibull_scale,
            batch["survival_time"],
            batch["event"],
        )
        if include_vade_kl:
            vade_kl = vade_kl_loss(
                output.latent,
                output.latent_mean,
                output.latent_log_variance,
                output.cluster_logits,
                self.mixture_logits,
                self.mixture_means,
                self.mixture_log_variances,
            )
        else:
            vade_kl = reconstruction.new_zeros(())

        if self.model_config.loss.weighting == "fixed":
            return self._compute_fixed_loss_breakdown(
                reconstruction=reconstruction,
                survival=survival,
                vade_kl=vade_kl,
            )

        return self._compute_uncertainty_loss_breakdown(
            reconstruction=reconstruction,
            survival=survival,
            vade_kl=vade_kl,
            include_vade_kl=include_vade_kl,
        )

    def _sample_latent(self, mean: Tensor, log_variance: Tensor) -> Tensor:
        if not self.training:
            return mean
        noise = torch.randn_like(mean)
        return mean + noise * torch.exp(0.5 * log_variance)

    def _compute_fixed_loss_breakdown(
        self,
        *,
        reconstruction: Tensor,
        survival: Tensor,
        vade_kl: Tensor,
    ) -> TrailsLossBreakdown:
        config = self.model_config.loss
        reconstruction_weight = reconstruction.new_tensor(config.reconstruction_weight)
        survival_weight = reconstruction.new_tensor(config.survival_weight)
        vade_kl_weight = reconstruction.new_tensor(config.cluster_weight)
        total = (
            reconstruction_weight * reconstruction
            + survival_weight * survival
            + vade_kl_weight * vade_kl
        )
        return TrailsLossBreakdown(
            loss=total,
            reconstruction_loss=reconstruction,
            survival_loss=survival,
            vade_kl_loss=vade_kl,
            reconstruction_loss_weight=reconstruction_weight,
            survival_loss_weight=survival_weight,
            vade_kl_loss_weight=vade_kl_weight,
        )

    def _compute_uncertainty_loss_breakdown(
        self,
        *,
        reconstruction: Tensor,
        survival: Tensor,
        vade_kl: Tensor,
        include_vade_kl: bool,
    ) -> TrailsLossBreakdown:
        # 多任务不确定性加权：s=log(sigma^2)，用可学习噪声自动调节各 loss 贡献。
        reconstruction_term = self._uncertainty_weighted_loss("reconstruction", reconstruction)
        survival_term = self._uncertainty_weighted_loss("survival", survival)
        total = reconstruction_term + survival_term
        if include_vade_kl:
            total = total + self._uncertainty_weighted_loss("vade_kl", vade_kl)

        reconstruction_weight = 0.5 * torch.exp(-self.loss_log_variances["reconstruction"])
        survival_weight = 0.5 * torch.exp(-self.loss_log_variances["survival"])
        vade_kl_weight = 0.5 * torch.exp(-self.loss_log_variances["vade_kl"])
        return TrailsLossBreakdown(
            loss=total,
            reconstruction_loss=reconstruction,
            survival_loss=survival,
            vade_kl_loss=vade_kl,
            reconstruction_loss_weight=reconstruction_weight,
            survival_loss_weight=survival_weight,
            vade_kl_loss_weight=vade_kl_weight,
            reconstruction_log_variance=self.loss_log_variances["reconstruction"],
            survival_log_variance=self.loss_log_variances["survival"],
            vade_kl_log_variance=(self.loss_log_variances["vade_kl"] if include_vade_kl else None),
        )

    def _uncertainty_weighted_loss(self, name: str, loss: Tensor) -> Tensor:
        log_variance = self.loss_log_variances[name]
        return 0.5 * torch.exp(-log_variance) * loss + 0.5 * log_variance

    def _cluster_logits(self, latent: Tensor) -> Tensor:
        log_prior = torch.log_softmax(self.mixture_logits, dim=-1)
        return log_prior.unsqueeze(0) + self._component_log_prob(latent)

    def _component_log_prob(self, latent: Tensor) -> Tensor:
        centered = latent.unsqueeze(1) - self.mixture_means.unsqueeze(0)
        log_variance = self.mixture_log_variances.unsqueeze(0)
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

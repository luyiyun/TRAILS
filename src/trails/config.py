from __future__ import annotations

import logging
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

LOGGER = logging.getLogger(__name__)

AUTO_BATCH_TARGET_UPDATES = 20
AUTO_BATCH_MIN_SIZE = 16
AUTO_BATCH_MAX_SIZE = 256


def resolve_batch_size(n_samples: int, configured_batch_size: int | None) -> int:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive when resolving batch size.")
    if configured_batch_size is not None:
        return configured_batch_size

    target_size = math.ceil(n_samples / AUTO_BATCH_TARGET_UPDATES)
    power_of_two_size = 1 << (target_size - 1).bit_length()
    bounded_size = min(max(power_of_two_size, AUTO_BATCH_MIN_SIZE), AUTO_BATCH_MAX_SIZE)
    used_batch_size = min(n_samples, bounded_size)
    LOGGER.info("Resolving batch size to %s", used_batch_size)
    return used_batch_size


class DataConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_features: int = Field(default=10, gt=0)


class EncoderInputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["grud", "mtan", "mtan2"] = "grud"
    hidden_dim: int = Field(default=32, gt=0)
    n_heads: int = Field(default=2, gt=0)
    num_ref_points: int = Field(default=16, gt=0)
    learn_time_embedding: bool = True
    time_embedding_dim: int | None = Field(default=None, gt=0)
    time_embedding_frequency: float = Field(default=10.0, gt=0.0)
    time_embedding_kind: Literal["mtan", "projection"] = "mtan"
    value_projection_dim: int = Field(default=1, gt=0)

    @model_validator(mode="after")
    def validate_attention_heads(self) -> EncoderInputConfig:
        if self.kind == "mtan2" and self.hidden_dim % self.n_heads != 0:
            raise ValueError("model.encoder.input.hidden_dim must be divisible by n_heads.")
        resolved_time_dim = (
            self.hidden_dim if self.time_embedding_dim is None else self.time_embedding_dim
        )
        if self.kind in {"mtan", "mtan2"} and resolved_time_dim % self.n_heads != 0:
            raise ValueError("model.encoder.input.time_embedding_dim must be divisible by n_heads.")
        return self


class EncoderMappingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["gru", "lstm", "transformer"] = "gru"
    hidden_dim: int = Field(default=32, gt=0)
    n_layers: int = Field(default=1, gt=0)
    n_heads: int = Field(default=2, gt=0)

    @model_validator(mode="after")
    def validate_attention_heads(self) -> EncoderMappingConfig:
        if self.kind == "transformer" and self.hidden_dim % self.n_heads != 0:
            raise ValueError("model.encoder.mapping.hidden_dim must be divisible by n_heads.")
        return self


class EncoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input: EncoderInputConfig = Field(default_factory=EncoderInputConfig)
    mapping: EncoderMappingConfig = Field(default_factory=EncoderMappingConfig)


class DecoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["gru", "lstm", "transformer"] = "gru"
    conditioning: Literal["initial_state", "concat_time"] = "initial_state"
    hidden_dim: int = Field(default=32, gt=0)
    n_layers: int = Field(default=1, gt=0)
    n_heads: int = Field(default=2, gt=0)

    @model_validator(mode="after")
    def validate_architecture(self) -> DecoderConfig:
        if self.kind == "transformer" and self.conditioning == "initial_state":
            raise ValueError("Transformer decoder only supports conditioning='concat_time'.")
        if self.kind == "transformer" and self.hidden_dim % self.n_heads != 0:
            raise ValueError("model.decoder.hidden_dim must be divisible by n_heads.")
        return self


class LossConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    weighting: Literal["uncertainty", "fixed"] = "uncertainty"
    reconstruction_weight: float = Field(default=1.0, ge=0.0)
    survival_weight: float = Field(default=0.2, ge=0.0)
    cluster_weight: float = Field(default=0.05, ge=0.0)

    @model_validator(mode="after")
    def validate_uncertainty_initial_weights(self) -> LossConfig:
        if self.weighting == "uncertainty" and (
            self.reconstruction_weight <= 0.0
            or self.survival_weight <= 0.0
            or self.cluster_weight <= 0.0
        ):
            raise ValueError("Uncertainty loss weighting requires all initial weights > 0.")
        return self


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    latent_dim: int = Field(default=8, gt=0)
    n_clusters: int = Field(default=3, gt=1)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    survival_head_hidden_layers: int = Field(default=0, ge=0)
    loss: LossConfig = Field(default_factory=LossConfig)
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    decoder: DecoderConfig = Field(default_factory=DecoderConfig)
    mixture_logits_trained: bool = Field(default=False)


class TrainerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    min_epochs: int = Field(default=1, gt=0)
    max_epochs: int = Field(default=10, gt=0)
    batch_size: int | None = Field(default=None, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    warmup_epochs: int = Field(default=1, ge=0)
    gmm_init_iters: int = Field(default=20, ge=0)
    gradient_clip_norm: float | None = Field(default=5.0, gt=0.0)
    device: str = "cpu"
    seed: int = 2026
    valid_size: float = Field(default=0.2, ge=0.0, le=1.0)
    early_stop: bool = Field(default=True)
    early_stopping_patience: int = Field(default=10, gt=0)
    early_stopping_min_delta: float = Field(default=0.0, ge=0.0)
    early_stopping_monitor: Literal["loss", "cindex"] = "loss"


class TrailsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    seed: int = 2026

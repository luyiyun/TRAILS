from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DataConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_features: int = Field(default=10, gt=0)


class EncoderInputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["grud", "mtan"] = "grud"
    hidden_dim: int = Field(default=32, gt=0)
    n_heads: int = Field(default=2, gt=0)

    @model_validator(mode="after")
    def validate_attention_heads(self) -> EncoderInputConfig:
        if self.kind == "mtan" and self.hidden_dim % self.n_heads != 0:
            raise ValueError("model.encoder.input.hidden_dim must be divisible by n_heads.")
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


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    latent_dim: int = Field(default=8, gt=0)
    n_clusters: int = Field(default=3, gt=1)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    survival_head_hidden_layers: int = Field(default=0, ge=0)
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    decoder: DecoderConfig = Field(default_factory=DecoderConfig)


class TrainerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_epochs: int = Field(default=10, gt=0)
    batch_size: int = Field(default=16, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    reconstruction_weight: float = Field(default=1.0, ge=0.0)
    survival_weight: float = Field(default=0.2, ge=0.0)
    cluster_weight: float = Field(default=0.05, ge=0.0)
    warmup_epochs: int = Field(default=1, ge=0)
    gmm_init_iters: int = Field(default=20, ge=0)
    gradient_clip_norm: float | None = Field(default=5.0, gt=0.0)
    device: str = "cpu"
    seed: int = 2026


class TrailsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    seed: int = 2026

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DataConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_features: int = Field(default=10, gt=0)


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    encoder_hidden_dim: int = Field(default=32, gt=0)
    decoder_hidden_dim: int = Field(default=32, gt=0)
    latent_dim: int = Field(default=8, gt=0)
    n_clusters: int = Field(default=3, gt=1)
    n_layers: int = Field(default=1, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    survival_head_hidden_layers: int = Field(default=0, ge=0)


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

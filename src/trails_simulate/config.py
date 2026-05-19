from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trails.config import ModelConfig, TrainerConfig

from .generators import ClinicalTimeSeriesDatasetGeneratorConfig

Command = Literal["experiment", "simulate", "train", "optim"]

OPTIM_PARAM_NAMES = (
    "encoder_input_kind",
    "encoder_mapping_kind",
    "decoder_kind",
    "decoder_conditioning",
    "hidden_dim",
    "latent_dim",
    "n_layers",
    "dropout",
    "survival_head_hidden_layers",
    "learning_rate",
    "batch_size",
    "warmup_epochs",
    "gmm_init_iters",
)


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "quick"
    repeats: int = Field(default=1, gt=0)
    train_size: int = 128
    test_size: int = 64
    seed: int = 20260517


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_root: Path | None = None
    data: Path | None = None
    test_data: Path | None = None
    train_root: Path | None = None


class ArtifactsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    names: tuple[str, ...] = ("all",)
    save: Path | None = None


class LatentEmbeddingDiagnosticsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False


class DiagnosticsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latent_embeddings: LatentEmbeddingDiagnosticsConfig = Field(
        default_factory=LatentEmbeddingDiagnosticsConfig
    )


class SwanLabConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    project: str = "TRAILS"
    experiment: str | None = None
    mode: str | None = None


class FloatSearchRangeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    low: float
    high: float
    log: bool = False

    @model_validator(mode="after")
    def validate_range(self) -> FloatSearchRangeConfig:
        if self.high <= self.low:
            raise ValueError("high must be greater than low.")
        if self.log and self.low <= 0.0:
            raise ValueError("log-scaled float search ranges require low > 0.")
        return self


class IntSearchRangeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    low: int
    high: int

    @model_validator(mode="after")
    def validate_range(self) -> IntSearchRangeConfig:
        if self.high < self.low:
            raise ValueError("high must be greater than or equal to low.")
        return self


class OptimSearchSpaceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    encoder_input_kind: tuple[Literal["grud", "mtan"], ...] = Field(
        default=("grud", "mtan"),
        min_length=1,
    )
    encoder_mapping_kind: tuple[Literal["gru", "lstm", "transformer"], ...] = Field(
        default=("gru", "lstm", "transformer"),
        min_length=1,
    )
    decoder_kind: tuple[Literal["gru", "lstm", "transformer"], ...] = Field(
        default=("gru", "lstm", "transformer"),
        min_length=1,
    )
    decoder_conditioning: tuple[Literal["initial_state", "concat_time"], ...] = Field(
        default=("initial_state", "concat_time"),
        min_length=1,
    )
    hidden_dim: tuple[int, ...] = Field(default=(32, 64, 128), min_length=1)
    latent_dim: tuple[int, ...] = Field(default=(8, 16, 32), min_length=1)
    n_layers: tuple[int, ...] = Field(default=(1, 2, 3), min_length=1)
    dropout: FloatSearchRangeConfig = Field(
        default_factory=lambda: FloatSearchRangeConfig(low=0.0, high=0.3)
    )
    survival_head_hidden_layers: tuple[int, ...] = Field(default=(0, 1, 2), min_length=1)
    batch_size: tuple[int, ...] = Field(default=(128, 256, 512), min_length=1)
    gmm_init_iters: tuple[int, ...] = Field(default=(10, 20, 50), min_length=1)
    learning_rate: FloatSearchRangeConfig = Field(
        default_factory=lambda: FloatSearchRangeConfig(low=1e-4, high=3e-2, log=True)
    )
    warmup_epochs: IntSearchRangeConfig = Field(
        default_factory=lambda: IntSearchRangeConfig(low=0, high=5)
    )

    @model_validator(mode="after")
    def validate_decoder_search_space(self) -> OptimSearchSpaceConfig:
        if "transformer" in self.decoder_kind and "concat_time" not in self.decoder_conditioning:
            raise ValueError(
                "optim.search.decoder_conditioning must include 'concat_time' when "
                "decoder_kind includes 'transformer'."
            )
        return self


class OptimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_trials: int = Field(default=30, gt=0)
    study_name: str = Field(default="optim", min_length=1)
    root: Path = Path("outputs/optim/optim")
    storage: str | None = None
    search: OptimSearchSpaceConfig = Field(default_factory=OptimSearchSpaceConfig)


class ApplicationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: Command = "experiment"
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    simulator: ClinicalTimeSeriesDatasetGeneratorConfig = Field(
        default_factory=ClinicalTimeSeriesDatasetGeneratorConfig
    )
    model: ModelConfig = Field(default_factory=ModelConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig)
    diagnostics: DiagnosticsConfig = Field(default_factory=DiagnosticsConfig)
    swanlab: SwanLabConfig = Field(default_factory=SwanLabConfig)
    optim: OptimConfig = Field(default_factory=OptimConfig)

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trails.config import ModelConfig, TrainerConfig

from .generators import ClinicalTimeSeriesDatasetGeneratorConfig

Command = Literal["baseline", "simulate", "summary", "train", "optim"]
BaselineMethod = Literal["summary_kmeans", "risk_stratified_kmeans", "fpca_kmeans"]

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


class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "quick"
    repeats: int = Field(default=1, gt=0)
    train_size: tuple[int, ...] = Field(default=(128,), min_length=1)
    test_size: tuple[int, ...] = Field(default=(64,), min_length=1)
    seed: int = 20260517
    mechanism_seed: int | None = None
    generator: ClinicalTimeSeriesDatasetGeneratorConfig = Field(
        default_factory=ClinicalTimeSeriesDatasetGeneratorConfig
    )

    @model_validator(mode="after")
    def resolve_mechanism_seed(self) -> SimulationConfig:
        if self.mechanism_seed is None:
            self.mechanism_seed = self.seed
        if len(self.train_size) != len(self.test_size):
            raise ValueError(
                "simulation.train_size and simulation.test_size must have equal length."
            )
        cluster_values = self.generator.n_clusters_tuple_
        if any(value <= 0 for value in (*self.train_size, *self.test_size)):
            raise ValueError("simulation train/test sizes must be positive.")
        if min(
            train + test for train, test in zip(self.train_size, self.test_size, strict=True)
        ) <= max(cluster_values):
            raise ValueError("Every simulated sample size must be greater than every requested K.")
        return self


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_root: Path | None = None
    data: Path | None = None
    test_data: Path | None = None
    train_root: Path | None = None
    save_name: str | None = None


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


class ParallelTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workers: int = Field(default=1, gt=0)
    devices: tuple[str, ...] = ()
    torch_threads: int | None = Field(default=None, gt=0)


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: ModelConfig = Field(default_factory=ModelConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig)
    diagnostics: DiagnosticsConfig = Field(default_factory=DiagnosticsConfig)
    parallel: ParallelTrainingConfig = Field(default_factory=ParallelTrainingConfig)
    swanlab: SwanLabConfig = Field(default_factory=SwanLabConfig)


class BaselineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    methods: tuple[BaselineMethod, ...] = Field(
        default=("summary_kmeans", "risk_stratified_kmeans", "fpca_kmeans"),
        min_length=1,
    )
    n_clusters: int | None = Field(default=None, gt=1)
    kmeans_iters: int = Field(default=50, ge=0)
    ridge_alpha: float = Field(default=1.0, ge=0.0)
    risk_feature_weight: float = Field(default=1.0, ge=0.0)
    fpca_components: int = Field(default=3, gt=0)
    fpca_grid_size: int = Field(default=16, gt=1)

    @model_validator(mode="after")
    def validate_unique_methods(self) -> BaselineConfig:
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("baseline.methods cannot contain duplicates.")
        return self


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

    encoder_input_kind: tuple[Literal["grud", "mtan", "mtan2"], ...] = Field(
        default=("grud", "mtan", "mtan2"),
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
    run_id: str | None = None
    study_name: str = Field(default="optim", min_length=1)
    storage: str | None = None
    search: OptimSearchSpaceConfig = Field(default_factory=OptimSearchSpaceConfig)


class SummaryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_roots: tuple[Path, ...] = ()
    baseline_roots: tuple[Path, ...] = ()
    train_labels: tuple[str, ...] = ()
    baseline_labels: tuple[str, ...] = ()
    metrics: tuple[str, ...] = Field(default=("acc", "ari", "nmi", "cindex"), min_length=1)

    @model_validator(mode="after")
    def validate_labels(self) -> SummaryConfig:
        train_roots = self.train_roots
        baseline_roots = self.baseline_roots
        if self.train_labels and len(self.train_labels) != len(train_roots):
            raise ValueError("summary.train_labels length must match summary train roots.")
        if self.baseline_labels and len(self.baseline_labels) != len(baseline_roots):
            raise ValueError("summary.baseline_labels length must match summary baseline roots.")
        return self


class ApplicationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: Command = "simulate"
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    baseline: BaselineConfig = Field(default_factory=BaselineConfig)
    optim: OptimConfig = Field(default_factory=OptimConfig)
    summary: SummaryConfig = Field(default_factory=SummaryConfig)

    @model_validator(mode="after")
    def validate_command_specific_config(self) -> ApplicationConfig:
        if self.command == "summary":
            if not self.summary.train_roots and not self.summary.baseline_roots:
                raise ValueError(
                    "command=summary requires at least one of summary.train_roots, "
                    "or summary.baseline_roots."
                )
        return self

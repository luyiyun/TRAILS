from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from trails.config import ModelConfig, TrainerConfig


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_root: Path = Path("outputs")
    prefix: str = Field(default="case", min_length=1)
    name: str = Field(default="case", min_length=1)


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


class PatientColumnsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_id: str = "patient_id"
    survival_time: str = "survival_time"
    event: str = "event"
    cluster_label: str = "cluster_label"


class ObservationColumnsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_id: str = "patient_id"
    time: str = "time"
    feature: str = "feature"
    value: str = "value"


class CaseColumnsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patients: PatientColumnsConfig = Field(default_factory=PatientColumnsConfig)
    observations: ObservationColumnsConfig = Field(default_factory=ObservationColumnsConfig)


class CaseOutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: Path = Path("case_dataset.pt")
    dataset_summary: Path = Path("case_dataset_summary.json")
    predictions: Path = Path("predictions.pt")
    patient_clusters: Path = Path("patient_clusters.csv")
    cluster_summary: Path = Path("cluster_summary.csv")
    cluster_feature_summary: Path = Path("cluster_feature_summary.csv")
    summary: Path = Path("case_summary.json")


class CaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observations_csv: Path = Path("data/case/observations.csv")
    patients_csv: Path = Path("data/case/patients.csv")
    description: str = "TRAILS case-study dataset"
    feature_order: tuple[str, ...] = ()
    outputs: CaseOutputConfig = Field(default_factory=CaseOutputConfig)
    columns: CaseColumnsConfig = Field(default_factory=CaseColumnsConfig)


class CaseApplicationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    command: Literal["case"] = "case"
    run: RunConfig = Field(default_factory=RunConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    case: CaseConfig = Field(default_factory=CaseConfig)

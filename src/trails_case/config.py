from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from trails_simulate.config import (
    ArtifactsConfig,
    DiagnosticsConfig,
    LatentEmbeddingDiagnosticsConfig,
    OutputPathsConfig,
    ParallelTrainingConfig,
    SwanLabConfig,
    TrainingConfig,
)

__all__ = [
    "ArtifactsConfig",
    "CaseApplicationConfig",
    "CaseConfig",
    "DiagnosticsConfig",
    "LatentEmbeddingDiagnosticsConfig",
    "ParallelTrainingConfig",
    "PathsConfig",
    "SwanLabConfig",
    "TrainingConfig",
]


class PathsConfig(OutputPathsConfig):
    root: Path = Path("outputs/case")
    prefix: str = Field(default="case", min_length=1)
    suffix: str = Field(default="default", min_length=1)
    dir: Path = Path("outputs/case/case-default")


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


class CaseApplicationConfig(TrainingConfig, CaseConfig):
    model_config = ConfigDict(extra="forbid")

    command: Literal["case"] = "case"
    name: str = "case"
    paths: PathsConfig = Field(default_factory=PathsConfig)

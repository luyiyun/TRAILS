"""不同数据工作流共用的冻结划分和候选建模配置。"""

from pathlib import Path
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from trails_simulate.config import OutputPathsConfig, TrainingConfig


class FrozenSplitConfig(BaseModel):
    """冻结数据的引用；路径和种子由数据集配置提供。"""

    model_config = ConfigDict(extra="forbid")
    seed: int
    dir: Path
    strategy: str | None = None


class CandidateRunConfig(TrainingConfig):
    """由公共评价器训练全部候选，不决定代表模型。"""

    split: FrozenSplitConfig
    n_clusters: (
        Annotated[int, Field(ge=2)]
        | Annotated[list[Annotated[int, Field(ge=2)]], Field(min_length=1)]
    )
    seeds: list[int] = Field(min_length=1)
    min_cluster_fraction: FiniteFloat = Field(ge=0, le=1)
    paths: OutputPathsConfig
    description: str

    @model_validator(mode="after")
    def validate_candidates(self) -> Self:
        if isinstance(self.n_clusters, list) and len(set(self.n_clusters)) != len(self.n_clusters):
            raise ValueError("n_clusters不能重复")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds不能重复")
        if any(seed < 0 or seed >= 2**32 for seed in self.seeds):
            raise ValueError("seed必须位于[0, 2**32)")
        return self

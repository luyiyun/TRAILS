"""CRC Yunnan划分命令的Pydantic配置模型；默认值由Hydra YAML提供。"""

import math
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator


class SplitPathsConfig(BaseModel):
    """源表、共享主划分与冻结产物路径。"""

    model_config = ConfigDict(extra="forbid")

    preprocessed_root: Path
    patients_csv: Path
    observations_csv: Path
    assignment_root: Path
    dir: Path
    output_dir: Path  # 保留既有CLI覆盖字段，当前不生成额外汇总。
    run_dir: Path


class CohortConfig(BaseModel):
    """基础纳排及最少观察日期要求。"""

    model_config = ConfigDict(extra="forbid")

    min_observation_dates: int = Field(ge=2)
    exclude_followup_before_surgery: bool
    max_outcome_days: FiniteFloat | None = Field(gt=0)


class RandomSplitConfig(BaseModel):
    """随机划分的三套比例。"""

    model_config = ConfigDict(extra="forbid")

    train_fraction: float = Field(gt=0, lt=1)
    validation_fraction: float = Field(gt=0, lt=1)
    test_fraction: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def validate_total(self) -> Self:
        """三套比例共同覆盖完整基础队列。"""
        if not math.isclose(self.train_fraction + self.validation_fraction + self.test_fraction, 1):
            raise ValueError("random三套比例之和须为1")
        return self


class TemporalSplitConfig(BaseModel):
    """按手术年份划分test，再从较早队列随机划分validation。"""

    model_config = ConfigDict(extra="forbid")

    test_start_year: int
    validation_fraction: float = Field(gt=0, lt=1)


class PatientSplitConfig(BaseModel):
    """本次需要生成的划分策略和种子。"""

    model_config = ConfigDict(extra="forbid")

    strategies: list[Literal["random", "temporal"]] = Field(min_length=1)
    seeds: list[Annotated[int, Field(ge=0, lt=2**32)]] = Field(min_length=1)
    random: RandomSplitConfig
    temporal: TemporalSplitConfig

    @model_validator(mode="after")
    def validate_unique_requests(self) -> Self:
        """重复请求会指向同一个冻结目录，应在执行前拒绝。"""
        if len(set(self.strategies)) != len(self.strategies):
            raise ValueError("split.strategies不能重复")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("split.seeds不能重复")
        return self


class FeaturePanelsConfig(BaseModel):
    """显式特征面板，不自动删减或扩充所选名单。"""

    model_config = ConfigDict(extra="forbid")

    panel: str
    panels: dict[str, list[str]]

    @model_validator(mode="after")
    def validate_selected_panel(self) -> Self:
        """只要求实际使用的面板存在、非空且不含重复特征。"""
        if self.panel not in self.panels:
            raise ValueError("features.panel须为已配置的面板名称")
        features = self.panels[self.panel]
        if not features or len(set(features)) != len(features):
            raise ValueError("面板的特征名单须非空且不重复")
        return self


class PreprocessingConfig(BaseModel):
    """训练集拟合的log及缩放步骤。"""

    model_config = ConfigDict(extra="forbid")

    name: str
    log_transform: Literal["none", "auto-log1p"]
    skew_threshold: FiniteFloat
    scaling: Literal["none", "robust", "standard", "minmax"]


class CRCSplitConfig(BaseModel):
    """Hydra解析后的CRC划分配置，在读取数据前完成校验。"""

    model_config = ConfigDict(extra="forbid")

    dataset: Literal["crc_yunnan"]
    version: str
    split_version: str
    outcome: Literal["dfs", "os"]
    landmark_months: FiniteFloat = Field(gt=0)
    preprocessing: PreprocessingConfig
    paths: SplitPathsConfig
    cohort: CohortConfig
    split: PatientSplitConfig
    features: FeaturePanelsConfig

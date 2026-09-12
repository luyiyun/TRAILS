"""不同数据工作流共用的冻结划分 TRAILS 建模配置。"""

from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trails_case.config import CaseApplicationConfig, CaseKSelectionConfig


class FrozenSplitConfig(BaseModel):
    """冻结数据的引用；路径和种子由数据集配置提供。"""

    model_config = ConfigDict(extra="forbid")

    seed: int
    dir: Path
    strategy: str | None = None


class KSelectionConfig(CaseKSelectionConfig):
    """沿用现有 K 选择配置，允许跳过跨 seed 稳定性计算。"""

    compute_stability: bool = True

    @model_validator(mode="after")
    def validate_stability(self) -> Self:
        """关闭稳定性计算时，不接受依赖 ARI 的选择门槛。"""
        if not self.compute_stability and self.min_mean_pairwise_ari is not None:
            raise ValueError("关闭compute_stability时不能设置ARI门槛")
        return self


class TrailsApplicationConfig(CaseApplicationConfig):
    """复用 MIMIC 的固定 K 或自动 K 建模参数及校验规则。"""

    n_clusters: int | None = Field(default=None, ge=2)
    split: FrozenSplitConfig
    # Pydantic 在构造时校验子类字段，保留现有配置继承关系。
    k_selection: KSelectionConfig = Field(  # pyright: ignore[reportIncompatibleVariableOverride]
        default_factory=KSelectionConfig
    )

    @model_validator(mode="after")
    def validate_k_resolution(self) -> Self:
        """自动选择须有明确候选和种子；固定 K 沿用 trainer.seed。"""
        if self.n_clusters is None:
            if not self.k_selection.enabled:
                raise ValueError("n_clusters未指定时必须启用k_selection")
            if not self.k_selection.candidate_clusters:
                raise ValueError("自动K选择要求k_selection.candidate_clusters非空")
            if not self.k_selection.seeds:
                raise ValueError("自动K选择要求k_selection.seeds非空")
            if self.trainer.seed not in self.k_selection.seeds:
                raise ValueError("trainer.seed必须包含在k_selection.seeds中")
        return self

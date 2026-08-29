"""MIMIC 工作流专用的 Hydra 配置模型。"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from trails_case.config import CaseApplicationConfig


class MimicSplitConfig(BaseModel):
    """正式训练所消费的固定患者划分。"""

    model_config = ConfigDict(extra="forbid")

    seed: int = 20260517
    dir: Path = Path("data/real/mimic-iv-3.1/derived/trails_splits/seed-20260517")


class MimicApplicationConfig(CaseApplicationConfig):
    """增加外部患者划分配置的 MIMIC 训练命令配置。"""

    split: MimicSplitConfig = Field(default_factory=MimicSplitConfig)


class MimicEvaluationPathsConfig(BaseModel):
    """正式评价输出目录。"""

    model_config = ConfigDict(extra="forbid")

    dir: Path


class MimicEvaluationConfig(BaseModel):
    """冻结预测的内部留出测试评价配置。"""

    model_config = ConfigDict(extra="forbid")

    input_dir: Path
    auc_times: tuple[float, ...] = (7.0, 14.0, 21.0)
    probability_times: tuple[float, ...] = tuple(float(day) for day in range(1, 28))
    calibration_bins: int = Field(default=10, ge=2)
    trajectory_bin_hours: float = Field(default=4.0, gt=0.0, le=48.0)
    tau: float = 28.0
    paths: MimicEvaluationPathsConfig

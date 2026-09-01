"""MIMIC 工作流专用的 Hydra 配置模型。"""

from itertools import pairwise
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trails_case.config import CaseApplicationConfig


class MimicSplitConfig(BaseModel):
    """正式训练所消费的固定患者划分。"""

    model_config = ConfigDict(extra="forbid")

    seed: int = 20260517
    dir: Path = Path("data/real/mimic-iv-3.1/derived/trails_splits/seed-20260517")


class MimicApplicationConfig(CaseApplicationConfig):
    """增加外部患者划分配置的 MIMIC 训练命令配置。"""

    split: MimicSplitConfig = Field(default_factory=MimicSplitConfig)


class MimicBaselinePathsConfig(BaseModel):
    """MIMIC基线运行的输出目录。"""

    model_config = ConfigDict(extra="forbid")

    dir: Path


class MimicBaselineMethodBaseConfig(BaseModel):
    """一项基线方法的共同标识与重复运行种子。"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    seeds: tuple[int, ...] = Field(default=(20260517,), min_length=1)


class MimicKMeansMethodConfig(MimicBaselineMethodBaseConfig):
    """共享KMeans迭代配置的聚类基线。"""

    kmeans_iters: int = Field(default=100, ge=1)


class SummaryKMeansMethodConfig(MimicKMeansMethodConfig):
    """纵向摘要特征KMeans配置。"""

    kind: Literal["summary_kmeans"] = "summary_kmeans"


class FPCAKMeansMethodConfig(MimicKMeansMethodConfig):
    """固定0–48小时网格的FPCA-KMeans配置。"""

    kind: Literal["fpca_kmeans"] = "fpca_kmeans"
    n_components: int = Field(default=3, ge=1)
    grid_size: int = Field(default=16, ge=2)


class CoxRiskKMeansMethodConfig(MimicKMeansMethodConfig):
    """加入删失感知Cox风险特征的KMeans配置。"""

    kind: Literal["cox_risk_kmeans"] = "cox_risk_kmeans"
    cox_alpha: float = Field(default=0.01, ge=0.0)
    risk_feature_weight: float = Field(default=1.0, gt=0.0)


class TrailsNoSurvivalMethodConfig(MimicBaselineMethodBaseConfig):
    """仅关闭生存损失的TRAILS消融配置。"""

    kind: Literal["trails_no_survival"] = "trails_no_survival"


class CoxPHMethodConfig(MimicBaselineMethodBaseConfig):
    """患者级Cox比例风险基线配置。"""

    kind: Literal["cox_ph"] = "cox_ph"
    alpha: float = Field(default=0.01, ge=0.0)


class RandomSurvivalForestMethodConfig(MimicBaselineMethodBaseConfig):
    """患者级随机生存森林基线配置。"""

    kind: Literal["random_survival_forest"] = "random_survival_forest"
    n_estimators: int = Field(default=500, ge=1)
    min_samples_split: int = Field(default=10, ge=2)
    min_samples_leaf: int = Field(default=15, ge=1)
    max_features: Literal["sqrt", "log2"] | int | float | None = "sqrt"
    n_jobs: int = 1


class MimicJointModelMethodConfig(MimicBaselineMethodBaseConfig):
    """MIMIC landmark时间轴与R后端的共享配置。"""

    landmark_time: float = Field(default=48.0, ge=0.0)
    observation_time_factor: float = Field(default=1.0 / 24.0, gt=0.0)
    survival_time_factor: float = Field(default=1.0, gt=0.0)
    rscript_executable: str = Field(default="Rscript", min_length=1)
    timeout_seconds: float | None = Field(default=None, gt=0.0)


class MPJLCMMMethodConfig(MimicJointModelMethodConfig):
    """lcmm多变量joint latent class model配置。"""

    kind: Literal["mpjlcmm"] = "mpjlcmm"
    max_iterations: int = Field(default=100, ge=1)
    grid_repetitions: int = Field(default=20, ge=1)
    grid_iterations: int = Field(default=15, ge=1)
    n_processes: int = Field(default=1, ge=1)


class JMbayes2MethodConfig(MimicJointModelMethodConfig):
    """JMbayes2多marker动态生存模型配置。"""

    kind: Literal["jmbayes2"] = "jmbayes2"
    n_chains: int = Field(default=3, ge=1)
    n_iter: int = Field(default=3500, ge=1)
    n_burnin: int = Field(default=500, ge=0)
    n_thin: int = Field(default=5, ge=1)
    n_cores: int = Field(default=1, ge=1)
    parallel: Literal["snow", "multicore"] = "snow"
    lme_max_iterations: int = Field(default=100, ge=1)
    n_samples: int = Field(default=200, ge=1)
    n_mcmc: int = Field(default=100, ge=1)
    patient_batch_size: int = Field(default=32, ge=1)

    @model_validator(mode="after")
    def validate_mcmc(self) -> Self:
        """确保burn-in和并行链配置可由JMbayes2执行。"""
        if self.n_burnin >= self.n_iter:
            raise ValueError("JMbayes2的n_burnin必须小于n_iter")
        if self.n_cores > self.n_chains:
            raise ValueError("JMbayes2的n_cores不能超过n_chains")
        return self


class DeepCoxMixturesMethodConfig(MimicBaselineMethodBaseConfig):
    """train-fitted FPCA输入的官方Deep Cox Mixtures配置。"""

    kind: Literal["deep_cox_mixtures"] = "deep_cox_mixtures"
    n_components: int = Field(default=3, ge=1)
    grid_size: int = Field(default=16, ge=2)
    hidden_dims: tuple[Annotated[int, Field(ge=1)], ...] = (50, 100)
    gamma: float = Field(default=10.0, gt=0.0)
    smoothing_factor: float = Field(default=1e-4, ge=0.0)
    use_activation: bool = False
    max_epochs: int = Field(default=300, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    batch_size: int = Field(default=256, ge=1)


class VaDeSCMethodConfig(MimicBaselineMethodBaseConfig):
    """train-fitted FPCA输入的VaDeSC配置。"""

    kind: Literal["vadesc"] = "vadesc"
    n_components: int = Field(default=3, ge=1)
    grid_size: int = Field(default=16, ge=2)
    hidden_dims: tuple[Annotated[int, Field(ge=1)], Annotated[int, Field(ge=1)]] = (50, 100)
    latent_dim: int = Field(default=16, ge=1)
    weibull_shape: float = Field(default=2.0, gt=0.0)
    max_epochs: int = Field(default=300, ge=1)
    patience: int = Field(default=30, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    batch_size: int = Field(default=256, ge=1)
    device: str = Field(default="cuda", min_length=1)


MimicBaselineMethodConfig = Annotated[
    SummaryKMeansMethodConfig
    | FPCAKMeansMethodConfig
    | CoxRiskKMeansMethodConfig
    | TrailsNoSurvivalMethodConfig
    | CoxPHMethodConfig
    | RandomSurvivalForestMethodConfig
    | MPJLCMMMethodConfig
    | JMbayes2MethodConfig
    | DeepCoxMixturesMethodConfig
    | VaDeSCMethodConfig,
    Field(discriminator="kind"),
]


class MimicBaselinesConfig(BaseModel):
    """在07冻结数据上训练全部聚类与患者级生存基线。"""

    model_config = ConfigDict(extra="forbid")

    input_dir: Path
    n_clusters: int | None = Field(default=None, ge=2)
    risk_horizon: float = Field(default=28.0, gt=0.0)
    prediction_times: tuple[float, ...] = Field(min_length=1)
    methods: tuple[MimicBaselineMethodConfig, ...] = Field(min_length=1)
    paths: MimicBaselinePathsConfig

    @model_validator(mode="after")
    def validate_method_contract(self) -> Self:
        """拒绝会覆盖产物或破坏预测时间语义的配置。"""
        names = [method.name for method in self.methods]
        if len(names) != len(set(names)):
            raise ValueError("基线方法name必须唯一")
        for method in self.methods:
            if len(method.seeds) != len(set(method.seeds)):
                raise ValueError(f"{method.name}的seeds不能重复")
        if any(right <= left for left, right in pairwise(self.prediction_times)):
            raise ValueError("prediction_times必须严格递增")
        if self.prediction_times[0] <= 0.0 or self.prediction_times[-1] >= self.risk_horizon:
            raise ValueError("prediction_times必须位于(0, risk_horizon)内")
        return self


class MimicEvaluationPathsConfig(BaseModel):
    """正式评价输出目录。"""

    model_config = ConfigDict(extra="forbid")

    dir: Path


class MimicEvaluationConfig(BaseModel):
    """冻结预测的内部留出测试评价配置。"""

    model_config = ConfigDict(extra="forbid")

    input_dir: Path
    baseline_dirs: tuple[Path, ...] = ()
    auc_times: tuple[float, ...] = (7.0, 14.0, 21.0)
    probability_times: tuple[float, ...] = tuple(float(day) for day in range(1, 28))
    calibration_bins: int = Field(default=10, ge=2)
    trajectory_bin_hours: float = Field(default=4.0, gt=0.0, le=48.0)
    tau: float = 28.0
    paths: MimicEvaluationPathsConfig

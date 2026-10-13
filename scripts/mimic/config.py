"""MIMIC 工作流专用的 Hydra 配置模型。"""

import math
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from scripts.configs import CandidateRunConfig, FrozenSplitConfig
from trails import ModelConfig, TrainerConfig

BASELINE_COVARIATE_COLUMNS = ("age", "gender", "race", "sofa_score")


class MimicPreparationPathsConfig(BaseModel):
    """06的固定输入与输出路径。"""

    model_config = ConfigDict(extra="forbid")
    patients_csv: Path
    observations_csv: Path
    dir: Path


class MimicRandomSplitConfig(BaseModel):
    """随机划分比例。"""

    model_config = ConfigDict(extra="forbid")
    train_fraction: float = Field(gt=0, lt=1)
    validation_fraction: float = Field(gt=0, lt=1)
    test_fraction: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def validate_fractions(self) -> Self:
        if not math.isclose(
            self.train_fraction + self.validation_fraction + self.test_fraction, 1.0
        ):
            raise ValueError("train、validation和test比例之和必须为1")
        return self


class MimicTemporalSplitConfig(BaseModel):
    """年代划分规则。"""

    model_config = ConfigDict(extra="forbid")
    test_start_year: int
    validation_fraction: float = Field(gt=0, lt=1)


class MimicPatientSplitConfig(BaseModel):
    """冻结ID划分设置。"""

    model_config = ConfigDict(extra="forbid")
    strategy: Literal["random", "temporal"]
    seeds: tuple[int, ...] = Field(min_length=1)
    random: MimicRandomSplitConfig
    temporal: MimicTemporalSplitConfig

    @field_validator("seeds")
    @classmethod
    def unique_seeds(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if len(set(values)) != len(values):
            raise ValueError("split.seeds不能重复")
        return values


class MimicPreparationConfig(BaseModel):
    """06在Hydra解析后统一校验；YAML提供默认值。"""

    model_config = ConfigDict(extra="forbid")
    paths: MimicPreparationPathsConfig
    split: MimicPatientSplitConfig
    description: str
    feature_order: tuple[str, ...]

    @field_validator("feature_order")
    @classmethod
    def unique_features(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(values)) != len(values):
            raise ValueError("feature_order不能重复")
        return values


ClusterCount = Annotated[int, Field(ge=2)]
ClusterCounts = ClusterCount | tuple[ClusterCount, ...]


def _requested_clusters(value: ClusterCounts) -> tuple[int, ...]:
    return (value,) if isinstance(value, int) else value


class MimicSplitConfig(FrozenSplitConfig):
    """保留 MIMIC 原有的冻结划分默认值。"""

    seed: int = 20260517
    dir: Path = Path("data/real/mimic-iv-3.1/derived/trails_splits/seed-20260517")


class MimicApplicationConfig(CandidateRunConfig):
    """引用共享建模配置，保留 MIMIC 配置入口。"""

    split: MimicSplitConfig = Field(  # pyright: ignore[reportIncompatibleVariableOverride]
        default_factory=MimicSplitConfig
    )


class MimicBaselinePathsConfig(BaseModel):
    """MIMIC基线运行的输出目录。"""

    model_config = ConfigDict(extra="forbid")

    dir: Path


class MimicBaselineMethodBaseConfig(BaseModel):
    """一项基线方法的共同标识与重复运行种子。"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    seeds: tuple[int, ...] = Field(default=(20260517,), min_length=1)


class MimicClusterMethodBaseConfig(MimicBaselineMethodBaseConfig):
    """需要显式簇数的基线方法。"""

    n_clusters: ClusterCounts

    @property
    def requested_clusters(self) -> tuple[int, ...]:
        """把固定K与候选K统一为只读元组。"""
        return _requested_clusters(self.n_clusters)


class MimicKMeansMethodConfig(MimicClusterMethodBaseConfig):
    """共享KMeans迭代配置的聚类基线。"""

    kmeans_iters: int = Field(default=100, ge=1)
    kmeans_n_init: int = Field(default=20, ge=1)
    silhouette_sample_size: int | None = Field(default=5000, ge=2)


class SummaryKMeansMethodConfig(MimicKMeansMethodConfig):
    """纵向摘要特征KMeans配置。"""

    kind: Literal["summary_kmeans"] = "summary_kmeans"


class UFPCAKMeansMethodConfig(MimicKMeansMethodConfig):
    """固定0–48小时网格的逐变量UFPCA-KMeans配置。"""

    kind: Literal["ufpca_kmeans"] = "ufpca_kmeans"
    n_components: int = Field(default=3, ge=1)
    grid_size: int = Field(default=16, ge=2)


class MFPCAKMeansMethodConfig(MimicKMeansMethodConfig):
    """固定0–48小时网格的FDApy多变量FPCA-KMeans配置。"""

    kind: Literal["mfpca_kmeans"] = "mfpca_kmeans"
    n_components: int = Field(default=3, ge=1)
    grid_size: int = Field(default=16, ge=2)


class CoxRiskKMeansMethodConfig(MimicKMeansMethodConfig):
    """加入删失感知Cox风险特征的KMeans配置。"""

    kind: Literal["cox_risk_kmeans"] = "cox_risk_kmeans"
    cox_alpha: float = Field(default=0.01, ge=0.0)
    risk_feature_weight: float = Field(default=1.0, gt=0.0)


class TrailsNoSurvivalMethodConfig(MimicClusterMethodBaseConfig):
    """仅关闭生存损失的TRAILS消融配置。"""

    kind: Literal["trails_no_survival"] = "trails_no_survival"
    n_clusters: ClusterCounts = 4
    model: ModelConfig = Field(default_factory=ModelConfig, validate_default=True)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig, validate_default=True)

    @field_validator("model")
    @classmethod
    def disable_survival_loss(cls, value: ModelConfig) -> ModelConfig:
        """由方法配置本身固定消融语义。"""
        return value.model_copy(
            update={"loss": value.loss.model_copy(update={"survival_weight": 0.0})}
        )

    @field_validator("trainer")
    @classmethod
    def disable_internal_validation_split(cls, value: TrainerConfig) -> TrainerConfig:
        """08直接传入冻结validation，不再从train内部切分。"""
        return value.model_copy(update={"valid_size": 0.0})


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
    n_clusters: ClusterCounts
    max_iterations: int = Field(default=100, ge=1)
    grid_repetitions: int = Field(default=20, ge=1)
    grid_iterations: int = Field(default=15, ge=1)
    n_processes: int = Field(default=1, ge=1)

    @property
    def requested_clusters(self) -> tuple[int, ...]:
        """把固定K与候选K统一为只读元组。"""
        return _requested_clusters(self.n_clusters)


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


class DeepCoxMixturesMethodConfig(MimicClusterMethodBaseConfig):
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


class VaDeSCMethodConfig(MimicClusterMethodBaseConfig):
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
    | UFPCAKMeansMethodConfig
    | MFPCAKMeansMethodConfig
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
    """在06冻结数据上训练全部聚类与患者级生存基线。"""

    model_config = ConfigDict(extra="forbid")

    split_dir: Path
    model: ModelConfig = Field(default_factory=ModelConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
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
            if isinstance(method, (MimicClusterMethodBaseConfig, MPJLCMMMethodConfig)):
                requested = method.requested_clusters
                if not requested or len(requested) != len(set(requested)):
                    raise ValueError(f"{method.name}的n_clusters不能为空或重复")
        if any(right <= left for left, right in pairwise(self.prediction_times)):
            raise ValueError("prediction_times必须严格递增")
        if (
            self.prediction_times[0] <= 0.0
            or self.prediction_times[-1] >= self.trainer.risk_horizon
        ):
            raise ValueError("prediction_times必须位于(0, risk_horizon)内")
        return self


class MimicEvaluationPathsConfig(BaseModel):
    """独立后处理输出目录。"""

    model_config = ConfigDict(extra="forbid")
    dir: Path


class MimicModelRequest(BaseModel):
    """人工指定的TRAILS候选；label同时用于输出目录。"""

    model_config = ConfigDict(extra="forbid")
    run_dir: Path
    n_clusters: int = Field(ge=2)
    model_seed: int
    label: str = Field(pattern=r"^[A-Za-z0-9_-]+$")


class MimicPostprocessInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_dir: Path


class MimicPostprocessConfig(BaseModel):
    """09和10分别读取冻结产物；选择候选的依据显式记录。"""

    model_config = ConfigDict(extra="forbid")
    inputs: MimicPostprocessInputs
    n_clusters: int = Field(ge=2)
    model_seed: int
    comparison_runs: tuple[MimicModelRequest, ...] = ()
    baseline_dirs: tuple[Path, ...] = ()
    interventions_csv: Path
    selection_basis: Literal["prespecified", "validation", "test_exploratory"]
    auc_times: tuple[float, ...] = (7.0, 14.0, 21.0)
    probability_times: tuple[float, ...] = tuple(float(day) for day in range(1, 28))
    calibration_bins: int = Field(default=10, ge=2)
    trajectory_bin_hours: float = Field(default=4.0, gt=0.0, le=48.0)
    min_cell_patients: int = Field(default=10, ge=1)
    tau: float = Field(default=28.0, gt=0.0)
    paths: MimicEvaluationPathsConfig

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        labels = ["trails", *(item.label for item in self.comparison_runs)]
        if len(set(labels)) != len(labels):
            raise ValueError("TRAILS展示名称不可重复或占用trails")
        for times in (self.auc_times, self.probability_times):
            if (
                len(times) < 2
                or times[0] <= 0
                or times[-1] >= self.tau
                or any(b <= a for a, b in pairwise(times))
            ):
                raise ValueError("报告时间网格须严格递增并位于(0,tau)")
        if not math.isclose(48 / self.trajectory_bin_hours, round(48 / self.trajectory_bin_hours)):
            raise ValueError("trajectory_bin_hours必须整除48")
        if len(set(self.baseline_dirs)) != len(self.baseline_dirs):
            raise ValueError("baseline_dirs不可重复")
        roots = [
            self.inputs.run_dir,
            *(item.run_dir for item in self.comparison_runs),
            *self.baseline_dirs,
        ]
        if any(
            self.paths.dir.resolve() == root.resolve()
            or root.resolve().is_relative_to(self.paths.dir.resolve())
            for root in roots
        ):
            raise ValueError("后处理输出不得覆盖输入运行目录")
        return self


class MimicFigureConfig(MimicPostprocessConfig):
    """综合主图只显示显式选择的一次重复，不按test性能排序。"""

    baseline_seeds: dict[str, int] = Field(default_factory=dict)
    comparison_dataset: Literal["validation", "test"] = "test"
    profile_dataset: Literal["train", "validation", "test"] = "train"
    width_mm: float = Field(default=240, gt=0)
    height_mm: float = Field(default=330, gt=0)
    dpi: int = Field(default=600, ge=72)


class MimicSupplementaryConfig(MimicPostprocessConfig):
    """补充材料保留全部基线重复和公共候选诊断。"""

    umap_seed: int = 20260517

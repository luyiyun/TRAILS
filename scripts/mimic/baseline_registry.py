"""把MIMIC Hydra配置适配到共享基线实现。"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from ..utils.baseline_cox_risk import CoxRiskKMeansBaseline
from ..utils.baseline_dcm import DeepCoxMixturesBaseline
from ..utils.baseline_fpca import FPCAKMeansBaseline
from ..utils.baseline_jmbayes2 import JMbayes2Baseline
from ..utils.baseline_mpjlcmm import MPJLCMMBaseline
from ..utils.baseline_summary import (
    CoxPHBaseline,
    RandomSurvivalForestBaseline,
    SummaryKMeansBaseline,
)
from ..utils.baseline_vadesc import VaDeSCBaseline
from ..utils.baselines import BaselineCapability, BaselineMethod
from .config import (
    CoxPHMethodConfig,
    CoxRiskKMeansMethodConfig,
    DeepCoxMixturesMethodConfig,
    FPCAKMeansMethodConfig,
    JMbayes2MethodConfig,
    MimicBaselineMethodConfig,
    MPJLCMMMethodConfig,
    RandomSurvivalForestMethodConfig,
    SummaryKMeansMethodConfig,
    VaDeSCMethodConfig,
)

# 这其实就是那个进行数据分析的函数，其作为一个RegisterBaseline的一个数据被记录。然后注册修饰器会将
# 包含这个函数的RegisterBaseline注册到BASELINE_REGISTRY中。
BaselineFactory = Callable[[MimicBaselineMethodConfig, int, int, Path], BaselineMethod]


@dataclass(frozen=True)
class RegisteredBaseline:
    """一项MIMIC基线的构造器与评价能力。"""

    capabilities: frozenset[BaselineCapability]
    factory: BaselineFactory


BASELINE_REGISTRY: dict[str, RegisteredBaseline] = {}


def register_baseline(
    kind: str,
    capabilities: Iterable[BaselineCapability],
) -> Callable[[BaselineFactory], BaselineFactory]:
    """注册MIMIC配置适配器，并拒绝重复名称或空能力声明。"""
    normalized = frozenset(capabilities)
    if not kind or not normalized:
        raise ValueError("基线kind和capabilities不能为空")

    def decorator(factory: BaselineFactory) -> BaselineFactory:
        if kind in BASELINE_REGISTRY:
            raise ValueError(f"基线方法重复注册：{kind}")
        BASELINE_REGISTRY[kind] = RegisteredBaseline(normalized, factory)
        return factory

    return decorator


@register_baseline("summary_kmeans", ("cluster",))
def build_summary_kmeans(
    config: MimicBaselineMethodConfig,
    n_clusters: int,
    seed: int,
    work_dir: Path,
) -> SummaryKMeansBaseline:
    """从MIMIC判别配置构造共享Summary-KMeans。"""
    del work_dir
    if not isinstance(config, SummaryKMeansMethodConfig):
        raise TypeError("summary_kmeans注册器收到不匹配的配置")
    return SummaryKMeansBaseline(
        config.name,
        n_clusters,
        seed,
        config.kmeans_iters,
    )


@register_baseline("fpca_kmeans", ("cluster",))
def build_fpca_kmeans(
    config: MimicBaselineMethodConfig,
    n_clusters: int,
    seed: int,
    work_dir: Path,
) -> FPCAKMeansBaseline:
    """构造固定0–48小时绝对时间网格的FPCA-KMeans。"""
    del work_dir
    if not isinstance(config, FPCAKMeansMethodConfig):
        raise TypeError("fpca_kmeans注册器收到不匹配的配置")
    return FPCAKMeansBaseline(
        config.name,
        n_clusters,
        seed,
        config.kmeans_iters,
        config.n_components,
        config.grid_size,
        0.0,
        48.0,
    )


@register_baseline("cox_risk_kmeans", ("cluster",))
def build_cox_risk_kmeans(
    config: MimicBaselineMethodConfig,
    n_clusters: int,
    seed: int,
    work_dir: Path,
) -> CoxRiskKMeansBaseline:
    """从MIMIC配置构造删失感知Cox-risk KMeans。"""
    del work_dir
    if not isinstance(config, CoxRiskKMeansMethodConfig):
        raise TypeError("cox_risk_kmeans注册器收到不匹配的配置")
    return CoxRiskKMeansBaseline(
        config.name,
        n_clusters,
        seed,
        config.kmeans_iters,
        config.cox_alpha,
        config.risk_feature_weight,
    )


@register_baseline("cox_ph", ("survival",))
def build_cox_ph(
    config: MimicBaselineMethodConfig,
    n_clusters: int,
    seed: int,
    work_dir: Path,
) -> CoxPHBaseline:
    """从MIMIC配置构造Cox PH患者级生存基线。"""
    del n_clusters, seed, work_dir
    if not isinstance(config, CoxPHMethodConfig):
        raise TypeError("cox_ph注册器收到不匹配的配置")
    return CoxPHBaseline(config.name, config.alpha)


@register_baseline("random_survival_forest", ("survival",))
def build_random_survival_forest(
    config: MimicBaselineMethodConfig,
    n_clusters: int,
    seed: int,
    work_dir: Path,
) -> RandomSurvivalForestBaseline:
    """从MIMIC配置构造随机生存森林。"""
    del n_clusters, work_dir
    if not isinstance(config, RandomSurvivalForestMethodConfig):
        raise TypeError("random_survival_forest注册器收到不匹配的配置")
    return RandomSurvivalForestBaseline(
        config.name,
        seed,
        config.n_estimators,
        config.min_samples_split,
        config.min_samples_leaf,
        config.max_features,
        config.n_jobs,
    )


@register_baseline("mpjlcmm", ("cluster", "survival"))
def build_mpjlcmm(
    config: MimicBaselineMethodConfig,
    n_clusters: int,
    seed: int,
    work_dir: Path,
) -> MPJLCMMBaseline:
    """构造R端多变量joint latent class model。"""
    if not isinstance(config, MPJLCMMMethodConfig):
        raise TypeError("mpjlcmm注册器收到不匹配的配置")
    return MPJLCMMBaseline(
        config.name,
        n_clusters,
        seed,
        work_dir / "r",
        landmark_time=config.landmark_time,
        observation_time_factor=config.observation_time_factor,
        survival_time_factor=config.survival_time_factor,
        max_iterations=config.max_iterations,
        grid_repetitions=config.grid_repetitions,
        grid_iterations=config.grid_iterations,
        n_processes=config.n_processes,
        rscript_executable=config.rscript_executable,
        timeout_seconds=config.timeout_seconds,
    )


@register_baseline("jmbayes2", ("survival",))
def build_jmbayes2(
    config: MimicBaselineMethodConfig,
    n_clusters: int,
    seed: int,
    work_dir: Path,
) -> JMbayes2Baseline:
    """构造R端JMbayes2动态生存模型。"""
    del n_clusters
    if not isinstance(config, JMbayes2MethodConfig):
        raise TypeError("jmbayes2注册器收到不匹配的配置")
    parallel_options = {"n_cores": config.n_cores, "parallel": config.parallel}
    return JMbayes2Baseline(
        config.name,
        seed,
        work_dir / "r",
        landmark_time=config.landmark_time,
        observation_time_factor=config.observation_time_factor,
        survival_time_factor=config.survival_time_factor,
        fit_options={
            "n_chains": config.n_chains,
            "n_iter": config.n_iter,
            "n_burnin": config.n_burnin,
            "n_thin": config.n_thin,
            "lme_max_iterations": config.lme_max_iterations,
            **parallel_options,
        },
        prediction_options={
            "n_samples": config.n_samples,
            "n_mcmc": config.n_mcmc,
            "patient_batch_size": config.patient_batch_size,
            **parallel_options,
        },
        rscript_executable=config.rscript_executable,
        timeout_seconds=config.timeout_seconds,
    )


@register_baseline("deep_cox_mixtures", ("cluster", "survival"))
def build_deep_cox_mixtures(
    config: MimicBaselineMethodConfig,
    n_clusters: int,
    seed: int,
    work_dir: Path,
) -> DeepCoxMixturesBaseline:
    """构造官方auton-survival Deep Cox Mixtures。"""
    del work_dir
    if not isinstance(config, DeepCoxMixturesMethodConfig):
        raise TypeError("deep_cox_mixtures注册器收到不匹配的配置")
    return DeepCoxMixturesBaseline(
        config.name,
        n_clusters,
        seed,
        config.n_components,
        config.grid_size,
        0.0,
        48.0,
        config.hidden_dims,
        config.gamma,
        config.smoothing_factor,
        config.use_activation,
        config.max_epochs,
        config.learning_rate,
        config.batch_size,
    )


@register_baseline("vadesc", ("cluster", "survival"))
def build_vadesc(
    config: MimicBaselineMethodConfig,
    n_clusters: int,
    seed: int,
    work_dir: Path,
) -> VaDeSCBaseline:
    """构造FPCA输入的VaDeSC深度生存聚类基线。"""
    del work_dir
    if not isinstance(config, VaDeSCMethodConfig):
        raise TypeError("vadesc注册器收到不匹配的配置")
    return VaDeSCBaseline(
        config.name,
        n_clusters,
        seed,
        config.n_components,
        config.grid_size,
        0.0,
        48.0,
        config.hidden_dims,
        config.latent_dim,
        config.weibull_shape,
        config.max_epochs,
        config.patience,
        config.learning_rate,
        config.batch_size,
        config.device,
    )

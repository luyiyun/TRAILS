"""TRAILS 数据、模型和训练流程使用的配置校验模型。"""

from __future__ import annotations

import logging
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

LOGGER = logging.getLogger(__name__)

AUTO_BATCH_TARGET_UPDATES = 20
AUTO_BATCH_MIN_SIZE = 16
AUTO_BATCH_MAX_SIZE = 256


def resolve_batch_size(n_samples: int, configured_batch_size: int | None) -> int:
    """解析显式指定或按保守规则自动确定的训练批大小。

    自动规则以每轮约 20 次参数更新为目标，将批大小向上取整为 2 的幂，并将
    结果限制在 ``[16, 256]`` 和可用样本数以内。显式配置的批大小会原样返回。

    参数：
        n_samples: 训练集中的样本数，必须为正数。
        configured_batch_size: 用户指定的批大小；为 ``None`` 时使用自动规则。

    返回：
        应传给训练数据加载器的批大小。

    异常：
        ValueError: 当 ``n_samples`` 不是正数时抛出。
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive when resolving batch size.")
    if configured_batch_size is not None:
        return configured_batch_size

    target_size = math.ceil(n_samples / AUTO_BATCH_TARGET_UPDATES)
    power_of_two_size = 1 << (target_size - 1).bit_length()
    bounded_size = min(max(power_of_two_size, AUTO_BATCH_MIN_SIZE), AUTO_BATCH_MAX_SIZE)
    used_batch_size = min(n_samples, bounded_size)
    LOGGER.info("Resolving batch size to %s", used_batch_size)
    return used_batch_size


class DataConfig(BaseModel):
    """描述 TRAILS 纵向数据集的特征维度。

    属性：
        n_features: 每位患者纵向观测变量的数量。
    """

    model_config = ConfigDict(frozen=True)

    n_features: int = Field(default=10, gt=0)


class EncoderInputConfig(BaseModel):
    """配置异步观测进入序列编码器的方式。

    属性：
        kind: 输入表示类型，可选 GRU-D、对齐的 mTAN 或按特征紧凑存储的 mTAN2。
        hidden_dim: 输入层表示的宽度。
        n_heads: mTAN 变体使用的时间注意力头数。
        num_ref_points: 注意力共享参考时间点的数量。
        learn_time_embedding: 时间嵌入是否包含可学习项。
        time_embedding_dim: 可选的时间嵌入宽度；``None`` 表示复用 ``hidden_dim``。
        time_embedding_frequency: 周期时间特征的频率尺度。
        time_embedding_kind: 原始 mTAN 嵌入或基于投影的变体。
        value_projection_dim: mTAN2 为每个特征值使用的投影宽度。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["grud", "mtan", "mtan2"] = "grud"
    hidden_dim: int = Field(default=32, gt=0)
    n_heads: int = Field(default=2, gt=0)
    num_ref_points: int = Field(default=16, gt=0)
    learn_time_embedding: bool = True
    time_embedding_dim: int | None = Field(default=None, gt=0)
    time_embedding_frequency: float = Field(default=10.0, gt=0.0)
    time_embedding_kind: Literal["mtan", "projection"] = "mtan"
    value_projection_dim: int = Field(default=1, gt=0)

    @model_validator(mode="after")
    def validate_attention_heads(self) -> EncoderInputConfig:
        """确保注意力嵌入宽度可被注意力头数整除。"""
        if self.kind == "mtan2" and self.hidden_dim % self.n_heads != 0:
            raise ValueError("model.encoder.input.hidden_dim must be divisible by n_heads.")
        resolved_time_dim = (
            self.hidden_dim if self.time_embedding_dim is None else self.time_embedding_dim
        )
        if self.kind in {"mtan", "mtan2"} and resolved_time_dim % self.n_heads != 0:
            raise ValueError("model.encoder.input.time_embedding_dim must be divisible by n_heads.")
        return self


class EncoderMappingConfig(BaseModel):
    """配置输入编码之后应用的时序映射层。

    属性：
        kind: 循环网络或 Transformer 序列映射架构。
        hidden_dim: 映射后访视表示的宽度。
        n_layers: 堆叠映射层的数量。
        n_heads: Transformer 注意力头数。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["gru", "lstm", "transformer"] = "gru"
    hidden_dim: int = Field(default=32, gt=0)
    n_layers: int = Field(default=1, gt=0)
    n_heads: int = Field(default=2, gt=0)

    @model_validator(mode="after")
    def validate_attention_heads(self) -> EncoderMappingConfig:
        """确保 Transformer 隐状态宽度可被注意力头数整除。"""
        if self.kind == "transformer" and self.hidden_dim % self.n_heads != 0:
            raise ValueError("model.encoder.mapping.hidden_dim must be divisible by n_heads.")
        return self


class EncoderConfig(BaseModel):
    """组合异步输入处理与时序序列映射配置。

    属性：
        input: 将原始纵向观测转换为参考时间表示的配置。
        mapping: 将参考时间表示映射为编码器隐序列的配置。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    input: EncoderInputConfig = Field(default_factory=EncoderInputConfig)
    mapping: EncoderMappingConfig = Field(default_factory=EncoderMappingConfig)


class DecoderConfig(BaseModel):
    """配置如何根据潜变量重建纵向访视。

    属性：
        kind: GRU、LSTM 或 Transformer 重建架构。
        conditioning: 通过循环网络初始状态注入潜编码，或在每一步将其与访视
            时间特征拼接。
        hidden_dim: 解码器隐状态宽度。
        n_layers: 堆叠解码层的数量。
        n_heads: Transformer 注意力头数。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["gru", "lstm", "transformer"] = "gru"
    conditioning: Literal["initial_state", "concat_time"] = "initial_state"
    hidden_dim: int = Field(default=32, gt=0)
    n_layers: int = Field(default=1, gt=0)
    n_heads: int = Field(default=2, gt=0)

    @model_validator(mode="after")
    def validate_architecture(self) -> DecoderConfig:
        """拒绝不受支持的条件注入方式和注意力宽度组合。"""
        if self.kind == "transformer" and self.conditioning == "initial_state":
            raise ValueError("Transformer decoder only supports conditioning='concat_time'.")
        if self.kind == "transformer" and self.hidden_dim % self.n_heads != 0:
            raise ValueError("model.decoder.hidden_dim must be divisible by n_heads.")
        return self


class LossConfig(BaseModel):
    """配置多任务重建、生存和聚类损失。

    属性：
        weighting: 学习同方差不确定性权重，或使用固定权重。
        reconstruction_weight: 重建损失的初始权重或固定权重。
        survival_weight: 生存损失的初始权重或固定权重。
        cluster_weight: VaDE 聚类损失的初始权重或固定权重。

    不确定性加权要求重建和聚类初始权重严格为正；生存初始权重可设为零，
    表示从总损失和可学习不确定性参数中移除生存任务。固定加权允许通过零
    权重关闭任意损失分量。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    weighting: Literal["uncertainty", "fixed"] = "uncertainty"
    reconstruction_weight: float = Field(default=1.0, ge=0.0)
    survival_weight: float = Field(default=0.2, ge=0.0)
    cluster_weight: float = Field(default=0.05, ge=0.0)

    @model_validator(mode="after")
    def validate_uncertainty_initial_weights(self) -> LossConfig:
        """要求不确定性加权的重建和聚类初始权重为正。"""
        if self.weighting == "uncertainty" and (
            self.reconstruction_weight <= 0.0 or self.cluster_weight <= 0.0
        ):
            raise ValueError(
                "Uncertainty loss weighting requires reconstruction and cluster weights > 0."
            )
        return self


class ModelConfig(BaseModel):
    """配置完整的 Surv-VaDER 模型架构。

    属性：
        latent_dim: 患者变分表示的宽度。
        n_clusters: 高斯混合分量及患者亚型的数量。
        dropout: 支持该操作的网络层所使用的 dropout 概率。
        survival_head_hidden_layers: 输出患者 Weibull 参数前使用的潜空间等宽
            隐藏层数量。
        loss: 多任务损失加权配置。
        encoder: 异步输入和时序映射配置。
        decoder: 纵向重建配置。
        mixture_logits_trained: 确定性初始化混合模型后，优化器是否继续更新
            混合比例。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    latent_dim: int = Field(default=8, gt=0)
    n_clusters: int = Field(default=3, gt=1)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    survival_head_hidden_layers: int = Field(default=0, ge=0)
    loss: LossConfig = Field(default_factory=LossConfig)
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    decoder: DecoderConfig = Field(default_factory=DecoderConfig)
    mixture_logits_trained: bool = Field(default=False)


class TrainerConfig(BaseModel):
    """配置优化、验证和早停流程。

    属性：
        min_epochs: 允许早停结束训练前至少执行的轮数。
        max_epochs: 最大训练轮数。
        batch_size: 显式批大小；``None`` 表示自动解析。
        learning_rate: 优化器学习率。
        warmup_epochs: 初始化混合模型前仅优化重建目标的轮数。
        gmm_init_iters: 初始化潜空间混合模型时使用的 K-means 迭代次数。
        gradient_clip_norm: 最大梯度范数；``None`` 表示不裁剪。
        device: 执行模型优化的 Torch 设备。
        seed: 训练和数据加载器操作使用的基础随机种子。
        valid_size: 从训练样本中留作验证集的比例。
        early_stop: 监控指标停止改善时是否提前终止训练。
        early_stopping_patience: 达到 ``min_epochs`` 后允许连续无改善的轮数。
        early_stopping_min_delta: 被视为改善所需的最小变化量。
        early_stopping_monitor: 监控总损失、生存损失或 C-index。
        risk_horizon: 计算 C-index 风险排序所用的固定结局时间窗。
    """

    model_config = ConfigDict(frozen=True)

    min_epochs: int = Field(default=1, gt=0)
    max_epochs: int = Field(default=10, gt=0)
    batch_size: int | None = Field(default=None, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    warmup_epochs: int = Field(default=1, ge=0)
    gmm_init_iters: int = Field(default=20, ge=0)
    gradient_clip_norm: float | None = Field(default=5.0, gt=0.0)
    device: str = "cpu"
    seed: int = 2026
    valid_size: float = Field(default=0.2, ge=0.0, le=1.0)
    early_stop: bool = Field(default=True)
    early_stopping_patience: int = Field(default=10, gt=0)
    early_stopping_min_delta: float = Field(default=0.0, ge=0.0)
    early_stopping_monitor: Literal["loss", "survival_loss", "cindex"] = "loss"
    risk_horizon: float = Field(default=1.0, gt=0.0)


class TrailsConfig(BaseModel):
    """:class:`TrailsEstimator` 使用的顶层不可变配置。

    属性：
        data: 数据集特征维度。
        model: 编码器、解码器、潜空间混合模型、生存头和损失设置。
        trainer: 优化和验证设置。
        seed: 在训练器专属操作前构建模型时使用的随机种子。
    """

    model_config = ConfigDict(frozen=True)

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    seed: int = 2026


class ClusterNumberSelectorConfig(BaseModel):
    """配置候选簇数的单 seed 或多 seed 选择过程。

    属性：
        candidates: 要比较的候选簇数，必须非空、唯一且均大于 1。
        seeds: 每个候选簇数重复训练所用的模型随机种子；单个整数会被规范为
            只包含一个元素的元组。
        split_seed: 未显式提供验证集时，固定训练/验证划分使用的随机种子。
        valid_fraction: 从训练数据内部留作验证集的比例。
        selection_rule: 按最高平均分或 one-standard-error 规则选择 K。
        require_non_empty: 是否排除任一重复运行产生空簇的候选 K。
        min_cluster_fraction: 可选的最小簇占比门槛。
        min_mean_pairwise_ari: 可选的 seed 间平均成对 ARI 门槛。
        estimator: 所有 K 和 seed 共享并按运行覆盖簇数与种子的估计器基础配置。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidates: tuple[int, ...]
    seeds: tuple[int, ...] = (2026,)
    split_seed: int = 2026
    valid_fraction: float = Field(default=0.2, gt=0.0, lt=1.0)
    selection_rule: Literal["best_mean", "one_standard_error"] = "best_mean"
    require_non_empty: bool = False
    min_cluster_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    min_mean_pairwise_ari: float | None = Field(default=None, ge=-1.0, le=1.0)
    estimator: TrailsConfig = Field(default_factory=TrailsConfig)

    @field_validator("seeds", mode="before")
    @classmethod
    def normalize_seeds(cls, value: object) -> object:
        """允许用单个整数表示不重复运行的选择过程。"""
        return (value,) if isinstance(value, int) and not isinstance(value, bool) else value

    @model_validator(mode="after")
    def validate_selection_settings(self) -> ClusterNumberSelectorConfig:
        """校验候选 K、随机种子以及多 seed 稳定性门槛之间的约束。"""
        if not self.candidates:
            raise ValueError("candidates must contain at least one K value.")
        if any(value <= 1 for value in self.candidates):
            raise ValueError("candidates values must be greater than 1.")
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError("candidates values must be unique.")
        if not self.seeds:
            raise ValueError("seeds must contain at least one value.")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds values must be unique.")
        if len(self.seeds) == 1 and self.min_mean_pairwise_ari is not None:
            raise ValueError("min_mean_pairwise_ari requires at least two seeds.")
        return self

"""TRAILS 的异步序列编码、重建、VaDE 聚类与 Weibull 生存模型。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import DataConfig, DecoderConfig, EncoderConfig, EncoderInputConfig, ModelConfig
from .data import Batch
from .metrics import (
    masked_mse,
    vade_kl_loss,
    weibull_mixture_negative_log_likelihood,
)


@dataclass(frozen=True)
class TrailsModelOutput:
    """一次模型前向传播的完整输出。

    属性：
        reconstruction: 与输入纵向值布局一致的重建张量。
        latent_mean: 患者变分后验均值。
        latent_log_variance: 患者变分后验对数方差。
        latent: 训练时重参数采样、评价时取均值的潜变量。
        cluster_logits: VaDE 高斯混合后验的未归一化对数分数。
        cluster_probabilities: 归一化后的后验簇概率。
        weibull_shape: 每位患者、每个簇的 Weibull 形状参数。
        weibull_scale: 每位患者、每个簇的 Weibull 尺度参数。
    """

    reconstruction: Tensor
    latent_mean: Tensor
    latent_log_variance: Tensor
    latent: Tensor
    cluster_logits: Tensor
    cluster_probabilities: Tensor
    weibull_shape: Tensor
    weibull_scale: Tensor


@dataclass(frozen=True)
class TrailsLossBreakdown:
    """总损失、原始分量、有效权重和可选不确定性参数的明细。"""

    loss: Tensor
    reconstruction_loss: Tensor
    survival_loss: Tensor
    vade_kl_loss: Tensor
    reconstruction_loss_weight: Tensor
    survival_loss_weight: Tensor
    vade_kl_loss_weight: Tensor
    reconstruction_log_variance: Tensor | None = None
    survival_log_variance: Tensor | None = None
    vade_kl_log_variance: Tensor | None = None

    def items(self) -> tuple[tuple[str, Tensor], ...]:
        """返回所有非空损失字段的名称与张量对。"""
        values: list[tuple[str, Tensor | None]] = [
            ("loss", self.loss),
            ("reconstruction_loss", self.reconstruction_loss),
            ("survival_loss", self.survival_loss),
            ("vade_kl_loss", self.vade_kl_loss),
            ("reconstruction_loss_weight", self.reconstruction_loss_weight),
            ("survival_loss_weight", self.survival_loss_weight),
            ("vade_kl_loss_weight", self.vade_kl_loss_weight),
            ("reconstruction_log_variance", self.reconstruction_log_variance),
            ("survival_log_variance", self.survival_log_variance),
            ("vade_kl_log_variance", self.vade_kl_log_variance),
        ]
        return tuple((name, value) for name, value in values if value is not None)


class GRUDCell(nn.Module):
    """对输入和隐状态应用缺失时间衰减的 GRU-D 单步单元。"""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        """创建特征级输入衰减、隐状态衰减和 GRUCell。"""
        super().__init__()
        self.input_decay = nn.Linear(input_size, input_size)
        self.hidden_decay = nn.Linear(input_size, hidden_size)
        self.gru_cell = nn.GRUCell(input_size * 2, hidden_size)

    def forward(
        self,
        x_t: Tensor,
        mask_t: Tensor,
        delta_t: Tensor,
        hidden: Tensor,
        last_observed: Tensor,
        feature_means: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """执行一个时间步的衰减插补和循环状态更新。

        缺失输入在最近观测与总体特征均值之间按 ``delta_t`` 衰减插补，隐状态
        单独衰减后送入 GRU。返回下一隐状态和更新后的最近观测值。
        """
        gamma_x = torch.exp(-torch.relu(self.input_decay(delta_t)))
        gamma_h = torch.exp(-torch.relu(self.hidden_decay(delta_t)))
        mean = feature_means.unsqueeze(0).expand_as(x_t)
        decayed_input = gamma_x * last_observed + (1.0 - gamma_x) * mean
        x_hat = mask_t * x_t + (1.0 - mask_t) * decayed_input
        decayed_hidden = gamma_h * hidden
        next_hidden = self.gru_cell(torch.cat([x_hat, mask_t], dim=-1), decayed_hidden)
        next_last_observed = mask_t * x_t + (1.0 - mask_t) * last_observed
        return next_hidden, next_last_observed


class SequencePool(nn.Module):
    """通过可学习时间注意力把变长隐序列汇总为患者表示。"""

    def __init__(self, hidden_size: int) -> None:
        """创建将每个有效时间步映射为标量分数的线性层。"""
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden_sequence: Tensor, sequence_lengths: Tensor) -> Tensor:
        """对有效时间步加权求和，返回患者级隐表示。"""
        # SeqPool：只在有效访问上归一化时间权重，再汇总为病人级表示。
        weights = self.attention_weights(hidden_sequence, sequence_lengths)
        return torch.sum(weights.unsqueeze(-1) * hidden_sequence, dim=1)

    def attention_weights(self, hidden_sequence: Tensor, sequence_lengths: Tensor) -> Tensor:
        """返回仅在各患者有效时间步上归一化的注意力权重。"""
        _batch_size, max_length, _hidden_size = hidden_sequence.shape
        steps = torch.arange(max_length, device=hidden_sequence.device).unsqueeze(0)
        active = steps < sequence_lengths.to(hidden_sequence.device).unsqueeze(1)
        logits = self.score(hidden_sequence).squeeze(-1)
        logits = logits.masked_fill(~active, torch.finfo(logits.dtype).min)
        return torch.softmax(logits, dim=-1)


def sequence_padding_mask(
    sequence_lengths: Tensor,
    max_length: int,
    device: torch.device,
) -> Tensor:
    """根据序列长度生成 ``True`` 表示补齐位置的二维掩码。"""
    lengths = sequence_lengths.to(device)
    steps = torch.arange(max_length, device=device).unsqueeze(0)
    return steps >= lengths.unsqueeze(1)


def active_sequence_mask(
    sequence_lengths: Tensor,
    max_length: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """生成形状为 ``(batch, max_length, 1)`` 的有效位置数值掩码。"""
    return (
        (~sequence_padding_mask(sequence_lengths, max_length, device)).to(dtype=dtype).unsqueeze(-1)
    )


def visit_time_features(times: Tensor) -> Tensor:
    """把访视时间扩展为原值、``log1p``、正弦和余弦四维特征。"""
    nonnegative_times = times.clamp_min(0.0)
    return torch.stack(
        [
            times,
            torch.log1p(nonnegative_times),
            torch.sin(times),
            torch.cos(times),
        ],
        dim=-1,
    )


class GRUDInputLayer(nn.Module):
    """沿 aligned 访视时间轴运行 GRU-D 的异步输入层。"""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        """创建指定特征数和隐状态宽度的 GRU-D 输入层。"""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = GRUDCell(input_size, hidden_size)

    def forward(
        self,
        *,
        x: Tensor,
        mask: Tensor,
        delta_time: Tensor,
        sequence_lengths: Tensor,
        feature_means: Tensor,
    ) -> Tensor:
        """编码补齐后的 aligned 批次并返回逐访视隐状态。

        超过患者真实序列长度的位置保持上一隐状态不变；缺失值由单步 GRU-D
        衰减规则处理。
        """
        batch_size, max_length, _n_features = x.shape
        hidden = x.new_zeros(batch_size, self.hidden_size)
        last_observed = feature_means.unsqueeze(0).expand(batch_size, self.input_size)
        hidden_states: list[Tensor] = []

        for step in range(max_length):
            next_hidden, next_last_observed = self.cell(
                x[:, step],
                mask[:, step],
                delta_time[:, step],
                hidden,
                last_observed,
                feature_means,
            )
            active = (step < sequence_lengths).to(dtype=x.dtype, device=x.device).unsqueeze(-1)
            hidden = active * next_hidden + (1.0 - active) * hidden
            last_observed = active * next_last_observed + (1.0 - active) * last_observed
            hidden_states.append(hidden)

        return torch.stack(hidden_states, dim=1)


class MTANInputLayer(nn.Module):
    """把 aligned ``(B, T, D)`` 观测映射到全局参考时间网格的原始 mTAN 输入层。

    注意力值由 ``x`` 与 ``mask`` 拼接，查询为训练数据范围内等距参考时间点。
    """

    def __init__(
        self,
        input_size: int,
        config: EncoderInputConfig,
        dropout: float,
    ) -> None:
        """根据输入配置创建时间嵌入、多时间注意力和参考时间 buffer。"""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = config.hidden_dim
        self.num_ref_points = config.num_ref_points

        time_embedding_dim = config.time_embedding_dim or config.hidden_dim
        self.time_embedding = OriginalMTANTimeEmbedding(
            embedding_dim=time_embedding_dim,
            learn_embedding=config.learn_time_embedding,
            frequency=config.time_embedding_frequency,
        )
        self.attention = MultiTimeAttention(
            input_dim=2 * input_size,
            hidden_dim=config.hidden_dim,
            time_embedding_dim=time_embedding_dim,
            n_heads=config.n_heads,
            dropout=dropout,
        )
        self.reference_times = nn.Buffer(
            torch.linspace(0.0, 1.0, config.num_ref_points, dtype=torch.float32)
        )

    def set_reference_time_range(self, min_time: float, max_time: float) -> None:
        """将参考时间 buffer 更新为给定闭区间上的等距网格。"""
        if max_time < min_time:
            raise ValueError("max_time must be greater than or equal to min_time.")
        reference_times = torch.linspace(
            min_time,
            max_time,
            self.num_ref_points,
            dtype=self.reference_times.dtype,
            device=self.reference_times.device,
        )
        self.reference_times.copy_(reference_times)

    def forward(
        self,
        *,
        times: Tensor,
        x: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """将 aligned 观测注意到参考网格。

        返回编码序列、每位患者共享的参考时间以及均等于参考点数的序列长度。
        输入特征数或 aligned 时间张量维度不匹配时抛出 :class:`ValueError`。
        """
        batch_size, _max_length, n_features = x.shape
        if n_features != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, got {n_features}.")
        if times.ndim != 2:
            raise ValueError("Original mTAN input requires aligned times with shape (B, T).")

        query_times = (
            self.reference_times.to(device=x.device, dtype=x.dtype)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        key_embedding = self.time_embedding(times)
        query_embedding = self.time_embedding(query_times)
        values = torch.cat([x, mask], dim=-1)
        value_mask = torch.cat([mask, mask], dim=-1) > 0
        encoded = self.attention(
            query=query_embedding,
            key=key_embedding,
            value=values,
            mask=value_mask,
        )
        sequence_lengths = torch.full(
            (batch_size,),
            self.num_ref_points,
            dtype=torch.long,
            device=x.device,
        )
        return encoded, query_times, sequence_lengths


class MTAN2InputLayer(nn.Module):
    """分别编码每个特征观测流并在参考时间网格拼接的 mTAN2 输入层。

    compact 输入被重排为 ``B * D`` 条特征级序列，经共享注意力和前馈网络后
    拼接为 ``(B, reference_points, D * hidden_dim)``。
    """

    def __init__(
        self,
        input_size: int,
        config: EncoderInputConfig,
        dropout: float,
    ) -> None:
        """创建时间/数值投影、多头注意力、前馈层和参考时间 buffer。"""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = config.hidden_dim
        self.num_ref_points = config.num_ref_points
        self.time_embedding_kind = config.time_embedding_kind

        time_embedding_dim = config.time_embedding_dim or config.hidden_dim
        if config.time_embedding_kind == "mtan":
            self.time_embedding = TimeEmbedding(
                embedding_dim=time_embedding_dim,
                learn_embedding=config.learn_time_embedding,
                frequency=config.time_embedding_frequency,
            )
        elif config.time_embedding_kind == "projection":
            self.time_embedding = nn.Linear(1, time_embedding_dim)
        else:
            raise ValueError(f"Unknown time embedding kind: {config.time_embedding_kind}")
        self.value_projection = nn.Linear(1, config.value_projection_dim)

        self.attention = MTAN2MultiTimeAttention(
            query_dim=time_embedding_dim,
            key_dim=time_embedding_dim,
            value_dim=config.value_projection_dim,
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            dropout=dropout,
        )
        self.attention_norm = nn.LayerNorm(config.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(config.hidden_dim)
        self.reference_times = nn.Buffer(
            torch.linspace(0.0, 1.0, config.num_ref_points, dtype=torch.float32)
        )

    def set_reference_time_range(self, min_time: float, max_time: float) -> None:
        """将参考时间 buffer 更新为给定闭区间上的等距网格。"""
        if max_time < min_time:
            raise ValueError("max_time must be greater than or equal to min_time.")
        reference_times = torch.linspace(
            min_time,
            max_time,
            self.num_ref_points,
            dtype=self.reference_times.dtype,
            device=self.reference_times.device,
        )
        self.reference_times.copy_(reference_times)

    def forward(
        self,
        *,
        times: Tensor,
        x: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """编码 compact 特征流并返回拼接后的参考时间序列。

        没有任何观测的特征流输出保持为零。返回编码序列、患者级参考时间和固定
        序列长度。
        """
        batch_size, max_length, n_features = x.shape
        if n_features != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, got {n_features}.")
        if times.ndim == 2:
            times = times.unsqueeze(-1).expand_as(x)

        flat_batch_size = batch_size * n_features
        key_times = times.permute(0, 2, 1).reshape(flat_batch_size, max_length)
        token_mask = mask.permute(0, 2, 1).reshape(flat_batch_size, max_length) > 0
        query_times = (
            self.reference_times.to(device=x.device, dtype=x.dtype)
            .unsqueeze(0)
            .expand(
                flat_batch_size,
                -1,
            )
        )
        values = self.value_projection(x.permute(0, 2, 1).reshape(flat_batch_size, max_length, 1))
        if self.time_embedding_kind == "projection":
            key_times = key_times.unsqueeze(-1)
            query_times = query_times.unsqueeze(-1)
        key_embedding = self.time_embedding(key_times)
        query_embedding = self.time_embedding(query_times)
        attended = self.attention(
            query=query_embedding,
            key=key_embedding,
            value=values,
            mask=token_mask,
        )
        feature_has_observation = token_mask.any(dim=1).to(dtype=attended.dtype).view(-1, 1, 1)
        attended = attended * feature_has_observation
        encoded = self.attention_norm(attended)
        encoded = self.ffn_norm(encoded + self.ffn(encoded))
        encoded = encoded * feature_has_observation
        encoded = (
            encoded.reshape(batch_size, n_features, self.num_ref_points, self.hidden_size)
            .transpose(1, 2)
            .contiguous()
            .reshape(batch_size, self.num_ref_points, n_features * self.hidden_size)
        )
        mapping_times = (
            self.reference_times.to(device=x.device, dtype=x.dtype)
            .unsqueeze(0)
            .expand(
                batch_size,
                -1,
            )
        )
        sequence_lengths = torch.full(
            (batch_size,),
            self.num_ref_points,
            dtype=torch.long,
            device=x.device,
        )
        return encoded, mapping_times, sequence_lengths


class TimeEmbedding(nn.Module):
    """为 mTAN2 提供可学习线性或固定正余弦时间嵌入。"""

    def __init__(self, *, embedding_dim: int, learn_embedding: bool, frequency: float) -> None:
        """配置嵌入宽度、是否学习投影以及固定嵌入频率尺度。"""
        super().__init__()
        self.embedding_dim = embedding_dim
        self.learn_embedding = learn_embedding
        self.frequency = frequency
        if learn_embedding:
            self.linear = nn.Linear(1, embedding_dim)
        else:
            self.linear = None

    def forward(self, times: Tensor) -> Tensor:
        """根据配置返回可学习线性或固定时间嵌入。"""
        if self.learn_embedding:
            return self._required_linear(times.unsqueeze(-1))
        return self.fixed_time_embedding(times)

    def fixed_time_embedding(self, times: Tensor) -> Tensor:
        """计算宽度为 ``embedding_dim`` 的交替正弦/余弦嵌入。"""
        position = 48.0 * times.unsqueeze(-1)
        embedding = times.new_zeros(*times.shape, self.embedding_dim)
        div_term = torch.exp(
            torch.arange(0, self.embedding_dim, 2, device=times.device, dtype=times.dtype)
            * -(math.log(self.frequency) / self.embedding_dim)
        )
        embedding[..., 0::2] = torch.sin(position * div_term)
        if self.embedding_dim > 1:
            embedding[..., 1::2] = torch.cos(position * div_term[: embedding[..., 1::2].shape[-1]])
        return embedding

    def _required_linear(self, times: Tensor) -> Tensor:
        """调用必须已初始化的可学习线性时间投影。"""
        if self.linear is None:
            raise RuntimeError("learned time embedding linear layer is not initialized.")
        return self.linear(times)


class OriginalMTANTimeEmbedding(nn.Module):
    """实现原始 mTAN 的线性加周期时间嵌入或固定正余弦嵌入。"""

    def __init__(self, *, embedding_dim: int, learn_embedding: bool, frequency: float) -> None:
        """按原始 mTAN 结构创建一维线性项和周期投影。"""
        super().__init__()
        self.embedding_dim = embedding_dim
        self.learn_embedding = learn_embedding
        self.frequency = frequency
        if learn_embedding:
            self.linear = nn.Linear(1, 1)
            self.periodic = nn.Linear(1, embedding_dim - 1)
        else:
            self.linear = None
            self.periodic = None

    def forward(self, times: Tensor) -> Tensor:
        """根据配置返回原始 mTAN 的可学习或固定时间嵌入。"""
        if self.learn_embedding:
            return self._learned_time_embedding(times.unsqueeze(-1))
        return self.fixed_time_embedding(times)

    def fixed_time_embedding(self, times: Tensor) -> Tensor:
        """计算宽度为 ``embedding_dim`` 的交替正弦/余弦嵌入。"""
        position = 48.0 * times.unsqueeze(-1)
        embedding = times.new_zeros(*times.shape, self.embedding_dim)
        div_term = torch.exp(
            torch.arange(0, self.embedding_dim, 2, device=times.device, dtype=times.dtype)
            * -(math.log(self.frequency) / self.embedding_dim)
        )
        embedding[..., 0::2] = torch.sin(position * div_term)
        if self.embedding_dim > 1:
            embedding[..., 1::2] = torch.cos(position * div_term[: embedding[..., 1::2].shape[-1]])
        return embedding

    def _learned_time_embedding(self, times: Tensor) -> Tensor:
        """拼接可学习线性时间项与正弦周期项。"""
        if self.linear is None or self.periodic is None:
            raise RuntimeError("learned mTAN time embedding layers are not initialized.")
        linear = self.linear(times)
        periodic = torch.sin(self.periodic(times))
        return torch.cat([linear, periodic], dim=-1)


class MultiTimeAttention(nn.Module):
    """原始 mTAN 的按值维度独立屏蔽多头时间注意力。

    查询和键来自时间嵌入；每个输入值维度拥有独立观测掩码，所有注意力头的
    结果拼接后投影到目标隐藏宽度。
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        time_embedding_dim: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        """创建时间查询/键投影和多头输出投影。"""
        super().__init__()
        if time_embedding_dim % n_heads != 0:
            raise ValueError("time_embedding_dim must be divisible by n_heads.")
        self.n_heads = n_heads
        self.head_time_dim = time_embedding_dim // n_heads
        self.input_dim = input_dim
        self.query_projection = nn.Linear(time_embedding_dim, time_embedding_dim)
        self.key_projection = nn.Linear(time_embedding_dim, time_embedding_dim)
        self.output_projection = nn.Linear(input_dim * n_heads, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        *,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """从键时间上的值计算每个查询时间的隐藏表示。

        ``mask`` 可为 ``(B, T)`` 或与 ``value`` 同形状；特征宽度和掩码形状
        不满足构造契约时抛出 :class:`ValueError`。
        """
        batch_size, _sequence_length, value_dim = value.shape
        if value_dim != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} value features, got {value_dim}.")
        if mask is not None and mask.ndim == 2:
            mask = mask.unsqueeze(-1).expand(-1, -1, value_dim)
        if mask is not None and mask.shape != value.shape:
            raise ValueError(
                f"mask must have shape {tuple(value.shape)} or (B, T), got {tuple(mask.shape)}."
            )

        query_heads = self.query_projection(query).view(
            query.shape[0],
            -1,
            self.n_heads,
            self.head_time_dim,
        )
        key_heads = self.key_projection(key).view(
            key.shape[0],
            -1,
            self.n_heads,
            self.head_time_dim,
        )
        query_heads = query_heads.transpose(1, 2)
        key_heads = key_heads.transpose(1, 2)
        attended = self._attention(query_heads, key_heads, value, mask)
        attended = (
            attended.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * value_dim)
        )
        return self.output_projection(attended)

    def _attention(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None) -> Tensor:
        """计算缩放点积注意力并按值维度汇总。"""
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(float(query.shape[-1]))
        scores = scores.unsqueeze(-1).repeat_interleave(value.shape[-1], dim=-1)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(-3) == 0, -1e9)
        attention = F.softmax(scores, dim=-2)
        attention = self.dropout(attention)
        return torch.sum(attention * value.unsqueeze(1).unsqueeze(-3), dim=-2)


class MTAN2MultiTimeAttention(nn.Module):
    """用于特征级 mTAN2 序列的标准掩码多头时间注意力。"""

    def __init__(
        self,
        *,
        query_dim: int,
        key_dim: int,
        value_dim: int,
        hidden_dim: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        """创建查询、键和输出投影，并校验隐藏宽度可分头。"""
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_heads.")
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.value_dim = value_dim
        self.query_projection = nn.Linear(query_dim, hidden_dim)
        self.key_projection = nn.Linear(key_dim, hidden_dim)
        # self.value_projection = nn.Linear(value_dim, hidden_dim)
        self.output_projection = nn.Linear(value_dim * n_heads, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        *,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """在有效键时间上计算多头注意力并投影为隐藏表示。"""
        batch_size, query_length, _embed_dim = query.shape
        key_length = int(key.shape[1])
        query_heads = self.query_projection(query).view(
            batch_size,
            query_length,
            self.n_heads,
            self.head_dim,
        )
        key_heads = self.key_projection(key).view(
            batch_size,
            key_length,
            self.n_heads,
            self.head_dim,
        )
        query_heads = query_heads.transpose(1, 2)
        key_heads = key_heads.transpose(1, 2)
        scores = torch.matmul(query_heads, key_heads.transpose(-2, -1)) / math.sqrt(
            float(self.head_dim)
        )
        attention = masked_softmax(scores, mask[:, None, None, :])
        attention = self.dropout(attention)
        value_heads = value.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        attended = torch.matmul(attention, value_heads)
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                query_length,
                self.n_heads * self.value_dim,
            )
        )
        return self.output_projection(attended)


def masked_softmax(scores: Tensor, mask: Tensor) -> Tensor:
    """在最后一维仅对有效位置执行数值稳定的 softmax。"""
    mask_float = mask.to(dtype=scores.dtype)
    masked_scores = scores.masked_fill(~mask, -1e9)
    shifted = masked_scores - masked_scores.amax(dim=-1, keepdim=True)
    weights = torch.exp(shifted) * mask_float
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)


class RecurrentMappingLayer(nn.Module):
    """使用 GRU 或 LSTM 映射输入序列并清零补齐位置。"""

    def __init__(
        self,
        *,
        kind: str,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        """创建批次优先的多层循环映射网络。"""
        super().__init__()
        recurrent_dropout = dropout if n_layers > 1 else 0.0
        rnn_cls = nn.GRU if kind == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )

    def forward(self, sequence: Tensor, times: Tensor, sequence_lengths: Tensor) -> Tensor:
        """映射输入序列；``times`` 仅为统一映射接口保留。"""
        del times
        encoded, _hidden = self.rnn(sequence)
        return encoded * active_sequence_mask(
            sequence_lengths,
            sequence.shape[1],
            dtype=sequence.dtype,
            device=sequence.device,
        )


class TransformerMappingLayer(nn.Module):
    """融合访视时间特征并应用 Transformer Encoder 的序列映射层。"""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        """创建输入/时间投影和多层 Transformer Encoder。"""
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.time_projection = nn.Linear(4, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, sequence: Tensor, times: Tensor, sequence_lengths: Tensor) -> Tensor:
        """编码有效序列位置，并将补齐位置输出清零。"""
        padding_mask = sequence_padding_mask(sequence_lengths, sequence.shape[1], sequence.device)
        tokens = self.input_projection(sequence) + self.time_projection(visit_time_features(times))
        encoded = self.transformer(tokens, src_key_padding_mask=padding_mask)
        return encoded * active_sequence_mask(
            sequence_lengths,
            sequence.shape[1],
            dtype=sequence.dtype,
            device=sequence.device,
        )


class TrailsEncoder(nn.Module):
    """组合异步输入层、时序映射层和患者级 SeqPool。

    输入层可选 GRU-D、aligned mTAN 或 compact mTAN2；映射层可选 GRU、LSTM
    或 Transformer。前向传播返回患者级表示及解码可复用的映射时间与长度。
    """

    def __init__(
        self,
        data_config: DataConfig,
        encoder_config: EncoderConfig,
        *,
        dropout: float,
    ) -> None:
        """根据数据维度和编码器配置组装输入、映射与池化组件。"""
        super().__init__()
        self.encoder_config = encoder_config
        if encoder_config.input.kind == "grud":
            self.input_layer = GRUDInputLayer(
                data_config.n_features,
                encoder_config.input.hidden_dim,
            )
        elif encoder_config.input.kind == "mtan":
            self.input_layer = MTANInputLayer(
                data_config.n_features,
                encoder_config.input,
                dropout,
            )
        else:
            self.input_layer = MTAN2InputLayer(
                data_config.n_features,
                encoder_config.input,
                dropout,
            )

        mapping_config = encoder_config.mapping
        mapping_input_dim = (
            data_config.n_features * encoder_config.input.hidden_dim
            if encoder_config.input.kind == "mtan2"
            else encoder_config.input.hidden_dim
        )
        if mapping_config.kind in {"gru", "lstm"}:
            self.mapping = RecurrentMappingLayer(
                kind=mapping_config.kind,
                input_dim=mapping_input_dim,
                hidden_dim=mapping_config.hidden_dim,
                n_layers=mapping_config.n_layers,
                dropout=dropout,
            )
        else:
            self.mapping = TransformerMappingLayer(
                input_dim=mapping_input_dim,
                hidden_dim=mapping_config.hidden_dim,
                n_layers=mapping_config.n_layers,
                n_heads=mapping_config.n_heads,
                dropout=dropout,
            )

        self.seq_pool = SequencePool(encoder_config.mapping.hidden_dim)

    def set_reference_time_range(self, min_time: float, max_time: float) -> None:
        """为 mTAN 输入设置参考时间范围；GRU-D 输入不执行操作。"""
        if isinstance(self.input_layer, (MTANInputLayer, MTAN2InputLayer)):
            self.input_layer.set_reference_time_range(min_time, max_time)

    @property
    def reference_times(self) -> Tensor | None:
        """返回 mTAN 参考时间 buffer；GRU-D 输入返回 ``None``。"""
        if isinstance(self.input_layer, (MTANInputLayer, MTAN2InputLayer)):
            return self.input_layer.reference_times
        return None

    def forward(
        self,
        *,
        times: Tensor,
        x: Tensor,
        mask: Tensor,
        delta_time: Tensor | None,
        sequence_lengths: Tensor | None,
        feature_means: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """编码 aligned 或 compact 批次并汇总为患者表示。

        GRU-D 要求 ``delta_time`` 和 ``sequence_lengths``；mTAN 变体自行产生
        参考时间序列。返回患者表示、映射时间和映射序列长度。
        """
        if self.encoder_config.input.kind == "grud":
            if delta_time is None or sequence_lengths is None:
                raise ValueError("GRUD encoder requires delta_time and sequence_lengths.")
            input_sequence = self.input_layer(
                x=x,
                mask=mask,
                delta_time=delta_time,
                sequence_lengths=sequence_lengths,
                feature_means=feature_means,
            )
            mapping_times = times
            mapping_lengths = sequence_lengths
        else:
            input_sequence, mapping_times, mapping_lengths = self.input_layer(
                times=times,
                x=x,
                mask=mask,
            )
        mapped_sequence = self.mapping(input_sequence, mapping_times, mapping_lengths)
        return self.seq_pool(mapped_sequence, mapping_lengths), mapping_times, mapping_lengths


class RecurrentDecoder(nn.Module):
    """使用 GRU/LSTM 从患者潜变量重建指定时间点的多变量观测。

    潜变量可以投影为循环网络初始状态，也可在每个时间步与时间值拼接。
    """

    def __init__(
        self,
        *,
        data_config: DataConfig,
        decoder_config: DecoderConfig,
        latent_dim: int,
        dropout: float,
    ) -> None:
        """根据循环架构和条件注入方式创建解码器。"""
        super().__init__()
        self.decoder_config = decoder_config
        recurrent_dropout = dropout if decoder_config.n_layers > 1 else 0.0
        input_size = 1 if decoder_config.conditioning == "initial_state" else latent_dim + 1
        rnn_cls = nn.GRU if decoder_config.kind == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=decoder_config.hidden_dim,
            num_layers=decoder_config.n_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        if decoder_config.conditioning == "initial_state":
            self.initial_hidden = nn.Linear(
                latent_dim,
                decoder_config.hidden_dim * decoder_config.n_layers,
            )
            self.initial_cell = (
                nn.Linear(latent_dim, decoder_config.hidden_dim * decoder_config.n_layers)
                if decoder_config.kind == "lstm"
                else None
            )
        else:
            self.initial_hidden = None
            self.initial_cell = None
        self.reconstruction_head = nn.Linear(decoder_config.hidden_dim, data_config.n_features)

    def forward(self, latent: Tensor, times: Tensor, sequence_lengths: Tensor) -> Tensor:
        """在给定时间网格上解码并返回全部特征的重建值。"""
        del sequence_lengths
        if self.decoder_config.conditioning == "initial_state":
            decoded = self._decode_with_initial_state(latent, times)
        else:
            decoded = self._decode_with_concat_time(latent, times)
        return self.reconstruction_head(decoded)

    def _decode_with_initial_state(self, latent: Tensor, times: Tensor) -> Tensor:
        """把潜变量映射为 GRU/LSTM 初始状态后按时间解码。"""
        if self.initial_hidden is None:
            raise RuntimeError("initial hidden layer is required for initial_state decoding.")
        batch_size = latent.shape[0]
        hidden = self.initial_hidden(latent).reshape(
            self.decoder_config.n_layers,
            batch_size,
            self.decoder_config.hidden_dim,
        )
        decoder_input = times.unsqueeze(-1)
        if self.decoder_config.kind == "lstm":
            if self.initial_cell is None:
                raise RuntimeError("initial cell layer is required for LSTM decoding.")
            cell = self.initial_cell(latent).reshape(
                self.decoder_config.n_layers,
                batch_size,
                self.decoder_config.hidden_dim,
            )
            decoded, _state = self.rnn(decoder_input, (hidden.contiguous(), cell.contiguous()))
        else:
            decoded, _state = self.rnn(decoder_input, hidden.contiguous())
        return decoded

    def _decode_with_concat_time(self, latent: Tensor, times: Tensor) -> Tensor:
        """在每个时间步拼接潜变量与时间后执行循环解码。"""
        repeated_latent = latent.unsqueeze(1).expand(-1, times.shape[1], -1)
        decoder_input = torch.cat([times.unsqueeze(-1), repeated_latent], dim=-1)
        decoded, _state = self.rnn(decoder_input)
        return decoded


class TransformerDecoder(nn.Module):
    """把重复潜变量与时间拼接后通过 Transformer 重建纵向观测。"""

    def __init__(
        self,
        *,
        data_config: DataConfig,
        decoder_config: DecoderConfig,
        latent_dim: int,
        dropout: float,
    ) -> None:
        """创建输入投影、多层 Transformer Encoder 和特征重建头。"""
        super().__init__()
        self.input_projection = nn.Linear(latent_dim + 1, decoder_config.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=decoder_config.hidden_dim,
            nhead=decoder_config.n_heads,
            dim_feedforward=decoder_config.hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=decoder_config.n_layers)
        self.reconstruction_head = nn.Linear(decoder_config.hidden_dim, data_config.n_features)

    def forward(self, latent: Tensor, times: Tensor, sequence_lengths: Tensor) -> Tensor:
        """在有效时间位置上解码潜变量并返回多特征重建。"""
        repeated_latent = latent.unsqueeze(1).expand(-1, times.shape[1], -1)
        decoder_input = torch.cat([times.unsqueeze(-1), repeated_latent], dim=-1)
        padding_mask = sequence_padding_mask(sequence_lengths, times.shape[1], times.device)
        decoded = self.transformer(
            self.input_projection(decoder_input),
            src_key_padding_mask=padding_mask,
        )
        return self.reconstruction_head(decoded)


class TrailsSurvVaderModel(nn.Module):
    """联合纵向重建、VaDE 聚类和簇特异 Weibull 生存风险的核心模型。

    编码器把异步纵向序列汇总为患者表示，变分层产生潜变量；高斯混合先验给出
    后验簇概率，解码器重建纵向输入，生存头为每个簇输出正的 Weibull 形状与
    尺度参数。损失可使用固定权重或可学习同方差不确定性权重。
    """

    def __init__(self, data_config: DataConfig, model_config: ModelConfig) -> None:
        """根据数据和模型配置组装编码器、潜空间、解码器、混合先验与生存头。"""
        super().__init__()
        self.data_config = data_config
        self.model_config = model_config
        self.register_buffer("_feature_means", torch.zeros(data_config.n_features))

        self.encoder = TrailsEncoder(
            data_config,
            model_config.encoder,
            dropout=model_config.dropout,
        )
        self.latent_mean = nn.Linear(
            model_config.encoder.mapping.hidden_dim,
            model_config.latent_dim,
        )
        self.latent_log_variance = nn.Linear(
            model_config.encoder.mapping.hidden_dim,
            model_config.latent_dim,
        )

        decoder_config = model_config.decoder
        if decoder_config.kind in {"gru", "lstm"}:
            self.decoder = RecurrentDecoder(
                data_config=data_config,
                decoder_config=decoder_config,
                latent_dim=model_config.latent_dim,
                dropout=model_config.dropout,
            )
        else:
            self.decoder = TransformerDecoder(
                data_config=data_config,
                decoder_config=decoder_config,
                latent_dim=model_config.latent_dim,
                dropout=model_config.dropout,
            )

        # VaDE 聚类先验：c ~ Cat(pi), z | c ~ Normal(mu_c, var_c)。
        if model_config.mixture_logits_trained:
            self.mixture_logits = nn.Parameter(torch.zeros(model_config.n_clusters))
        else:
            self.mixture_logits = nn.Buffer(torch.zeros(model_config.n_clusters))
        self.mixture_means = nn.Parameter(
            torch.randn(model_config.n_clusters, model_config.latent_dim) * 0.01
        )
        self.mixture_log_variances = nn.Parameter(
            torch.zeros(model_config.n_clusters, model_config.latent_dim)
        )
        self.survival_head = build_survival_head(model_config)
        self.loss_log_variances = nn.ParameterDict()
        if model_config.loss.weighting == "uncertainty":
            for name, initial_weight in {
                "reconstruction": model_config.loss.reconstruction_weight,
                "survival": model_config.loss.survival_weight,
                "vade_kl": model_config.loss.cluster_weight,
            }.items():
                initial_log_variance = -math.log(2.0 * initial_weight)
                self.loss_log_variances[name] = nn.Parameter(
                    torch.tensor(initial_log_variance, dtype=torch.float32)
                )

    def set_feature_means(self, feature_means: Tensor) -> None:
        """更新 GRU-D 缺失值衰减使用的各特征训练集均值。"""
        if feature_means.shape != self.feature_means.shape:
            raise ValueError(
                f"feature_means must have shape {tuple(self.feature_means.shape)}, "
                f"got {tuple(feature_means.shape)}."
            )
        self.feature_means.copy_(feature_means.to(self.feature_means.device))

    @property
    def feature_means(self) -> Tensor:
        """返回注册为 buffer 的特征均值张量。"""
        return cast(Tensor, self._buffers["_feature_means"])

    def set_reference_time_range(self, min_time: float, max_time: float) -> None:
        """把训练数据观测时间范围传给编码器中的 mTAN 输入层。"""
        self.encoder.set_reference_time_range(min_time, max_time)

    @property
    def reference_times(self) -> Tensor | None:
        """返回 mTAN 参考时间；非 mTAN 输入返回 ``None``。"""
        return self.encoder.reference_times

    def set_mixture_parameters(
        self,
        prior_probabilities: Tensor,
        means: Tensor,
        variances: Tensor,
    ) -> None:
        """用给定混合比例、均值和对角方差初始化 VaDE 先验。

        输入形状必须分别为 ``(K,)`` 和 ``(K, latent_dim)``；概率与方差在取
        对数前限制最小值为 ``1e-6``，参数复制过程不记录梯度。
        """
        expected_prior_shape = (self.model_config.n_clusters,)
        expected_component_shape = (self.model_config.n_clusters, self.model_config.latent_dim)
        if prior_probabilities.shape != expected_prior_shape:
            raise ValueError(
                "prior_probabilities must have shape "
                f"{expected_prior_shape}, got {tuple(prior_probabilities.shape)}."
            )
        if means.shape != expected_component_shape:
            raise ValueError(
                f"means must have shape {expected_component_shape}, got {tuple(means.shape)}."
            )
        if variances.shape != expected_component_shape:
            raise ValueError(
                "variances must have shape "
                f"{expected_component_shape}, got {tuple(variances.shape)}."
            )
        with torch.no_grad():
            self.mixture_logits.copy_(
                torch.log(prior_probabilities.to(self.mixture_logits.device).clamp_min(1e-6))
            )
            self.mixture_means.copy_(means.to(self.mixture_means.device))
            self.mixture_log_variances.copy_(
                torch.log(variances.to(self.mixture_log_variances.device).clamp_min(1e-6))
            )

    def forward(
        self,
        *,
        times: Tensor,
        x: Tensor,
        mask: Tensor,
        delta_time: Tensor | None = None,
        sequence_lengths: Tensor | None = None,
        feature_lengths: Tensor | None = None,
    ) -> TrailsModelOutput:
        """执行编码、潜变量采样、重建、聚类和生存参数预测。

        aligned 输入提供 ``delta_time`` 与 ``sequence_lengths``；compact mTAN2
        输入通过 ``feature_lengths`` 标识视图。训练模式使用重参数采样，评价
        模式直接使用潜空间均值。

        返回：
            包含重建、潜变量、簇后验和 Weibull 参数的
            :class:`TrailsModelOutput`。
        """
        hidden, encoder_times, encoder_sequence_lengths = self.encoder(
            times=times,
            x=x,
            mask=mask,
            delta_time=delta_time,
            sequence_lengths=sequence_lengths,
            feature_means=self.feature_means,
        )
        latent_mean = self.latent_mean(hidden)
        latent_log_variance = self.latent_log_variance(hidden)
        latent = self._sample_latent(latent_mean, latent_log_variance)
        reconstruction = self._decode_reconstruction(
            latent,
            batch_times=times,
            batch_mask=mask,
            aligned_sequence_lengths=sequence_lengths,
            encoder_times=encoder_times,
            encoder_sequence_lengths=encoder_sequence_lengths,
            feature_lengths=feature_lengths,
        )
        cluster_logits = self._cluster_logits(latent)
        cluster_probabilities = torch.softmax(cluster_logits, dim=-1)
        survival_raw = self.survival_head(latent).reshape(-1, self.model_config.n_clusters, 2)
        weibull_params = F.softplus(survival_raw) + 1e-3
        return TrailsModelOutput(
            reconstruction=reconstruction,
            latent_mean=latent_mean,
            latent_log_variance=latent_log_variance,
            latent=latent,
            cluster_logits=cluster_logits,
            cluster_probabilities=cluster_probabilities,
            weibull_shape=weibull_params[..., 0],
            weibull_scale=weibull_params[..., 1],
        )

    def _decode_reconstruction(
        self,
        latent: Tensor,
        *,
        batch_times: Tensor,
        batch_mask: Tensor,
        aligned_sequence_lengths: Tensor | None,
        encoder_times: Tensor,
        encoder_sequence_lengths: Tensor,
        feature_lengths: Tensor | None,
    ) -> Tensor:
        """按 aligned 或 compact 输入布局解码并还原重建张量。"""
        if feature_lengths is None:
            if aligned_sequence_lengths is None:
                raise ValueError("Aligned reconstruction requires sequence_lengths.")
            return self.decoder(latent, batch_times, aligned_sequence_lengths)

        del encoder_times, encoder_sequence_lengths, feature_lengths
        batch_size, max_length, n_features = batch_times.shape
        flat_times = batch_times.reshape(batch_size, max_length * n_features)
        flat_lengths = torch.full(
            (batch_size,),
            max_length * n_features,
            dtype=torch.long,
            device=batch_times.device,
        )
        flat_reconstruction = self.decoder(latent, flat_times, flat_lengths)
        feature_index = torch.arange(n_features, device=batch_times.device).repeat(max_length)
        gathered = flat_reconstruction.gather(
            dim=-1,
            index=feature_index.view(1, max_length * n_features, 1).expand(batch_size, -1, 1),
        )
        return gathered.reshape(batch_size, max_length, n_features) * (batch_mask > 0).to(
            dtype=gathered.dtype
        )

    def compute_loss(
        self,
        output: TrailsModelOutput,
        batch: Batch,
        *,
        include_vade_kl: bool,
    ) -> TrailsLossBreakdown:
        """计算重建、生存、VaDE KL 及其固定或不确定性加权总损失。

        参数：
            output: 当前批次模型输出。
            batch: 含真实纵向值、掩码和生存结局的批次。
            include_vade_kl: 是否在当前阶段启用 VaDE KL 分量。

        返回：
            原始损失、有效权重和总损失组成的 :class:`TrailsLossBreakdown`。
        """
        reconstruction = masked_mse(output.reconstruction, batch["x"], batch["mask"])
        survival = weibull_mixture_negative_log_likelihood(
            output.cluster_logits,
            output.weibull_shape,
            output.weibull_scale,
            batch["survival_time"],
            batch["event"],
        )
        if include_vade_kl:
            vade_kl = vade_kl_loss(
                output.latent,
                output.latent_mean,
                output.latent_log_variance,
                output.cluster_logits,
                self.mixture_logits,
                self.mixture_means,
                self.mixture_log_variances,
            )
        else:
            vade_kl = reconstruction.new_zeros(())

        if self.model_config.loss.weighting == "fixed":
            return self._compute_fixed_loss_breakdown(
                reconstruction=reconstruction,
                survival=survival,
                vade_kl=vade_kl,
            )

        return self._compute_uncertainty_loss_breakdown(
            reconstruction=reconstruction,
            survival=survival,
            vade_kl=vade_kl,
            include_vade_kl=include_vade_kl,
        )

    def _sample_latent(self, mean: Tensor, log_variance: Tensor) -> Tensor:
        """训练时执行重参数采样，评价时直接返回后验均值。"""
        if not self.training:
            return mean
        noise = torch.randn_like(mean)
        return mean + noise * torch.exp(0.5 * log_variance)

    def _compute_fixed_loss_breakdown(
        self,
        *,
        reconstruction: Tensor,
        survival: Tensor,
        vade_kl: Tensor,
    ) -> TrailsLossBreakdown:
        """使用配置中的固定权重组合三个损失分量。"""
        config = self.model_config.loss
        reconstruction_weight = reconstruction.new_tensor(config.reconstruction_weight)
        survival_weight = reconstruction.new_tensor(config.survival_weight)
        vade_kl_weight = reconstruction.new_tensor(config.cluster_weight)
        total = (
            reconstruction_weight * reconstruction
            + survival_weight * survival
            + vade_kl_weight * vade_kl
        )
        return TrailsLossBreakdown(
            loss=total,
            reconstruction_loss=reconstruction,
            survival_loss=survival,
            vade_kl_loss=vade_kl,
            reconstruction_loss_weight=reconstruction_weight,
            survival_loss_weight=survival_weight,
            vade_kl_loss_weight=vade_kl_weight,
        )

    def _compute_uncertainty_loss_breakdown(
        self,
        *,
        reconstruction: Tensor,
        survival: Tensor,
        vade_kl: Tensor,
        include_vade_kl: bool,
    ) -> TrailsLossBreakdown:
        """使用可学习对数方差组合启用的多任务损失。"""
        # 多任务不确定性加权：s=log(sigma^2)，用可学习噪声自动调节各 loss 贡献。
        reconstruction_term = self._uncertainty_weighted_loss("reconstruction", reconstruction)
        survival_term = self._uncertainty_weighted_loss("survival", survival)
        total = reconstruction_term + survival_term
        if include_vade_kl:
            total = total + self._uncertainty_weighted_loss("vade_kl", vade_kl)

        reconstruction_weight = 0.5 * torch.exp(-self.loss_log_variances["reconstruction"])
        survival_weight = 0.5 * torch.exp(-self.loss_log_variances["survival"])
        vade_kl_weight = 0.5 * torch.exp(-self.loss_log_variances["vade_kl"])
        return TrailsLossBreakdown(
            loss=total,
            reconstruction_loss=reconstruction,
            survival_loss=survival,
            vade_kl_loss=vade_kl,
            reconstruction_loss_weight=reconstruction_weight,
            survival_loss_weight=survival_weight,
            vade_kl_loss_weight=vade_kl_weight,
            reconstruction_log_variance=self.loss_log_variances["reconstruction"],
            survival_log_variance=self.loss_log_variances["survival"],
            vade_kl_log_variance=(self.loss_log_variances["vade_kl"] if include_vade_kl else None),
        )

    def _uncertainty_weighted_loss(self, name: str, loss: Tensor) -> Tensor:
        """计算单项同方差不确定性加权损失。"""
        log_variance = self.loss_log_variances[name]
        return 0.5 * torch.exp(-log_variance) * loss + 0.5 * log_variance

    def _cluster_logits(self, latent: Tensor) -> Tensor:
        """计算包含混合比例先验的各簇后验 logits。"""
        log_prior = torch.log_softmax(self.mixture_logits, dim=-1)
        return log_prior.unsqueeze(0) + self._component_log_prob(latent)

    def _component_log_prob(self, latent: Tensor) -> Tensor:
        """计算潜变量在各对角高斯混合分量下的对数密度。"""
        centered = latent.unsqueeze(1) - self.mixture_means.unsqueeze(0)
        log_variance = self.mixture_log_variances.unsqueeze(0)
        variance = torch.exp(log_variance)
        log_density = -0.5 * (
            torch.log(torch.tensor(2.0 * torch.pi, device=latent.device, dtype=latent.dtype))
            + log_variance
            + centered.pow(2) / variance
        )
        return log_density.sum(dim=-1)


def build_survival_head(model_config: ModelConfig) -> nn.Sequential:
    """构建输出每簇 Weibull 形状与尺度原始值的生存头。

    可选隐藏层均保持 ``latent_dim`` 宽度并使用 ReLU，最终输出宽度为
    ``n_clusters * 2``。
    """
    layers: list[nn.Module] = []
    for _layer in range(model_config.survival_head_hidden_layers):
        layers.append(nn.Linear(model_config.latent_dim, model_config.latent_dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(model_config.latent_dim, model_config.n_clusters * 2))
    return nn.Sequential(*layers)

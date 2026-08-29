"""面向用户的 TRAILS 单模型估计、推理和持久化接口。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from .config import TrailsConfig
from .data import ClinicalTimeSeriesDataset, infer_data_config
from .model import TrailsSurvVaderModel
from .prediction import TrailsPrediction
from .trainer import HistoryEntry, TrailsTrainer

HistoryCallback = Callable[[HistoryEntry], None]


class TrailsEstimator:
    """封装 TRAILS 模型、训练器和训练历史的高级估计器。

    估计器负责校验数据特征维度、设置特征均值和 mTAN 参考时间范围，并提供
    拟合、预测、评价及检查点保存/加载的一致接口。

    属性：
        config: 完整且不可变的 TRAILS 配置。
        model: Surv-VaDER 神经网络模型。
        trainer: 与模型和训练配置绑定的训练器。
        history: 最近一次拟合产生的逐轮历史。
    """

    def __init__(self, config: TrailsConfig | None = None) -> None:
        """根据配置初始化模型和训练器，并设置模型构造随机种子。

        参数：
            config: 可选的完整配置；``None`` 时使用 :class:`TrailsConfig` 默认值。
        """
        self.config = config or TrailsConfig()
        torch.manual_seed(self.config.seed)
        self.model = TrailsSurvVaderModel(self.config.data, self.config.model)
        trainer_config = self.config.trainer.model_copy(update={"seed": self.config.seed})
        self.trainer = TrailsTrainer(self.model, trainer_config)
        self.history: list[HistoryEntry] = []

    def fit(
        self,
        data: ClinicalTimeSeriesDataset,
        history_callback: HistoryCallback | None = None,
        validation_data: ClinicalTimeSeriesDataset | None = None,
    ) -> TrailsEstimator:
        """拟合 TRAILS 模型并返回当前估计器。

        训练前根据数据设置特征均值；mTAN 输入还会根据训练数据的真实观测范围
        设置全局参考时间网格。显式验证集会直接传给训练器。

        参数：
            data: 用于拟合的临床时间序列数据集。
            history_callback: 每完成一个训练轮次后调用的可选回调。
            validation_data: 可选的独立验证数据集。

        返回：
            已拟合的当前 :class:`TrailsEstimator`。

        异常：
            ValueError: 当数据特征维度与配置不一致或 mTAN 数据没有观测时抛出。
        """
        self._validate_data_config(data)
        if validation_data is not None:
            self._validate_data_config(validation_data)
        self.model.set_feature_means(data.feature_means)
        if self.config.model.encoder.input.kind in {"mtan", "mtan2"}:
            min_time, max_time = observed_time_range(data)
            self.model.set_reference_time_range(min_time, max_time)
        self.history = self.trainer.fit(
            data,
            history_callback=history_callback,
            validation_data=validation_data,
        )
        return self

    def predict(self, data: ClinicalTimeSeriesDataset) -> TrailsPrediction:
        """一次推理返回可派生全部患者级预测的结构化对象。

        参数：
            data: 特征维度与估计器配置一致的数据集。

        返回：
            CPU 上的 :class:`TrailsPrediction`，包含潜表示、簇后验和 Weibull 参数。
        """
        self._validate_data_config(data)
        outputs, batch = self.trainer._collect_outputs(data)
        return TrailsPrediction(
            latent_representation=outputs.latent_mean.detach().cpu(),
            cluster_probabilities=outputs.cluster_probabilities.detach().cpu(),
            weibull_shape=outputs.weibull_shape.detach().cpu(),
            weibull_scale=outputs.weibull_scale.detach().cpu(),
            true_cluster=(
                batch["cluster_label"].detach().cpu().long() if "cluster_label" in batch else None
            ),
        )

    def test(self, data: ClinicalTimeSeriesDataset) -> dict[str, float]:
        """计算数据集损失、生存指标、可选聚类指标和簇占用诊断。

        参数：
            data: 要评价的临床时间序列数据集。

        返回：
            指标名称到浮点值的字典；参考簇标签存在时包含 ACC、ARI 和 NMI。
        """
        self._validate_data_config(data)
        return self.trainer.test(data)

    def save(self, path: str | Path) -> None:
        """保存配置、训练历史和模型状态，并自动创建目标父目录。

        参数：
            path: PyTorch 检查点目标路径。
        """
        checkpoint = {
            "config": self.config.model_dump(mode="json"),
            "history": self.history,
            "model_state": self.model.state_dict(),
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, destination)

    @classmethod
    def load(cls, path: str | Path, *, device: str | None = None) -> TrailsEstimator:
        """从检查点重建估计器，并可覆盖训练设备。

        检查点先映射到 CPU；指定 ``device`` 时仅更新训练器设备配置，随后加载
        模型参数和历史记录。

        参数：
            path: :meth:`save` 生成的检查点路径。
            device: 可选的训练与推理设备覆盖值。

        返回：
            恢复配置、参数和历史的 :class:`TrailsEstimator`。
        """
        checkpoint: dict[str, Any] = torch.load(Path(path), map_location="cpu", weights_only=False)
        config = TrailsConfig.model_validate(checkpoint["config"])
        if device is not None:
            config = config.model_copy(
                update={"trainer": config.trainer.model_copy(update={"device": device})}
            )
        estimator = cls(config)
        estimator.model.load_state_dict(checkpoint["model_state"])
        estimator.history = list(checkpoint.get("history", []))
        return estimator

    def _validate_data_config(self, data: ClinicalTimeSeriesDataset) -> None:
        """校验数据集特征维度是否与估计器配置一致。"""
        inferred = infer_data_config(data)
        if inferred != self.config.data:
            raise ValueError(
                "Data shape does not match estimator config: "
                f"expected {self.config.data}, got {inferred}."
            )


def observed_time_range(data: ClinicalTimeSeriesDataset) -> tuple[float, float]:
    """返回数据集中所有真实观测时间的全局最小值和最大值。

    数据会转换为 compact 视图，从 ``mask > 0`` 的位置收集时间；没有任何真实
    观测时抛出 :class:`ValueError`。
    """
    compact_data = data.with_return_kind("compact")
    min_time: float | None = None
    max_time: float | None = None
    for sample in compact_data.samples:
        observed_times = sample.times[sample.mask > 0].float()
        if observed_times.numel() == 0:
            continue
        sample_min = float(observed_times.min().item())
        sample_max = float(observed_times.max().item())
        min_time = sample_min if min_time is None else min(min_time, sample_min)
        max_time = sample_max if max_time is None else max(max_time, sample_max)

    if min_time is None or max_time is None:
        raise ValueError("mTAN reference time grid requires at least one observed time.")
    return min_time, max_time

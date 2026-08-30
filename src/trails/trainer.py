"""TRAILS 的分阶段训练、早停、推理和混合模型初始化逻辑。"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from itertools import chain
from typing import Literal, NotRequired, TypedDict

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.clustering import AdjustedRandScore, NormalizedMutualInfoScore

from .config import TrainerConfig
from .data import Batch, ClinicalTimeSeriesDataset, make_data_loader
from .metrics import (
    Cindex,
    ClusteringAccuracy,
    cluster_assignment_diagnostics,
    weibull_event_probability,
)
from .model import TrailsLossBreakdown, TrailsModelOutput, TrailsSurvVaderModel
from .progress import ProgressBar

LOGGER = logging.getLogger(__name__)


class HistoryEntry(TypedDict):
    """单个训练轮次的结构化历史记录。

    属性：
        epoch: 当前训练阶段内的轮次编号。
        global_epoch: 跨 warmup 和 VaDE 阶段的全局轮次编号。
        stage: ``"warmup"`` 或 ``"vade"`` 训练阶段。
        train: 训练集损失和指标。
        valid: 可选的验证集损失和指标。
        best_global_epoch: 早停监控下的最佳全局轮次。
        best_monitor: 早停监控指标的完整名称。
        best_monitor_value: 当前最佳监控值。
        early_stopped: 是否因早停结束训练的可选标记。
    """

    epoch: int
    global_epoch: int
    stage: str
    train: dict[str, float]
    valid: NotRequired[dict[str, float]]
    best_global_epoch: NotRequired[int]
    best_monitor: NotRequired[str]
    best_monitor_value: NotRequired[float]
    early_stopped: NotRequired[bool]


HistoryCallback = Callable[[HistoryEntry], None]


class LossAccumulator:
    """按患者数加权累积并平均一个轮次中的损失分量。"""

    def __init__(self) -> None:
        """初始化空损失累计状态。"""
        self.reset()

    def reset(self) -> None:
        """清空累计损失和患者计数。"""
        self.total: dict[str, float] = {}
        self.count: int = 0

    def update(self, batch_size: int, batch_loss: TrailsLossBreakdown) -> None:
        """按批次患者数累加各项标量损失。"""
        self.count += batch_size
        for k, v in batch_loss.items():
            self.total[k] = self.total.get(k, 0.0) + v.item() * batch_size

    def compute(self) -> dict[str, float]:
        """返回各损失分量的患者加权平均值。"""
        return {name: value / max(1, self.count) for name, value in self.total.items()}


class EarlyStopper:
    """监控总损失、生存损失或 C-index，并保存最佳模型状态。

    两类损失以减少超过 ``min_delta`` 为改善，C-index 以增加超过该阈值为改善；
    连续 ``patience`` 轮未改善时请求停止。
    """

    def __init__(
        self,
        patience: int,
        monitor: Literal["loss", "survival_loss", "cindex"],
        min_delta: float,
        has_validation: bool = True,
    ) -> None:
        """配置耐心轮数、监控方向和训练/验证数据源。"""
        self.patience = patience
        self.monitor = monitor
        self.min_delta = min_delta
        self.has_validation = has_validation
        self.reset()

    def reset(self):
        """清空最佳状态和连续未改善轮数。"""
        self.best_state: dict[str, Tensor] | None = None
        self.best_value: float | None = None
        self.best_global_epoch: int | None = None
        self.stale_epochs = 0

    def update(self, entry: HistoryEntry, model: TrailsSurvVaderModel) -> bool:
        """根据当前历史更新最佳模型，并返回是否达到停止条件。"""
        monitor_value = self._monitor_value(entry)
        if self._is_monitor_improved(monitor_value, self.best_value):
            self.best_state = copy.deepcopy(model.state_dict())
            self.best_value = monitor_value
            self.best_global_epoch = entry["global_epoch"]
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1

        if self.best_value is not None and self.best_global_epoch is not None:
            entry["best_global_epoch"] = self.best_global_epoch
            entry["best_monitor"] = self._monitor_name(self.has_validation)
            entry["best_monitor_value"] = self.best_value

        return self.stale_epochs >= self.patience

    def _monitor_name(self, has_validation: bool) -> str:
        """生成包含数据划分的监控指标名称。"""
        split_name = "valid" if has_validation else "train"
        return f"{split_name}/{self.monitor}"

    def _monitor_value(self, entry: HistoryEntry) -> float:
        """从验证记录或训练记录中读取监控值。"""
        metrics = entry["valid"] if "valid" in entry else entry["train"]
        split_name = "valid" if "valid" in entry else "train"
        name = self.monitor
        if name not in metrics:
            raise ValueError(f"Early stopping monitor '{split_name}/{name}' is unavailable.")
        return float(metrics[name])

    def _is_monitor_improved(self, value: float, best_value: float | None) -> bool:
        """根据监控指标方向和最小变化量判断是否改善。"""
        if best_value is None:
            return True
        if self.monitor == "cindex":
            return value > best_value + self.min_delta
        return value < best_value - self.min_delta


class TrailsTrainer:
    """执行 Surv-VaDER 分阶段优化与批量推理。

    训练器根据输入层自动选择 aligned 或 compact 数据视图，可先执行仅重建的
    warmup，再用训练集潜空间均值初始化高斯混合先验并进行完整 VaDE 训练。
    它还负责内部验证划分、早停、梯度裁剪和跨批次指标累计。

    属性：
        model: 已移动到配置设备的 TRAILS 模型。
        config: 优化、验证和早停配置。
        optimizer: Adam 优化器。
        losses: 当前轮次的损失累计器。
    """

    def __init__(self, model: TrailsSurvVaderModel, config: TrainerConfig) -> None:
        """绑定模型与训练配置，并创建 Adam 优化器。"""
        self.model = model.to(config.device)
        self.config = config
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

        self.losses = LossAccumulator()

    def fit(
        self,
        data: ClinicalTimeSeriesDataset,
        history_callback: HistoryCallback | None = None,
        validation_data: ClinicalTimeSeriesDataset | None = None,
    ) -> list[HistoryEntry]:
        """拟合模型并返回逐轮训练历史。

        显式验证集优先于 ``trainer.valid_size``；否则从训练数据内部留出验证集。
        warmup 阶段不包含 VaDE KL，结束后初始化混合先验；正式阶段包含全部损失
        并在达到 ``min_epochs`` 后应用可选早停。训练结束时恢复最佳模型状态。

        参数：
            data: 训练数据集。
            history_callback: 每轮记录生成后调用的可选回调。
            validation_data: 可选的显式验证数据集。

        返回：
            按执行顺序排列的 warmup 与 VaDE 历史记录。
        """
        # 根据我们使用的input调整数据格式
        data = data.with_return_kind(self._model_return_kind())
        if validation_data is not None:
            if self.config.valid_size > 0:
                LOGGER.warning(
                    "Explicit validation_data was provided; trainer.valid_size=%s is ignored "
                    "for this fit call.",
                    self.config.valid_size,
                )
            validation_data = validation_data.with_return_kind(self._model_return_kind())
            valid_loader = make_data_loader(validation_data, self.config, shuffle=False)
        elif self.config.valid_size > 0:
            data, validation_data = data.split([1 - self.config.valid_size, self.config.valid_size])
            valid_loader = make_data_loader(validation_data, self.config, shuffle=False)
        else:
            validation_data = None
            valid_loader = None

        loader = make_data_loader(data, self.config, shuffle=True)

        history: list[HistoryEntry] = []

        if self.config.warmup_epochs > 0:
            for epoch in ProgressBar(range(self.config.warmup_epochs), desc="Warmup", leave=False):
                losses, scores = self._epoch_loop(loader, phase="train", include_vade_kl=False)
                entry: HistoryEntry = {  # type: ignore
                    "epoch": epoch + 1,
                    "global_epoch": len(history) + 1,
                    "stage": "warmup",
                    "train": {
                        **losses,
                        **scores,
                    },
                }

                if valid_loader is not None:
                    losses, scores = self._epoch_loop(
                        valid_loader, phase="valid", include_vade_kl=False
                    )
                    entry["valid"] = {**losses, **scores}

                history.append(entry)
                if history_callback is not None:
                    history_callback(entry)

            self.initialize_mixture_from_data(data)

        survival_metrics: dict[str, Metric] = {"cindex": Cindex()}
        cluster_metrics: dict[str, Metric] = {
            "acc": ClusteringAccuracy(),
            "nmi": NormalizedMutualInfoScore(),
            "ari": AdjustedRandScore(),
        }
        for v in chain(survival_metrics.values(), cluster_metrics.values()):
            v.to(self.config.device)

        if self.config.early_stop:
            early_stopper = EarlyStopper(
                self.config.early_stopping_patience,
                self.config.early_stopping_monitor,
                self.config.early_stopping_min_delta,
                has_validation=validation_data is not None,
            )
        else:
            early_stopper = None

        for epoch in ProgressBar(range(self.config.max_epochs), desc="Epoch", leave=False):
            losses, scores = self._epoch_loop(
                loader,
                phase="train",
                include_vade_kl=True,
                survival_metrics=survival_metrics,
                cluster_metrics=cluster_metrics if data.has_cluster_labels else {},
            )
            entry: HistoryEntry = {
                "epoch": epoch + 1,
                "global_epoch": len(history) + 1,
                "stage": "vade",
                "train": {
                    **losses,
                    **scores,
                },
            }

            if valid_loader is not None:
                losses, scores = self._epoch_loop(
                    valid_loader,
                    phase="valid",
                    include_vade_kl=True,
                    survival_metrics=survival_metrics,
                    cluster_metrics=cluster_metrics
                    if validation_data is not None and validation_data.has_cluster_labels
                    else {},
                )
                entry["valid"] = {**losses, **scores}

            history.append(entry)
            should_stop = (
                (epoch + 1) >= self.config.min_epochs
                and early_stopper is not None
                and early_stopper.update(entry, self.model)
            )
            if should_stop:
                entry["early_stopped"] = True
            if history_callback is not None:
                history_callback(entry)
            if should_stop:
                break

        if early_stopper is not None and early_stopper.best_state is not None:
            self.model.load_state_dict(early_stopper.best_state)
        return history

    def test(self, data: ClinicalTimeSeriesDataset) -> dict[str, float]:
        """在完整数据集上计算损失、预测指标和簇占用诊断。

        有参考簇标签时额外报告 ACC、NMI 和 ARI；始终报告 C-index、损失分量、
        空簇数、簇比例范围与归一化熵。
        """
        data = data.with_return_kind(self._model_return_kind())
        loader = make_data_loader(data, self.config, shuffle=False)
        survival_metrics: dict[str, Metric] = {"cindex": Cindex()}
        cluster_metrics: dict[str, Metric] = (
            {
                "acc": ClusteringAccuracy(),
                "nmi": NormalizedMutualInfoScore(),
                "ari": AdjustedRandScore(),
            }
            if data.has_cluster_labels
            else {}
        )
        losses, scores = self._epoch_loop(
            loader,
            phase="valid",
            include_vade_kl=True,
            survival_metrics=survival_metrics,
            cluster_metrics=cluster_metrics,
        )
        outputs, _batch = self._collect_outputs(data)
        cluster_scores = cluster_assignment_diagnostics(
            torch.argmax(outputs.cluster_probabilities, dim=-1),
            n_clusters=self.model.model_config.n_clusters,
        )

        return {**losses, **scores, **cluster_scores}

    def initialize_mixture_from_data(self, data: ClinicalTimeSeriesDataset) -> None:
        """用训练集潜空间均值的确定性 K-means 结果初始化高斯混合先验。"""
        latent_means = self._collect_latent_means(data)
        # warmup 后用训练集 latent_mean 初始化 MoG，避免 VaDE 早期责任度塌缩。
        prior_probabilities, means, variances = fit_kmeans_mixture(
            latent_means,
            n_clusters=self.model.model_config.n_clusters,
            n_iters=self.config.gmm_init_iters,
            seed=self.config.seed,
        )
        self.model.set_mixture_parameters(prior_probabilities, means, variances)

    def _epoch_loop(
        self,
        loader: torch.utils.data.DataLoader[Batch],
        *,
        phase: Literal["train", "valid"] = "train",
        include_vade_kl: bool = True,
        survival_metrics: dict[str, Metric] | None = None,
        cluster_metrics: dict[str, Metric] | None = None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """执行一次训练或验证循环并返回平均损失与指标。"""
        self.losses.reset()
        if survival_metrics is not None:
            for m in survival_metrics.values():
                m.reset()
        if cluster_metrics is not None:
            for m in cluster_metrics.values():
                m.reset()

        if phase == "train":
            self.model.train()
        else:
            self.model.eval()

        with torch.set_grad_enabled(phase == "train"):
            for batch in ProgressBar(loader, desc=phase.capitalize(), leave=False):
                device_batch = self._move_batch(batch)
                output = self._model_output(device_batch)
                loss = self.model.compute_loss(
                    output,
                    device_batch,
                    include_vade_kl=include_vade_kl,
                )

                if phase == "train":
                    self.optimizer.zero_grad()
                    loss.loss.backward()
                    if self.config.gradient_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.gradient_clip_norm,
                        )
                    self.optimizer.step()

                self.losses.update(device_batch["x"].size(0), loss)

                if survival_metrics is not None:
                    for m in survival_metrics.values():
                        m.update(
                            weibull_event_probability(
                                output.weibull_shape,
                                output.weibull_scale,
                                self.config.risk_horizon,
                            ),
                            device_batch["survival_time"],
                            device_batch["event"],
                        )
                if cluster_metrics is not None:
                    for m in cluster_metrics.values():
                        m.update(
                            torch.argmax(output.cluster_probabilities, dim=-1),
                            device_batch["cluster_label"],
                        )

        return self.losses.compute(), {
            **(
                {}
                if survival_metrics is None
                else {k: m.compute().item() for k, m in survival_metrics.items()}
            ),
            **(
                {}
                if cluster_metrics is None
                else {k: m.compute().item() for k, m in cluster_metrics.items()}
            ),
        }

    def _collect_outputs(self, data: ClinicalTimeSeriesDataset) -> tuple[TrailsModelOutput, Batch]:
        """按顺序收集并拼接整个数据集的模型输出和批次。"""
        data = data.with_return_kind(self._model_return_kind())
        loader = make_data_loader(data, self.config, shuffle=False)
        outputs: list[TrailsModelOutput] = []
        batches: list[Batch] = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                device_batch = self._move_batch(batch)
                outputs.append(self._model_output(device_batch))
                batches.append(device_batch)
        return concatenate_outputs(outputs), concatenate_batches(batches)

    def _move_batch(self, batch: Batch) -> Batch:
        """将批次中的全部张量移动到训练设备。"""
        return {name: value.to(self.config.device) for name, value in batch.items()}

    def _collect_latent_means(self, data: ClinicalTimeSeriesDataset) -> Tensor:
        """在 CPU 上收集整个数据集的确定性潜空间均值。"""
        data = data.with_return_kind(self._model_return_kind())
        loader = make_data_loader(data, self.config, shuffle=False)
        latent_means: list[Tensor] = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                device_batch = self._move_batch(batch)
                output = self._model_output(device_batch)
                latent_means.append(output.latent_mean.detach().cpu())
        return torch.cat(latent_means, dim=0)

    def _model_output(self, batch: Batch) -> TrailsModelOutput:
        """根据批次字段将 aligned 或 compact 输入分派给模型。"""
        if "delta_time" in batch and "sequence_lengths" in batch:
            return self.model(
                times=batch["times"],
                x=batch["x"],
                mask=batch["mask"],
                delta_time=batch["delta_time"],
                sequence_lengths=batch["sequence_lengths"],
            )
        if "feature_lengths" in batch:
            return self.model(
                times=batch["times"],
                x=batch["x"],
                mask=batch["mask"],
                feature_lengths=batch["feature_lengths"],
            )
        raise ValueError("Batch must contain either aligned or compact time-series fields.")

    def _model_return_kind(self) -> Literal["aligned", "compact"]:
        """根据输入层类型返回训练器需要的数据视图。"""
        return "compact" if self.model.model_config.encoder.input.kind == "mtan2" else "aligned"


def concatenate_batches(batches: list[Batch]) -> Batch:
    """将多个变长批次沿患者维拼接为单个批次。

    时间轴先补齐到所有批次的最大长度，并根据输入视图保留
    ``delta_time/sequence_lengths`` 或 ``feature_lengths`` 以及可选簇标签。
    """
    max_length = max(int(batch["times"].shape[1]) for batch in batches)
    batch = {
        "times": torch.cat([pad_time_axis(batch["times"], max_length) for batch in batches], dim=0),
        "x": torch.cat([pad_time_axis(batch["x"], max_length) for batch in batches], dim=0),
        "mask": torch.cat([pad_time_axis(batch["mask"], max_length) for batch in batches], dim=0),
        "survival_time": torch.cat([batch["survival_time"] for batch in batches], dim=0),
        "event": torch.cat([batch["event"] for batch in batches], dim=0),
    }
    if "delta_time" in batches[0]:
        batch["delta_time"] = torch.cat(
            [pad_time_axis(batch["delta_time"], max_length) for batch in batches],
            dim=0,
        )
        batch["sequence_lengths"] = torch.cat(
            [batch["sequence_lengths"] for batch in batches], dim=0
        )
    if "feature_lengths" in batches[0]:
        batch["feature_lengths"] = torch.cat([batch["feature_lengths"] for batch in batches], dim=0)
    if "cluster_label" in batches[0]:
        batch["cluster_label"] = torch.cat([batch["cluster_label"] for batch in batches], dim=0)
    return batch


def concatenate_outputs(outputs: list[TrailsModelOutput]) -> TrailsModelOutput:
    """补齐重建时间轴并沿患者维拼接模型输出。"""
    max_length = max(int(output.reconstruction.shape[1]) for output in outputs)
    return TrailsModelOutput(
        reconstruction=torch.cat(
            [pad_time_axis(output.reconstruction, max_length) for output in outputs],
            dim=0,
        ),
        latent_mean=torch.cat([output.latent_mean for output in outputs], dim=0),
        latent_log_variance=torch.cat([output.latent_log_variance for output in outputs], dim=0),
        latent=torch.cat([output.latent for output in outputs], dim=0),
        cluster_logits=torch.cat([output.cluster_logits for output in outputs], dim=0),
        cluster_probabilities=torch.cat(
            [output.cluster_probabilities for output in outputs],
            dim=0,
        ),
        weibull_shape=torch.cat([output.weibull_shape for output in outputs], dim=0),
        weibull_scale=torch.cat([output.weibull_scale for output in outputs], dim=0),
    )


def pad_time_axis(tensor: Tensor, target_length: int) -> Tensor:
    """在张量第二维末尾补零到目标时间长度。"""
    current_length = int(tensor.shape[1])
    if current_length == target_length:
        return tensor
    padded_shape = list(tensor.shape)
    padded_shape[1] = target_length
    padded = tensor.new_zeros(padded_shape)
    padded[:, :current_length] = tensor
    return padded


def fit_kmeans_mixture(
    latents: Tensor,
    *,
    n_clusters: int,
    n_iters: int,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """用确定性 K-means 估计高斯混合先验初值。

    初始中心从种子控制的样本排列中选择；迭代中若出现空簇，则使用距现有中心
    最远的样本重置该中心。最终返回经验混合比例、中心和带 ``1e-4`` 下界的
    各簇对角方差。

    参数：
        latents: 形状为 ``(n_samples, latent_dim)`` 的潜空间表示。
        n_clusters: 混合分量数量。
        n_iters: K-means 更新轮数。
        seed: 初始中心选择随机种子。

    返回：
        ``(prior_probabilities, means, variances)`` 三个 CPU 张量。

    异常：
        ValueError: 当样本数少于簇数时抛出。
    """
    if int(latents.shape[0]) < n_clusters:
        raise ValueError("At least one latent embedding per cluster is required.")
    generator = torch.Generator()
    generator.manual_seed(seed)
    latents = latents.float().cpu()
    permutation = torch.randperm(int(latents.shape[0]), generator=generator)
    centers = latents[permutation[:n_clusters]].clone()

    for _iteration in range(n_iters):
        distances = torch.cdist(latents, centers)
        assignments = torch.argmin(distances, dim=1)
        closest_distance = torch.min(distances, dim=1).values
        for cluster in range(n_clusters):
            cluster_mask = assignments == cluster
            if bool(cluster_mask.any()):
                centers[cluster] = latents[cluster_mask].mean(dim=0)
            else:
                centers[cluster] = latents[torch.argmax(closest_distance)]

    distances = torch.cdist(latents, centers)
    assignments = torch.argmin(distances, dim=1)
    counts = torch.bincount(assignments, minlength=n_clusters).float().clamp_min(1.0)
    prior_probabilities = counts / counts.sum()
    global_variance = latents.var(dim=0, unbiased=False).clamp_min(1e-4)
    variances = torch.zeros(n_clusters, int(latents.shape[1]), dtype=latents.dtype)
    for cluster in range(n_clusters):
        cluster_mask = assignments == cluster
        if bool(cluster_mask.any()):
            variances[cluster] = latents[cluster_mask].var(dim=0, unbiased=False).clamp_min(1e-4)
        else:
            variances[cluster] = global_variance
    return prior_probabilities, centers, variances

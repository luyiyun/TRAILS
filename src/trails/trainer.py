from __future__ import annotations

import copy
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
)
from .model import TrailsLossBreakdown, TrailsModelOutput, TrailsSurvVaderModel
from .progress import ProgressBar


class HistoryEntry(TypedDict):
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
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total: dict[str, float] = {}
        self.count: int = 0

    def update(self, batch_size: int, batch_loss: TrailsLossBreakdown) -> None:
        self.count += batch_size
        for k, v in batch_loss.items():
            self.total[k] = self.total.get(k, 0.0) + v.item() * batch_size

    def compute(self) -> dict[str, float]:
        return {name: value / max(1, self.count) for name, value in self.total.items()}


class EarlyStopper:
    def __init__(
        self,
        patience: int,
        monitor: Literal["loss", "cindex"],
        min_delta: float,
        has_validation: bool = True,
    ) -> None:
        self.patience = patience
        self.monitor = monitor
        self.min_delta = min_delta
        self.has_validation = has_validation
        self.reset()

    def reset(self):
        self.best_state: dict[str, Tensor] | None = None
        self.best_value: float | None = None
        self.best_global_epoch: int | None = None
        self.stale_epochs = 0

    def update(self, entry: HistoryEntry, model: TrailsSurvVaderModel) -> bool:
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
        split_name = "valid" if has_validation else "train"
        return f"{split_name}/{self.monitor}"

    def _monitor_value(self, entry: HistoryEntry) -> float:
        metrics = entry["valid"] if "valid" in entry else entry["train"]
        split_name = "valid" if "valid" in entry else "train"
        name = self.monitor
        if name not in metrics:
            raise ValueError(f"Early stopping monitor '{split_name}/{name}' is unavailable.")
        return float(metrics[name])

    def _is_monitor_improved(self, value: float, best_value: float | None) -> bool:
        if best_value is None:
            return True
        if self.monitor == "loss":
            return value < best_value - self.min_delta
        return value > best_value + self.min_delta


class TrailsTrainer:
    def __init__(self, model: TrailsSurvVaderModel, config: TrainerConfig) -> None:
        self.model = model.to(config.device)
        self.config = config
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

        self.losses = LossAccumulator()

    def fit(
        self,
        data: ClinicalTimeSeriesDataset,
        history_callback: HistoryCallback | None = None,
    ) -> list[HistoryEntry]:
        # 根据我们使用的input调整数据格式
        data = data.with_return_kind(self._model_return_kind())
        if self.config.valid_size > 0:
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
            if history_callback is not None:
                history_callback(entry)

            if (
                (epoch + 1) >= self.config.min_epochs
                and early_stopper is not None
                and early_stopper.update(entry, self.model)
            ):
                break

        if early_stopper is not None and early_stopper.best_state is not None:
            self.model.load_state_dict(early_stopper.best_state)
        return history

    def predict(self, data: ClinicalTimeSeriesDataset) -> Tensor:
        data = data.with_return_kind(self._model_return_kind())
        outputs, _batch = self._collect_outputs(data)
        return torch.argmax(outputs.cluster_probabilities, dim=-1).cpu()

    def predict_proba(self, data: ClinicalTimeSeriesDataset) -> Tensor:
        data = data.with_return_kind(self._model_return_kind())
        outputs, _batch = self._collect_outputs(data)
        return outputs.cluster_probabilities.cpu()

    def predict_risk(self, data: ClinicalTimeSeriesDataset) -> Tensor:
        data = data.with_return_kind(self._model_return_kind())
        outputs, _batch = self._collect_outputs(data)
        return self._risk_score(outputs).cpu()

    def test(self, data: ClinicalTimeSeriesDataset) -> dict[str, float]:
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
                            self._risk_score(output),
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
        return {name: value.to(self.config.device) for name, value in batch.items()}

    def _risk_score(self, output: TrailsModelOutput) -> Tensor:
        expected_scale = torch.sum(output.cluster_probabilities * output.weibull_scale, dim=-1)
        return -expected_scale

    def _collect_latent_means(self, data: ClinicalTimeSeriesDataset) -> Tensor:
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
        return "compact" if self.model.model_config.encoder.input.kind == "mtan2" else "aligned"


def concatenate_batches(batches: list[Batch]) -> Batch:
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

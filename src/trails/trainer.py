from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.clustering import AdjustedRandScore, NormalizedMutualInfoScore
from tqdm import tqdm

from .config import TrainerConfig
from .data import Batch, ClinicalTimeSeriesDataset, make_data_loader
from .metrics import (
    Cindex,
    masked_mse,
    vade_kl_loss,
    weibull_mixture_negative_log_likelihood,
)
from .model import TrailsModelOutput, TrailsSurvVaderModel


class HistoryEntry(TypedDict):
    epoch: int
    global_epoch: int
    stage: str
    train: dict[str, float]
    valid: NotRequired[dict[str, float]]


HistoryCallback = Callable[[HistoryEntry], None]


@dataclass(frozen=True)
class LossBreakdown:
    loss: Tensor
    reconstruction_loss: Tensor
    survival_loss: Tensor
    vade_kl_loss: Tensor

    def items(self) -> tuple[tuple[str, Tensor], ...]:
        return (
            ("loss", self.loss),
            ("reconstruction_loss", self.reconstruction_loss),
            ("survival_loss", self.survival_loss),
            ("vade_kl_loss", self.vade_kl_loss),
        )


class LossAccumulator:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total: dict[str, float] = {}
        self.count: int = 0

    def update(self, batch_size: int, batch_loss: LossBreakdown) -> None:
        self.count += batch_size
        for k, v in batch_loss.items():
            self.total[k] = self.total.get(k, 0.0) + v.item() * batch_size

    def compute(self) -> dict[str, float]:
        return {name: value / max(1, self.count) for name, value in self.total.items()}


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
        loader = make_data_loader(data, self.config, shuffle=True)
        if self.config.valid_size > 0:
            data, validation_data = data.split([1 - self.config.valid_size, self.config.valid_size])
            valid_loader = make_data_loader(validation_data, self.config, shuffle=False)
        else:
            validation_data = None
            valid_loader = None

        history: list[HistoryEntry] = []

        if self.config.warmup_epochs > 0:
            for epoch in tqdm(range(self.config.warmup_epochs), desc="Warmup"):
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
        cluster_metrics: dict[str, Metric] = (
            {"nmi": NormalizedMutualInfoScore(), "ari": AdjustedRandScore()}
            if validation_data is not None and validation_data.has_cluster_labels
            else {}
        )

        for epoch in tqdm(range(self.config.max_epochs), desc="Epoch"):
            losses, scores = self._epoch_loop(
                loader,
                phase="train",
                include_vade_kl=True,
                survival_metrics=survival_metrics,
                cluster_metrics=cluster_metrics,
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
                    cluster_metrics=cluster_metrics,
                )
                entry["valid"] = {**losses, **scores}

            history.append(entry)
            if history_callback is not None:
                history_callback(entry)
        return history

    def predict(self, data: ClinicalTimeSeriesDataset) -> Tensor:
        outputs, _batch = self._collect_outputs(data)
        return torch.argmax(outputs.cluster_probabilities, dim=-1).cpu()

    def predict_proba(self, data: ClinicalTimeSeriesDataset) -> Tensor:
        outputs, _batch = self._collect_outputs(data)
        return outputs.cluster_probabilities.cpu()

    def test(self, data: ClinicalTimeSeriesDataset) -> dict[str, float]:
        loader = make_data_loader(data, self.config, shuffle=False)
        survival_metrics: dict[str, Metric] = {"cindex": Cindex()}
        cluster_metrics: dict[str, Metric] = (
            {"nmi": NormalizedMutualInfoScore(), "ari": AdjustedRandScore()}
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

        return {**losses, **scores}

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
            for batch in tqdm(loader, desc=phase.capitalize(), leave=False):
                device_batch = self._move_batch(batch)
                output = self.model(
                    times=device_batch["times"],
                    x=device_batch["x"],
                    mask=device_batch["mask"],
                    delta_time=device_batch["delta_time"],
                    sequence_lengths=device_batch["sequence_lengths"],
                )
                loss = self._compute_loss(output, device_batch, include_vade_kl=include_vade_kl)

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

    # def _train_epoch(
    #     self,
    #     loader: torch.utils.data.DataLoader[Batch],
    #     *,
    #     include_vade_kl: bool,
    # ) -> dict[str, float]:
    #     self.losses.reset()
    #
    #     self.model.train()
    #     for batch in tqdm(loader, desc="Batch", leave=False):
    #         device_batch = self._move_batch(batch)
    #         output = self.model(
    #             times=device_batch["times"],
    #             x=device_batch["x"],
    #             mask=device_batch["mask"],
    #             delta_time=device_batch["delta_time"],
    #             sequence_lengths=device_batch["sequence_lengths"],
    #         )
    #         loss = self._compute_loss(output, device_batch, include_vade_kl=include_vade_kl)
    #         self.optimizer.zero_grad()
    #         loss.loss.backward()
    #         if self.config.gradient_clip_norm is not None:
    #             torch.nn.utils.clip_grad_norm_(
    #                 self.model.parameters(),
    #                 self.config.gradient_clip_norm,
    #             )
    #         self.optimizer.step()
    #         self.losses.update(device_batch["x"].size(0), loss)
    #
    #     return self.losses.compute()

    def _compute_loss(
        self,
        output: TrailsModelOutput,
        batch: Batch,
        *,
        include_vade_kl: bool,
    ) -> LossBreakdown:
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
                self.model.mixture_logits,
                self.model.mixture_means,
                self.model.mixture_log_variances,
            )
        else:
            vade_kl = reconstruction.new_zeros(())
        total = (
            self.config.reconstruction_weight * reconstruction
            + self.config.survival_weight * survival
            + self.config.cluster_weight * vade_kl
        )
        return LossBreakdown(
            loss=total,
            reconstruction_loss=reconstruction,
            survival_loss=survival,
            vade_kl_loss=vade_kl,
        )

    # def _evaluate(
    #     self,
    #     loader: torch.utils.data.DataLoader[Batch],
    #     *,
    #     include_vade_kl: bool,
    #     cal_cluster_metrics: bool = False,
    # ) -> dict[str, float]:
    #     self.losses.reset()
    #     if cal_cluster_metrics:
    #         self.metrics.reset()
    #
    #     risk_scores: list[Tensor] = []
    #     survival_times: list[Tensor] = []
    #     events: list[Tensor] = []
    #
    #     self.model.eval()
    #     with torch.no_grad():
    #         for batch in tqdm(loader, desc="Eval", leave=False):
    #             device_batch = self._move_batch(batch)
    #             output = self.model(
    #                 times=device_batch["times"],
    #                 x=device_batch["x"],
    #                 mask=device_batch["mask"],
    #                 delta_time=device_batch["delta_time"],
    #                 sequence_lengths=device_batch["sequence_lengths"],
    #             )
    #             loss = self._compute_loss(
    #                 output,
    #                 device_batch,
    #                 include_vade_kl=include_vade_kl,
    #             )
    #             self.losses.update(device_batch["x"].size(0), loss)
    #
    #             risk_scores.append(self._risk_score(output).cpu())
    #             survival_times.append(device_batch["survival_time"].cpu())
    #             events.append(device_batch["event"].cpu())
    #
    #             if cal_cluster_metrics:
    #                 self.metrics.update(
    #                     torch.argmax(output.cluster_probabilities, dim=-1),
    #                     device_batch["cluster_label"],
    #                 )
    #
    #     scores = self.losses.compute()
    #     scores["c_index"] = concordance_index(
    #         torch.cat(risk_scores),
    #         torch.cat(survival_times),
    #         torch.cat(events),
    #     )
    #     if cal_cluster_metrics:
    #         scores.update(self.metrics.compute())
    #     return scores

    def _collect_outputs(self, data: ClinicalTimeSeriesDataset) -> tuple[TrailsModelOutput, Batch]:
        loader = make_data_loader(data, self.config, shuffle=False)
        outputs: list[TrailsModelOutput] = []
        batches: list[Batch] = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                device_batch = self._move_batch(batch)
                outputs.append(
                    self.model(
                        times=device_batch["times"],
                        x=device_batch["x"],
                        mask=device_batch["mask"],
                        delta_time=device_batch["delta_time"],
                        sequence_lengths=device_batch["sequence_lengths"],
                    )
                )
                batches.append(device_batch)
        return concatenate_outputs(outputs), concatenate_batches(batches)

    def _move_batch(self, batch: Batch) -> Batch:
        return {name: value.to(self.config.device) for name, value in batch.items()}

    def _risk_score(self, output: TrailsModelOutput) -> Tensor:
        expected_scale = torch.sum(output.cluster_probabilities * output.weibull_scale, dim=-1)
        return -expected_scale

    def _collect_latent_means(self, data: ClinicalTimeSeriesDataset) -> Tensor:
        loader = make_data_loader(data, self.config, shuffle=False)
        latent_means: list[Tensor] = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                device_batch = self._move_batch(batch)
                output = self.model(
                    times=device_batch["times"],
                    x=device_batch["x"],
                    mask=device_batch["mask"],
                    delta_time=device_batch["delta_time"],
                    sequence_lengths=device_batch["sequence_lengths"],
                )
                latent_means.append(output.latent_mean.detach().cpu())
        return torch.cat(latent_means, dim=0)


def concatenate_batches(batches: list[Batch]) -> Batch:
    max_length = max(int(batch["times"].shape[1]) for batch in batches)
    batch = {
        "times": torch.cat([pad_time_axis(batch["times"], max_length) for batch in batches], dim=0),
        "x": torch.cat([pad_time_axis(batch["x"], max_length) for batch in batches], dim=0),
        "mask": torch.cat([pad_time_axis(batch["mask"], max_length) for batch in batches], dim=0),
        "delta_time": torch.cat(
            [pad_time_axis(batch["delta_time"], max_length) for batch in batches],
            dim=0,
        ),
        "sequence_lengths": torch.cat([batch["sequence_lengths"] for batch in batches], dim=0),
        "survival_time": torch.cat([batch["survival_time"] for batch in batches], dim=0),
        "event": torch.cat([batch["event"] for batch in batches], dim=0),
    }
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

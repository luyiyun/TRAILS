from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from tqdm import tqdm

from .config import TrainerConfig
from .data import Batch, ClinicalTimeSeriesDataset, make_data_loader
from .metrics import (
    cluster_balance_loss,
    concordance_index,
    masked_mse,
    weibull_mixture_negative_log_likelihood,
)
from .model import TrailsModelOutput, TrailsSurvVaderModel


@dataclass(frozen=True)
class LossBreakdown:
    total: Tensor
    reconstruction: Tensor
    survival: Tensor
    cluster: Tensor


class TrailsTrainer:
    def __init__(self, model: TrailsSurvVaderModel, config: TrainerConfig) -> None:
        self.model = model.to(config.device)
        self.config = config
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

    def fit(self, data: ClinicalTimeSeriesDataset) -> list[dict[str, float]]:
        loader = make_data_loader(data, self.config, shuffle=True)
        history: list[dict[str, float]] = []
        for _epoch in tqdm(range(self.config.max_epochs), desc="Epoch"):
            history.append(self._train_epoch(loader))
        return history

    def predict(self, data: ClinicalTimeSeriesDataset) -> Tensor:
        outputs, _batch = self._collect_outputs(data)
        return torch.argmax(outputs.cluster_logits, dim=-1).cpu()

    def test(self, data: ClinicalTimeSeriesDataset) -> dict[str, float]:
        outputs, batch = self._collect_outputs(data)
        loss = self._compute_loss(outputs, batch)
        risk_score = self._risk_score(outputs)
        c_index = concordance_index(
            risk_score.cpu(),
            batch["survival_time"].cpu(),
            batch["event"].cpu(),
        )
        return {
            "cluster_loss": float(loss.cluster.detach().cpu()),
            "c_index": c_index,
            "loss": float(loss.total.detach().cpu()),
            "reconstruction_loss": float(loss.reconstruction.detach().cpu()),
            "survival_loss": float(loss.survival.detach().cpu()),
        }

    def _train_epoch(self, loader: torch.utils.data.DataLoader[Batch]) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_batches = 0
        for batch in tqdm(loader, desc="Batch", leave=False):
            device_batch = self._move_batch(batch)
            output = self.model(
                device_batch["x"],
                device_batch["mask"],
                device_batch["delta_time"],
                device_batch["sequence_lengths"],
            )
            loss = self._compute_loss(output, device_batch)
            self.optimizer.zero_grad()
            loss.total.backward()
            if self.config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm,
                )
            self.optimizer.step()
            total_loss += float(loss.total.detach().cpu())
            total_batches += 1
        return {"loss": total_loss / max(total_batches, 1)}

    def _compute_loss(self, output: TrailsModelOutput, batch: Batch) -> LossBreakdown:
        reconstruction = masked_mse(output.reconstruction, batch["x"], batch["mask"])
        survival = weibull_mixture_negative_log_likelihood(
            output.cluster_logits,
            output.weibull_shape,
            output.weibull_scale,
            batch["survival_time"],
            batch["event"],
        )
        cluster = cluster_balance_loss(output.cluster_logits)
        total = (
            self.config.reconstruction_weight * reconstruction
            + self.config.survival_weight * survival
            + self.config.cluster_weight * cluster
        )
        return LossBreakdown(
            total=total,
            reconstruction=reconstruction,
            survival=survival,
            cluster=cluster,
        )

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
                        device_batch["x"],
                        device_batch["mask"],
                        device_batch["delta_time"],
                        device_batch["sequence_lengths"],
                    )
                )
                batches.append(device_batch)
        return concatenate_outputs(outputs), concatenate_batches(batches)

    def _move_batch(self, batch: Batch) -> Batch:
        return {name: value.to(self.config.device) for name, value in batch.items()}

    def _risk_score(self, output: TrailsModelOutput) -> Tensor:
        cluster_probabilities = torch.softmax(output.cluster_logits, dim=-1)
        expected_scale = torch.sum(cluster_probabilities * output.weibull_scale, dim=-1)
        return -expected_scale


def concatenate_batches(batches: list[Batch]) -> Batch:
    max_length = max(int(batch["times"].shape[1]) for batch in batches)
    return {
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
        "cluster_label": torch.cat([batch["cluster_label"] for batch in batches], dim=0),
    }


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

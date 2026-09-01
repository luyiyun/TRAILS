"""基于静态FPCA特征的VaDeSC深度生存聚类基线。"""

from __future__ import annotations

from pathlib import Path
from typing import Self

import joblib
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from trails import ClinicalTimeSeriesDataset

from .baseline_features import dataset_patient_ids, dataset_survival_arrays
from .baseline_fpca import UFPCAFeaturePipeline
from .baselines import BaselineCapability, BaselinePrediction


class VaDeSCNetwork(nn.Module):
    """复现VaDeSC的MLP-VAE、Gaussian mixture先验和Weibull生存头。"""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, int],
        latent_dim: int,
        n_clusters: int,
        weibull_shape: float,
    ) -> None:
        super().__init__()
        first, second = hidden_dims
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, first), nn.ReLU(), nn.Linear(first, second), nn.ReLU()
        )
        self.latent_mean = nn.Linear(second, latent_dim)
        self.latent_log_variance = nn.Linear(second, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, second),
            nn.ReLU(),
            nn.Linear(second, first),
            nn.ReLU(),
            nn.Linear(first, input_dim),
        )
        self.prior_logits = nn.Parameter(torch.zeros(n_clusters))
        self.cluster_means = nn.Parameter(torch.empty(n_clusters, latent_dim))
        self.cluster_log_variances = nn.Parameter(torch.zeros(n_clusters, latent_dim))
        self.survival_coefficients = nn.Parameter(torch.empty(n_clusters, latent_dim + 1))
        self.weibull_shape = weibull_shape
        nn.init.xavier_normal_(self.cluster_means)
        nn.init.xavier_normal_(self.survival_coefficients)

    def encode(self, features: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.encoder(features)
        return self.latent_mean(hidden), self.latent_log_variance(hidden).clamp(-12.0, 12.0)

    def components(
        self,
        latent: Tensor,
        time: Tensor | None = None,
        event: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        difference = latent.unsqueeze(1) - self.cluster_means.unsqueeze(0)
        log_variance = self.cluster_log_variances.clamp(-12.0, 12.0).unsqueeze(0)
        log_p_z = -0.5 * (
            np.log(2.0 * np.pi) + log_variance + difference.square() / log_variance.exp()
        ).sum(dim=-1)
        linear = latent @ self.survival_coefficients[:, :-1].T
        linear = linear + self.survival_coefficients[:, -1]
        scales = F.softplus(-linear).clamp_min(1e-4)
        log_joint = F.log_softmax(self.prior_logits, dim=0).unsqueeze(0) + log_p_z
        log_survival = torch.zeros_like(log_joint)
        if time is not None and event is not None:
            log_time = time.clamp_min(1e-6).log().unsqueeze(1)
            log_scale = scales.log()
            cumulative_hazard = (time.unsqueeze(1) / scales).pow(self.weibull_shape)
            log_survival = event.unsqueeze(1) * (
                np.log(self.weibull_shape)
                - log_scale
                + (self.weibull_shape - 1.0) * (log_time - log_scale)
            ) - cumulative_hazard.clamp_max(1e6)
            log_joint = log_joint + log_survival
        return F.softmax(log_joint, dim=-1), scales, log_p_z, log_survival

    def loss(self, features: Tensor, time: Tensor, event: Tensor, *, sample: bool) -> Tensor:
        mean, log_variance = self.encode(features)
        latent = mean
        if sample:
            latent = mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)
        reconstruction = self.decoder(latent)
        posterior, _scales, log_p_z, log_survival = self.components(latent, time, event)
        log_prior = F.log_softmax(self.prior_logits, dim=0).unsqueeze(0)
        mixture_kl = (posterior * (posterior.clamp_min(1e-8).log() - log_prior - log_p_z)).sum(
            dim=-1
        )
        latent_entropy = -0.5 * (log_variance + 1.0).sum(dim=-1)
        survival = -(posterior * log_survival).sum(dim=-1)
        reconstruction_loss = (reconstruction - features).square().mean(dim=-1)
        return (reconstruction_loss + mixture_kl + latent_entropy + survival).mean()


class VaDeSCBaseline:
    """用train-fitted FPCA训练VaDeSC并统一导出聚类和生存预测。"""

    capabilities: frozenset[BaselineCapability] = frozenset({"cluster", "survival"})

    def __init__(
        self,
        name: str,
        n_clusters: int,
        seed: int,
        n_components: int,
        grid_size: int,
        time_start: float,
        time_end: float,
        hidden_dims: tuple[int, int],
        latent_dim: int,
        weibull_shape: float,
        max_epochs: int,
        patience: int,
        learning_rate: float,
        batch_size: int,
        device: str,
    ) -> None:
        if weibull_shape <= 0.0 or max_epochs <= 0 or patience <= 0:
            raise ValueError("VaDeSC的shape、epoch和patience必须为正数")
        self.name, self.n_clusters, self.seed = name, n_clusters, seed
        self.hidden_dims, self.latent_dim = hidden_dims, latent_dim
        self.weibull_shape, self.max_epochs = weibull_shape, max_epochs
        self.patience, self.learning_rate, self.batch_size = patience, learning_rate, batch_size
        self.device = device
        self.features = UFPCAFeaturePipeline(n_components, grid_size, time_start, time_end)
        self.time_scale: float | None = None
        self.model: VaDeSCNetwork | None = None
        self.best_epoch: int | None = None
        self.epochs_trained = 0
        self.best_validation_loss: float | None = None

    def fit(
        self,
        train: ClinicalTimeSeriesDataset,
        validation: ClinicalTimeSeriesDataset,
    ) -> Self:
        """只用train拟合FPCA和时间尺度，以validation ELBO早停。"""
        torch.manual_seed(self.seed)
        features = self.features.fit_transform(train).astype(np.float32)
        valid_features = self.features.transform(validation).astype(np.float32)
        train_event, train_time = dataset_survival_arrays(train)
        valid_event, valid_time = dataset_survival_arrays(validation)
        self.time_scale = float(train_time.max())
        training_data = TensorDataset(
            torch.from_numpy(features),
            torch.from_numpy((train_time / self.time_scale).astype(np.float32)),
            torch.from_numpy(train_event.astype(np.float32)),
        )
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            training_data, batch_size=self.batch_size, shuffle=True, generator=generator
        )
        model = VaDeSCNetwork(
            features.shape[1],
            self.hidden_dims,
            self.latent_dim,
            self.n_clusters,
            self.weibull_shape,
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        validation_tensors = (
            torch.from_numpy(valid_features).to(self.device),
            torch.from_numpy((valid_time / self.time_scale).astype(np.float32)).to(self.device),
            torch.from_numpy(valid_event.astype(np.float32)).to(self.device),
        )

        best_loss, stale_epochs, best_state = float("inf"), 0, None
        for epoch in range(1, self.max_epochs + 1):
            model.train()
            for batch in loader:
                optimizer.zero_grad()
                loss = model.loss(*(value.to(self.device) for value in batch), sample=True)
                if not torch.isfinite(loss):
                    raise RuntimeError("VaDeSC训练产生非有限loss")
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                valid_loss = float(model.loss(*validation_tensors, sample=False).item())
            if not np.isfinite(valid_loss):
                raise RuntimeError("VaDeSC validation产生非有限loss")
            self.epochs_trained = epoch
            if valid_loss < best_loss:
                best_loss, stale_epochs, self.best_epoch = valid_loss, 0, epoch
                best_state = {
                    name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                }
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break
        if best_state is None:
            raise RuntimeError("VaDeSC未得到可恢复的validation模型")
        model.load_state_dict(best_state)
        self.model = model.cpu().eval()
        self.best_validation_loss = best_loss
        return self

    def predict(
        self,
        data: ClinicalTimeSeriesDataset,
        *,
        prediction_times: NDArray[np.float64],
        risk_horizon: float,
    ) -> BaselinePrediction:
        """外部预测只用p(c|z)，再按硬分配簇计算Weibull生存曲线。"""
        if self.model is None or self.time_scale is None:
            raise RuntimeError("VaDeSCBaseline必须先拟合")
        features = torch.from_numpy(self.features.transform(data).astype(np.float32))
        with torch.no_grad():
            mean, _ = self.model.encode(features)
            probabilities, scales, _, _ = self.model.components(mean)
            labels = probabilities.argmax(dim=1)
            selected_scales = scales.gather(1, labels.unsqueeze(1)).squeeze(1)
            times = torch.from_numpy(prediction_times / self.time_scale).float()
            survival = torch.exp(
                -(times.unsqueeze(0) / selected_scales.unsqueeze(1)).pow(self.weibull_shape)
            )
            horizon = selected_scales.new_tensor(risk_horizon / self.time_scale)
            risk = 1.0 - torch.exp(-(horizon / selected_scales).pow(self.weibull_shape))
        return BaselinePrediction(
            method_name=self.name,
            patient_ids=dataset_patient_ids(data),
            cluster_labels=labels.numpy().astype(np.int64),
            n_clusters=self.n_clusters,
            risk_score=risk.numpy().astype(np.float64),
            risk_horizon=risk_horizon,
            survival_times=prediction_times.copy(),
            survival_probabilities=survival.numpy().astype(np.float64),
        )

    def save_model(self, path: Path) -> None:
        """保存FPCA、时间尺度、早停状态和已拟合PyTorch模型。"""
        if self.model is None:
            raise RuntimeError("VaDeSCBaseline必须先拟合")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

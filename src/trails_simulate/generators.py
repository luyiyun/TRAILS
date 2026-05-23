from __future__ import annotations

import math
from typing import Self

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor

from trails.data import AlignedClinicalSample, ClinicalTimeSeriesDataset, make_clinical_sample

from .utils import (
    _low_rank_matrix,
    _random_spd_matrix,
    _sample_latent_profiles,
    _sequence_mask,
    _standardize_active_profile,
    _uniform,
)

DEFAULT_FEATURE_NAMES = [
    "hemoglobin",
    "albumin",
    "creatinine",
    "alanine_aminotransferase",
    "c_reactive_protein",
    "neutrophil_count",
    "lymphocyte_count",
    "carcinoembryonic_antigen",
    "carbohydrate_antigen_199",
    "tumor_size",
]


class ClinicalTimeSeriesDatasetGeneratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_clusters: int = Field(default=3, gt=1)
    min_visits: int = Field(default=4, gt=0)
    max_visits: int = Field(default=8, gt=0)
    hidden_size: int = Field(default=100, gt=0)
    latent_dim: int = Field(default=5, gt=0)
    attention_layers: int = Field(default=3, gt=0)
    attention_heads: int | None = None
    censoring_rate: float = Field(default=0.3, ge=0.0, le=1.0)
    weibull_shape: float = Field(default=1.0, gt=0.0)
    x_low: float = -10.0
    x_high: float = 10.0
    beta_low: float = -2.5
    beta_high: float = 2.5
    feature_names: list[str] = Field(default_factory=lambda: list(DEFAULT_FEATURE_NAMES))
    mechanism_seed: int = 20260517

    @model_validator(mode="after")
    def check_config(self) -> Self:
        if self.max_visits < self.min_visits:
            raise ValueError("max_visits must be greater than or equal to min_visits.")
        if self.attention_heads is not None and (
            self.attention_heads <= 0 or self.hidden_size % self.attention_heads != 0
        ):
            raise ValueError("attention_heads must divide hidden_size.")
        if self.x_low >= self.x_high or self.beta_low >= self.beta_high:
            raise ValueError("low bounds must be smaller than high bounds.")

        return self


class ClinicalTimeSeriesDatasetGenerator:
    def __init__(
        self,
        config: ClinicalTimeSeriesDatasetGeneratorConfig,
        mechanism_seed: int | None = None,
    ):
        self.config = config
        self.mechanism_seed = (
            self.config.mechanism_seed if mechanism_seed is None else mechanism_seed
        )
        mechanism_generator = torch.Generator().manual_seed(self.mechanism_seed)
        self.resolved_attention_heads = config.attention_heads or _default_attention_heads(
            config.hidden_size
        )
        n_features = len(config.feature_names)

        # 机制参数在实例化时固定；simulate 只改变样本抽样随机性。
        self.cluster_prior = torch.ones(config.n_clusters, dtype=torch.float32) / config.n_clusters
        self.cluster_means = _uniform(
            (config.n_clusters, config.latent_dim),
            config.x_low,
            config.x_high,
            mechanism_generator,
        )
        self.cluster_covariances = torch.stack(
            [
                _random_spd_matrix(config.latent_dim, mechanism_generator)
                for _cluster in range(config.n_clusters)
            ]
        )
        self.survival_coefficients = _uniform(
            (config.n_clusters, config.latent_dim),
            config.beta_low,
            config.beta_high,
            mechanism_generator,
        )
        self.survival_intercepts = _uniform(
            (config.n_clusters,),
            config.beta_low,
            config.beta_high,
            mechanism_generator,
        )
        self.profile_weight = _low_rank_matrix(
            config.latent_dim,
            config.hidden_size * config.max_visits,
            mechanism_generator,
        )
        self.attention_weights = [
            (
                _low_rank_matrix(config.hidden_size, config.hidden_size, mechanism_generator),
                _low_rank_matrix(config.hidden_size, config.hidden_size, mechanism_generator),
                _low_rank_matrix(config.hidden_size, config.hidden_size, mechanism_generator),
            )
            for _layer in range(config.attention_layers)
        ]
        decoder_intermediate_size = max(config.hidden_size // 2, n_features)
        self.decoder_weight_1 = _low_rank_matrix(
            config.hidden_size,
            decoder_intermediate_size,
            mechanism_generator,
        )
        self.decoder_weight_2 = _low_rank_matrix(
            decoder_intermediate_size,
            n_features,
            mechanism_generator,
        )
        self.feature_location = torch.linspace(-0.8, 0.8, n_features)
        self.feature_scale = torch.linspace(0.6, 1.6, n_features)

    def simulate(
        self,
        n_patients: int,
        seed: int = 2026,
    ) -> ClinicalTimeSeriesDataset:
        if n_patients <= self.config.n_clusters:
            raise ValueError("n_patients must be greater than n_clusters.")

        generator = torch.Generator().manual_seed(seed)
        names = self.config.feature_names

        # 1. 固定机制参数下抽取本批患者的 cluster 与 latent profile。
        cluster_labels = torch.multinomial(
            self.cluster_prior,
            num_samples=n_patients,
            replacement=True,
            generator=generator,
        )
        latent_z = _sample_latent_profiles(
            cluster_labels=cluster_labels,
            cluster_means=self.cluster_means,
            cluster_covariances=self.cluster_covariances,
            generator=generator,
        )

        # 2. 生存变量模拟只使用本次样本 seed；风险机制由 __init__ 固定。
        event_times, events = _sample_survival_times(
            latent_z=latent_z,
            cluster_labels=cluster_labels,
            survival_coefficients=self.survival_coefficients,
            survival_intercepts=self.survival_intercepts,
            weibull_shape=self.config.weibull_shape,
            censoring_rate=self.config.censoring_rate,
            generator=generator,
        )

        # 3. Z -> patient-level hidden trajectory，映射矩阵固定，患者 latent 抽样变化。
        hidden_profile = torch.relu(latent_z @ self.profile_weight).reshape(
            n_patients,
            self.config.max_visits,
            self.config.hidden_size,
        )
        hidden_profile = _standardize_active_profile(hidden_profile)
        sequence_lengths = torch.randint(
            self.config.min_visits,
            self.config.max_visits + 1,
            size=(n_patients,),
            generator=generator,
        )
        visit_mask = _sequence_mask(sequence_lengths, self.config.max_visits)
        hidden_profile = hidden_profile * visit_mask.unsqueeze(-1)

        # 4. pseudo-attention 的权重固定，保留原模拟器的非线性轨迹扰动。
        for query_weight, key_weight, value_weight in self.attention_weights:
            attention_output = _pseudo_attention(
                hidden_profile,
                visit_mask,
                hidden_size=self.config.hidden_size,
                attention_heads=self.resolved_attention_heads,
                query_weight=query_weight,
                key_weight=key_weight,
                value_weight=value_weight,
            )
            hidden_profile = (hidden_profile + attention_output) * visit_mask.unsqueeze(-1)

        # 5. 连续临床变量 decoder 固定；测量噪声仍由样本 seed 控制。
        true_values = _decode_continuous_clinical_values(
            hidden_profile,
            weight_1=self.decoder_weight_1,
            weight_2=self.decoder_weight_2,
            feature_location=self.feature_location,
            feature_scale=self.feature_scale,
            generator=generator,
        )

        # 6. 非同步采样
        samples: list[AlignedClinicalSample] = []
        for patient_index in range(n_patients):
            seq_len = int(sequence_lengths[patient_index])
            # 检查记录只能发生在患者实际观测终点之前；否则会出现事件/删失后仍有检查的矛盾。
            times = _sample_visit_times(seq_len, event_times[patient_index], generator)
            values = true_values[patient_index, :seq_len]
            severity = torch.sigmoid(latent_z[patient_index, 0] / max(abs(self.config.x_high), 1.0))
            mask = _sample_asynchronous_observation_mask(
                values=values,
                severity=severity,
                generator=generator,
            )
            delta_time = _compute_delta_time(times, mask)

            samples.append(
                make_clinical_sample(
                    times=times,
                    x=values * mask,
                    mask=mask,
                    delta_time=delta_time,
                    survival_time=event_times[patient_index],
                    event=events[patient_index],
                    cluster_label=int(cluster_labels[patient_index]),
                )
            )

        metadata = {
            "generation_params": {
                "n_patients": n_patients,
                "n_clusters": self.config.n_clusters,
                "min_visits": self.config.min_visits,
                "max_visits": self.config.max_visits,
                "latent_dim": self.config.latent_dim,
                "hidden_size": self.config.hidden_size,
                "attention_layers": self.config.attention_layers,
                "attention_heads": self.resolved_attention_heads,
                "censoring_rate": self.config.censoring_rate,
                "weibull_shape": self.config.weibull_shape,
                "x_low": self.config.x_low,
                "x_high": self.config.x_high,
                "beta_low": self.config.beta_low,
                "beta_high": self.config.beta_high,
                "mechanism_seed": self.mechanism_seed,
                "sample_seed": seed,
            },
            "cluster_prior": self.cluster_prior,
            "cluster_means": self.cluster_means,
            "cluster_covariances": self.cluster_covariances,
            "latent_z": latent_z,
            "mechanism_parameters": self.mechanism_parameters(),
            "sequence_lengths": sequence_lengths,
            "survival_coefficients": self.survival_coefficients,
            "survival_intercepts": self.survival_intercepts,
        }
        return ClinicalTimeSeriesDataset(
            samples,
            feature_names=names,
            description=(
                "VaDeSC-EHR-style synthetic asynchronous continuous clinical time-series "
                "dataset with latent clusters and Weibull survival outcomes."
            ),
            metadata=metadata,
        )

    def mechanism_parameters(self) -> dict[str, object]:
        return {
            "attention_weights": [
                {
                    "key": key_weight,
                    "query": query_weight,
                    "value": value_weight,
                }
                for query_weight, key_weight, value_weight in self.attention_weights
            ],
            "decoder_weight_1": self.decoder_weight_1,
            "decoder_weight_2": self.decoder_weight_2,
            "feature_location": self.feature_location,
            "feature_scale": self.feature_scale,
            "profile_weight": self.profile_weight,
        }


def _pseudo_attention(
    inputs: Tensor,
    visit_mask: Tensor,
    *,
    hidden_size: int,
    attention_heads: int,
    query_weight: Tensor,
    key_weight: Tensor,
    value_weight: Tensor,
) -> Tensor:
    head_size = hidden_size // attention_heads
    query = (inputs @ query_weight).reshape(
        inputs.shape[0], inputs.shape[1], attention_heads, head_size
    )
    key = (inputs @ key_weight).reshape(
        inputs.shape[0], inputs.shape[1], attention_heads, head_size
    )
    value = (inputs @ value_weight).reshape(
        inputs.shape[0], inputs.shape[1], attention_heads, head_size
    )
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(float(head_size))
    scores = scores.masked_fill(~visit_mask[:, None, None, :], -torch.inf)
    attention = torch.softmax(scores, dim=-1)
    attended = torch.matmul(attention, value)
    attended = attended.transpose(1, 2).reshape(inputs.shape[0], inputs.shape[1], hidden_size)
    return attended * visit_mask.unsqueeze(-1)


def _decode_continuous_clinical_values(
    hidden_profile: Tensor,
    *,
    weight_1: Tensor,
    weight_2: Tensor,
    feature_location: Tensor,
    feature_scale: Tensor,
    generator: torch.Generator,
) -> Tensor:
    raw_values = torch.relu(hidden_profile @ weight_1) @ weight_2
    noise = 0.08 * torch.randn(*raw_values.shape, generator=generator)
    return feature_location + feature_scale * torch.tanh(raw_values) + noise


def _sample_survival_times(
    *,
    latent_z: Tensor,
    cluster_labels: Tensor,
    survival_coefficients: Tensor,
    survival_intercepts: Tensor,
    weibull_shape: float,
    censoring_rate: float,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    n_patients = int(latent_z.shape[0])
    event_times = torch.zeros(n_patients)
    events = (torch.rand(n_patients, generator=generator) >= censoring_rate).float()
    n_events = int(events.sum().item())

    coefficient = survival_coefficients[cluster_labels]
    intercept = survival_intercepts[cluster_labels]
    risk = (latent_z * coefficient).sum(dim=1) / latent_z.size(1) + intercept
    scale = torch.nn.functional.softplus(risk)
    event_draw = torch.rand(n_patients, generator=generator).clamp(1e-4, 1.0 - 1e-4)
    true_time = scale * (-torch.log(1.0 - event_draw)) ** (1.0 / weibull_shape)

    mask_event_1 = events == 1
    event_times[mask_event_1] = true_time[mask_event_1]
    event_times[~mask_event_1] = (
        torch.rand(n_patients - n_events, generator=generator) * true_time[~mask_event_1]
    )

    return event_times, events


def _default_attention_heads(hidden_size: int) -> int:
    for candidate in [10, 8, 6, 5, 4, 3, 2]:
        if hidden_size % candidate == 0:
            return candidate
    return 1


def _sample_asynchronous_observation_mask(
    *,
    values: Tensor,
    severity: Tensor,
    generator: torch.Generator,
) -> Tensor:
    n_visits, n_features = values.shape
    base_rates = torch.linspace(0.25, 0.7, n_features)
    value_signal = 0.08 * torch.sigmoid(values.abs())
    probability = (base_rates.unsqueeze(0) + 0.18 * severity + value_signal).clamp(0.05, 0.95)
    mask = (torch.rand(n_visits, n_features, generator=generator) < probability).float()
    _ensure_at_least_one_observation_per_visit(mask, generator)
    _ensure_asynchronous_visit(mask, generator)
    return mask


def _compute_delta_time(times: Tensor, mask: Tensor) -> Tensor:
    delta_time = torch.zeros_like(mask)
    for step in range(1, int(times.shape[0])):
        gap = times[step] - times[step - 1]
        delta_time[step] = torch.where(mask[step - 1] > 0, gap, delta_time[step - 1] + gap)
    return delta_time


def _sample_visit_times(seq_len: int, observed_end: Tensor, generator: torch.Generator) -> Tensor:
    upper = observed_end.clamp_min(1e-3)
    times = torch.sort(torch.rand(seq_len, generator=generator) * upper).values
    times[0] = 0.0
    return times


def _ensure_at_least_one_observation_per_visit(mask: Tensor, generator: torch.Generator) -> None:
    n_features = int(mask.shape[-1])
    for step in range(int(mask.shape[0])):
        if int(mask[step].sum()) == 0:
            feature = int(torch.randint(n_features, size=(1,), generator=generator))
            mask[step, feature] = 1.0


def _ensure_asynchronous_visit(mask: Tensor, generator: torch.Generator) -> None:
    n_visits, n_features = mask.shape
    if n_features < 2:
        return
    has_partial_visit = bool(torch.any((mask.sum(dim=1) > 0) & (mask.sum(dim=1) < n_features)))
    if not has_partial_visit:
        visit = int(torch.randint(n_visits, size=(1,), generator=generator))
        feature = int(torch.randint(n_features, size=(1,), generator=generator))
        mask[visit, feature] = 0.0

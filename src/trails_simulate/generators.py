from __future__ import annotations

import math

import torch
from torch import Tensor

from trails.data import ClinicalSample, ClinicalTimeSeriesDataset, make_clinical_sample

from .utils import (
    _low_rank_matrix,
    _random_nonlin_map,
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


def _pseudo_attention(
    inputs: Tensor,
    visit_mask: Tensor,
    *,
    hidden_size: int,
    attention_heads: int,
    generator: torch.Generator,
) -> Tensor:
    head_size = hidden_size // attention_heads
    query_weight = _low_rank_matrix(hidden_size, hidden_size, generator)
    key_weight = _low_rank_matrix(hidden_size, hidden_size, generator)
    value_weight = _low_rank_matrix(hidden_size, hidden_size, generator)

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
    scores = scores.masked_fill(~visit_mask[:, None, None, :], -10000.0)
    attention = torch.softmax(scores, dim=-1)
    attended = torch.matmul(attention, value)
    attended = attended.transpose(1, 2).reshape(inputs.shape[0], inputs.shape[1], hidden_size)
    return attended * visit_mask.unsqueeze(-1)


def _decode_continuous_clinical_values(
    hidden_profile: Tensor,
    *,
    n_features: int,
    generator: torch.Generator,
) -> Tensor:
    hidden_size = int(hidden_profile.shape[-1])
    intermediate_size = max(hidden_size // 2, n_features)
    weight_1 = _low_rank_matrix(hidden_size, intermediate_size, generator)
    weight_2 = _low_rank_matrix(intermediate_size, n_features, generator)
    feature_location = torch.linspace(-0.8, 0.8, n_features)
    feature_scale = torch.linspace(0.6, 1.6, n_features)
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


def generate_clinical_time_series_dataset(
    *,
    n_patients: int = 128,
    n_clusters: int = 3,
    min_visits: int = 4,
    max_visits: int = 8,
    latent_dim: int = 5,
    hidden_size: int = 100,
    attention_layers: int = 3,
    attention_heads: int | None = None,
    censoring_rate: float = 0.3,
    weibull_shape: float = 1.0,
    x_low: float = -10.0,
    x_high: float = 10.0,
    beta_low: float = -2.5,
    beta_high: float = 2.5,
    feature_names: list[str] | None = None,
    seed: int = 2026,
) -> ClinicalTimeSeriesDataset:
    _validate_generation_args(
        n_patients=n_patients,
        n_clusters=n_clusters,
        min_visits=min_visits,
        max_visits=max_visits,
        latent_dim=latent_dim,
        hidden_size=hidden_size,
        attention_layers=attention_layers,
        attention_heads=attention_heads,
        censoring_rate=censoring_rate,
        weibull_shape=weibull_shape,
        x_low=x_low,
        x_high=x_high,
        beta_low=beta_low,
        beta_high=beta_high,
    )

    names = feature_names or DEFAULT_FEATURE_NAMES
    generator = torch.Generator().manual_seed(seed)
    n_features = len(names)
    resolved_attention_heads = attention_heads or _default_attention_heads(hidden_size)

    # 1. VaDeSC-EHR 风格的均匀 cluster prior 与 cluster-specific latent profile。
    cluster_prior = torch.ones(n_clusters, dtype=torch.float32) / n_clusters
    cluster_labels = torch.multinomial(
        cluster_prior,
        num_samples=n_patients,
        replacement=True,
        generator=generator,
    )
    cluster_means = _uniform((n_clusters, latent_dim), x_low, x_high, generator)
    cluster_covariances = torch.stack(
        [_random_spd_matrix(latent_dim, generator) for _cluster in range(n_clusters)]
    )

    latent_z = _sample_latent_profiles(
        cluster_labels=cluster_labels,
        cluster_means=cluster_means,
        cluster_covariances=cluster_covariances,
        generator=generator,
    )

    # 2. 生存变量模拟
    survival_coefficients = _uniform((n_clusters, latent_dim), beta_low, beta_high, generator)
    survival_intercepts = _uniform((n_clusters,), beta_low, beta_high, generator)
    event_times, events = _sample_survival_times(
        latent_z=latent_z,
        cluster_labels=cluster_labels,
        survival_coefficients=survival_coefficients,
        survival_intercepts=survival_intercepts,
        weibull_shape=weibull_shape,
        censoring_rate=censoring_rate,
        generator=generator,
    )

    # 3. 随机非线性 profile generator：对应原实现中 Z -> max_pos * hidden_size。
    hidden_profile = _random_nonlin_map(
        latent_z,
        hidden_size * max_visits,
        generator=generator,
    ).reshape(n_patients, max_visits, hidden_size)
    hidden_profile = _standardize_active_profile(hidden_profile)

    # 上面我们模拟得到的是patient-level hidden variable，但是每个patient都是一个
    # sequence，所以现在我们需要模拟每个patient的sequence length
    sequence_lengths = torch.randint(
        min_visits,
        max_visits + 1,
        size=(n_patients,),
        generator=generator,
    )
    visit_mask = _sequence_mask(sequence_lengths, max_visits)
    hidden_profile = hidden_profile * visit_mask.unsqueeze(-1)

    # 4. 多层 pseudo attention：保留原模拟器“随机注意力层 + 残差”的结构。
    for _layer in range(attention_layers):
        attention_output = _pseudo_attention(
            hidden_profile,
            visit_mask,
            hidden_size=hidden_size,
            attention_heads=resolved_attention_heads,
            generator=generator,
        )
        hidden_profile = (hidden_profile + attention_output) * visit_mask.unsqueeze(-1)

    # 5. 连续临床变量 decoder：替换 ICD softmax/argmax，输出多变量检查值。
    true_values = _decode_continuous_clinical_values(
        hidden_profile,
        n_features=n_features,
        generator=generator,
    )

    # 6. 非同步采样
    samples: list[ClinicalSample] = []
    for patient_index in range(n_patients):
        seq_len = int(sequence_lengths[patient_index])
        # 检查记录只能发生在患者实际观测终点之前；否则会出现事件/删失后仍有检查的矛盾。
        times = _sample_visit_times(seq_len, event_times[patient_index], generator)
        values = true_values[patient_index, :seq_len]
        severity = torch.sigmoid(latent_z[patient_index, 0] / max(abs(x_high), 1.0))
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
            "n_clusters": n_clusters,
            "min_visits": min_visits,
            "max_visits": max_visits,
            "latent_dim": latent_dim,
            "hidden_size": hidden_size,
            "attention_layers": attention_layers,
            "attention_heads": resolved_attention_heads,
            "censoring_rate": censoring_rate,
            "weibull_shape": weibull_shape,
            "x_low": x_low,
            "x_high": x_high,
            "beta_low": beta_low,
            "beta_high": beta_high,
            "seed": seed,
        },
        "cluster_prior": cluster_prior,
        "cluster_means": cluster_means,
        "cluster_covariances": cluster_covariances,
        "latent_z": latent_z,
        "survival_coefficients": survival_coefficients,
        "survival_intercepts": survival_intercepts,
        "sequence_lengths": sequence_lengths,
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


def _validate_generation_args(
    *,
    n_patients: int,
    n_clusters: int,
    min_visits: int,
    max_visits: int,
    latent_dim: int,
    hidden_size: int,
    attention_layers: int,
    attention_heads: int | None,
    censoring_rate: float,
    weibull_shape: float,
    x_low: float,
    x_high: float,
    beta_low: float,
    beta_high: float,
) -> None:
    if n_patients <= n_clusters:
        raise ValueError("n_patients must be greater than n_clusters.")
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2.")
    if min_visits < 2:
        raise ValueError("min_visits must be at least 2.")
    if max_visits < min_visits:
        raise ValueError("max_visits must be greater than or equal to min_visits.")
    if latent_dim <= 0 or hidden_size <= 0:
        raise ValueError("latent_dim and hidden_size must be positive.")
    if attention_layers < 0:
        raise ValueError("attention_layers must be non-negative.")
    if attention_heads is None:
        attention_heads = _default_attention_heads(hidden_size)
    if attention_heads <= 0 or hidden_size % attention_heads != 0:
        raise ValueError("attention_heads must divide hidden_size.")
    if not 0.0 <= censoring_rate < 1.0:
        raise ValueError("censoring_rate must be in [0, 1).")
    if weibull_shape <= 0:
        raise ValueError("weibull_shape must be positive.")
    if x_low >= x_high or beta_low >= beta_high:
        raise ValueError("low bounds must be smaller than high bounds.")


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

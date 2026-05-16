import math

import torch
from torch import Tensor


def _uniform(
    shape: tuple[int, ...],
    low: float,
    high: float,
    generator: torch.Generator,
) -> Tensor:
    return low + (high - low) * torch.rand(*shape, generator=generator)


def _random_spd_matrix(size: int, generator: torch.Generator) -> Tensor:
    matrix = torch.randn(size, size, generator=generator)
    return matrix @ matrix.T / size + 0.25 * torch.eye(size)


def _sample_latent_profiles(
    *,
    cluster_labels: Tensor,
    cluster_means: Tensor,
    cluster_covariances: Tensor,
    generator: torch.Generator,
) -> Tensor:
    n_patients = int(cluster_labels.shape[0])
    latent_dim = int(cluster_means.shape[1])

    # latent_z = torch.zeros(n_patients, latent_dim)
    # for cluster in range(int(cluster_means.shape[0])):
    #     cluster_index = cluster_labels == cluster
    #     n_cluster = int(cluster_index.sum())
    #     if n_cluster == 0:
    #         continue
    #     chol = torch.linalg.cholesky(cluster_covariances[cluster])
    #     eps = torch.randn(n_cluster, latent_dim, generator=generator)
    #     latent_z[cluster_index] = cluster_means[cluster] + eps @ chol.T

    # NOTE: 这样实现，更快一点
    eps = torch.randn(n_patients, latent_dim, generator=generator)
    chol = torch.linalg.cholesky(cluster_covariances[cluster_labels])
    latent_z = cluster_means[cluster_labels] + torch.einsum("bj,bij->bi", eps, chol)
    return latent_z


def _low_rank_matrix(n_in: int, n_out: int, generator: torch.Generator) -> Tensor:
    rank = max(1, min(n_in, n_out, 32))
    left = torch.randn(n_in, rank, generator=generator)
    right = torch.randn(rank, n_out, generator=generator)
    return (left @ right) / math.sqrt(float(rank * n_in))


def _random_nonlin_map(values: Tensor, n_out: int, generator: torch.Generator) -> Tensor:
    weight = _low_rank_matrix(int(values.shape[-1]), n_out, generator)
    return torch.relu(values @ weight)


def _standardize_active_profile(profile: Tensor) -> Tensor:
    mean = profile.mean(dim=(0, 1), keepdim=True)
    std = profile.std(dim=(0, 1), keepdim=True).clamp_min(1e-4)
    return (profile - mean) / std


def _sequence_mask(sequence_lengths: Tensor, max_length: int) -> Tensor:
    positions = torch.arange(max_length).unsqueeze(0)
    return positions < sequence_lengths.unsqueeze(1)

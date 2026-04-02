from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class PairSampleConfig:
    neighbors_per_point: int = 5
    seed: int = 14


@dataclass(frozen=True)
class TemporalSplitConfig:
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    min_test_points: int = 8
    min_val_points: int = 2


def sample_temporal_pairs(num_points: int, config: PairSampleConfig) -> list[tuple[int, int]]:
    rng = np.random.default_rng(config.seed)
    pairs: set[tuple[int, int]] = set()
    for idx in range(num_points):
        candidates = {max(0, idx - 1), min(num_points - 1, idx + 1)}
        while len(candidates) < config.neighbors_per_point:
            candidates.add(int(rng.integers(0, num_points)))
        for jdx in candidates:
            if idx == jdx:
                continue
            a, b = (idx, jdx) if idx < jdx else (jdx, idx)
            pairs.add((a, b))
    return sorted(pairs)


def pairwise_alignment_loss(
    latent_embeddings: torch.Tensor,
    topology_embeddings: torch.Tensor,
    pairs: list[tuple[int, int]],
) -> torch.Tensor:
    if len(pairs) == 0:
        return torch.tensor(0.0, device=latent_embeddings.device)
    loss_values = []
    for a, b in pairs:
        latent_distance = torch.norm(latent_embeddings[a] - latent_embeddings[b], p=2)
        topology_distance = torch.norm(topology_embeddings[a] - topology_embeddings[b], p=2)
        loss_values.append((latent_distance - topology_distance).pow(2))
    return torch.stack(loss_values).mean()


def pairwise_rkhs_alignment_loss(
    latent_embeddings: torch.Tensor,
    rkhs_features: torch.Tensor,
    pairs: list[tuple[int, int]],
) -> torch.Tensor:
    if len(pairs) == 0:
        return torch.tensor(0.0, device=latent_embeddings.device)
    loss_values = []
    for a, b in pairs:
        latent_distance = torch.norm(latent_embeddings[a] - latent_embeddings[b], p=2)
        rkhs_distance = torch.norm(rkhs_features[a] - rkhs_features[b], p=2)
        loss_values.append((latent_distance - rkhs_distance).pow(2))
    return torch.stack(loss_values).mean()


def pointwise_alignment_loss(
    latent_embeddings: torch.Tensor,
    topology_embeddings: torch.Tensor,
    indices: list[int] | None = None,
) -> torch.Tensor:
    if indices is None:
        latent = latent_embeddings
        topo = topology_embeddings
    else:
        if len(indices) == 0:
            return torch.tensor(0.0, device=latent_embeddings.device)
        index_tensor = torch.tensor(indices, dtype=torch.long, device=latent_embeddings.device)
        latent = latent_embeddings.index_select(0, index_tensor)
        topo = topology_embeddings.index_select(0, index_tensor)
    return torch.mean(torch.sum((latent - topo) ** 2, dim=1))


def chronological_split_indices(
    num_points: int,
    config: TemporalSplitConfig,
) -> tuple[list[int], list[int], list[int]]:
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    if not (0.0 < config.train_fraction < 1.0):
        raise ValueError("train_fraction must be in (0, 1)")
    if not (0.0 <= config.val_fraction < 1.0):
        raise ValueError("val_fraction must be in [0, 1)")
    if config.train_fraction + config.val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be < 1")

    # Explicit split behavior for tiny datasets.
    if num_points == 1:
        return [0], [], []
    if num_points == 2:
        return [0], [], [1]
    if num_points == 3:
        return [0], [1], [2]

    train_end = int(np.floor(num_points * config.train_fraction))
    val_end = int(np.floor(num_points * (config.train_fraction + config.val_fraction)))

    # Enforce minimum test/val support when dataset size allows it.
    if num_points >= config.min_test_points + config.min_val_points + 1:
        max_train_end = num_points - config.min_test_points - config.min_val_points
        train_end = min(train_end, max_train_end)
        max_val_end = num_points - config.min_test_points
        val_end = max(val_end, train_end + config.min_val_points)
        val_end = min(val_end, max_val_end)

    # Guarantee non-empty train/test and disjoint chronological blocks.
    train_end = max(1, min(train_end, num_points - 2))
    val_end = max(train_end, min(val_end, num_points - 1))

    train_ids = list(range(0, train_end))
    val_ids = list(range(train_end, val_end))
    test_ids = list(range(val_end, num_points))
    return train_ids, val_ids, test_ids


def rmse_on_indices(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    indices: list[int],
) -> float:
    if len(indices) == 0:
        raise ValueError("rmse_on_indices received empty index list")
    device = predictions.device
    idx = torch.tensor(indices, dtype=torch.long, device=device)
    diff = predictions.index_select(0, idx) - targets.index_select(0, idx)
    return float(torch.sqrt(torch.mean(diff * diff)).item())

from __future__ import annotations

import torch
from torch import nn


class PersLayModule(nn.Module):
    """Point-set PersLay encoder for one persistence diagram."""

    def __init__(self, latent_dim: int = 64) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )
        self.weight = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid(),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        # points shape: (n_points, 2)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"expected (n_points, 2), got {tuple(points.shape)}")
        if points.shape[0] == 0:
            return torch.zeros(self.phi[-2].out_features, device=points.device, dtype=points.dtype)
        phi = self.phi(points)
        weights = self.weight(points)
        return torch.sum(phi * weights, dim=0)

from __future__ import annotations

import torch
from torch import nn


class PersLayPointSetEncoder(nn.Module):
    """Classical PersLay point-set encoder.

    Input is a persistence diagram point set with shape (n_points, 2),
    where each point is (birth, death). The encoder computes:
        sum_{p in D} w(p) * phi(p)
    which is permutation invariant over diagram points.
    """

    def __init__(self, latent_dim: int = 64) -> None:
        super().__init__()
        self.weighting = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid(),
        )
        self.phi = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )

    def forward(self, diagram_points: torch.Tensor) -> torch.Tensor:
        if diagram_points.ndim != 2 or diagram_points.shape[1] != 2:
            raise ValueError(f"expected (n_points, 2), got {tuple(diagram_points.shape)}")
        if diagram_points.shape[0] == 0:
            return torch.zeros(self.phi[-2].out_features, device=diagram_points.device, dtype=diagram_points.dtype)
        points = diagram_points.to(dtype=torch.float32)
        weights = self.weighting(points)
        features = self.phi(points)
        return torch.sum(weights * features, dim=0)


class PersLayLikeEncoder(PersLayPointSetEncoder):
    """Backward-compatible alias. Uses true point-set PersLay semantics."""

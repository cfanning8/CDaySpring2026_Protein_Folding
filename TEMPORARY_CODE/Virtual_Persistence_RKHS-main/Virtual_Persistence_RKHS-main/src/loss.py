"""Topological RKHS loss function."""

from __future__ import annotations

import numpy as np
import torch

from src.kernels import HeatRandomFeatures


class TopologicalRKHSLoss:
    def __init__(
        self,
        grid_size: int = 50,
        n_components: int = 256,
        temperature: float = 0.2,
        lambda_weights: np.ndarray | None = None,
        random_state: int | None = None,
    ):
        self.input_dim = grid_size * grid_size
        self.rff = HeatRandomFeatures(
            input_dim=self.input_dim,
            n_components=n_components,
            temperature=temperature,
            lambda_weights=lambda_weights,
            random_state=random_state,
        )
        zero_vec = np.zeros(self.input_dim, dtype=np.float32)
        self._zero_embed = self.rff.transform(zero_vec.reshape(1, -1))[0]
        self._k_zero = float(np.sum(self._zero_embed ** 2))

    def __call__(self, vpd_diff: np.ndarray) -> float:
        if vpd_diff.shape != (self.input_dim,):
            raise ValueError(f"Expected shape ({self.input_dim},), got {vpd_diff.shape}")
        vpd_embed = self.rff.transform(vpd_diff.astype(np.float32).reshape(1, -1))[0]
        k_gamma_zero = float(np.dot(vpd_embed, self._zero_embed))
        loss_val = 2.0 * (self._k_zero - k_gamma_zero)
        return max(0.0, loss_val) if np.isfinite(loss_val) else 0.0


def topological_loss_batch_torch(vpd_diffs: torch.Tensor, loss_fn: TopologicalRKHSLoss) -> torch.Tensor:
    losses = [loss_fn(vpd_diffs[i].detach().cpu().numpy()) for i in range(vpd_diffs.shape[0])]
    return torch.tensor(np.mean(losses), dtype=torch.float32, device=vpd_diffs.device)
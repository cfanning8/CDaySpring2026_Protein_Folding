from __future__ import annotations

import numpy as np

from src.topology.kernels import HeatRandomFeatures


class TopologicalRKHSLoss:
    def __init__(
        self,
        grid_size: int = 50,
        n_components: int = 256,
        temperature: float = 0.2,
        lambda_weights: np.ndarray | None = None,
        random_state: int | None = None,
    ) -> None:
        self.input_dim = grid_size * grid_size
        self.rff = HeatRandomFeatures(
            input_dim=self.input_dim,
            n_components=n_components,
            temperature=temperature,
            lambda_weights=lambda_weights,
            random_state=random_state,
        )
        zero_vec = np.zeros(self.input_dim, dtype=np.float32)
        self._zero_embed = self.rff.transform(zero_vec)[0]
        self._k_zero = float(np.dot(self._zero_embed, self._zero_embed))

    def __call__(self, vpd_diff: np.ndarray) -> float:
        vector = np.asarray(vpd_diff, dtype=np.float32)
        if vector.shape != (self.input_dim,):
            raise ValueError(f"expected shape ({self.input_dim},), got {vector.shape}")
        vpd_embed = self.rff.transform(vector)[0]
        k_gamma_zero = float(np.dot(vpd_embed, self._zero_embed))
        loss_value = 2.0 * (self._k_zero - k_gamma_zero)
        if not np.isfinite(loss_value):
            return 0.0
        return max(0.0, loss_value)


def topological_loss_batch_numpy(vpd_diffs: np.ndarray, loss_fn: TopologicalRKHSLoss) -> float:
    batch = np.asarray(vpd_diffs)
    if batch.ndim != 2:
        raise ValueError("vpd_diffs must be shape (batch_size, input_dim)")
    losses = [loss_fn(batch[index]) for index in range(batch.shape[0])]
    return float(np.mean(losses)) if losses else 0.0

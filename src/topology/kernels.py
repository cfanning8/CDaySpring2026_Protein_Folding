from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def laplacian_symbol(theta: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    theta_values = np.asarray(theta, dtype=float)
    if weights is None:
        weights_array = np.ones(theta_values.shape[-1], dtype=float)
    else:
        weights_array = np.asarray(weights, dtype=float)
        if weights_array.shape[-1] != theta_values.shape[-1]:
            raise ValueError("weights dimensionality mismatch")
    return np.tensordot(1.0 - np.cos(theta_values), weights_array, axes=([-1], [0]))


def heat_multiplier(theta: np.ndarray, temperature: float, weights: np.ndarray | None = None) -> np.ndarray:
    if temperature <= 0:
        return np.ones(theta.shape[:-1] if theta.ndim > 0 else (), dtype=float)
    return np.exp(-temperature * laplacian_symbol(theta, weights=weights))


@dataclass
class HeatRandomFeatures:
    input_dim: int
    n_components: int = 256
    temperature: float = 0.2
    lambda_weights: np.ndarray | None = None
    random_state: int | None = None

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.random_state)
        self.lambda_weights_ = (
            np.asarray(self.lambda_weights, dtype=float)
            if self.lambda_weights is not None
            else np.ones(self.input_dim, dtype=float)
        )
        self.omega_ = rng.uniform(0.0, 2.0 * np.pi, size=(self.n_components, self.input_dim))
        self.bias_ = rng.uniform(0.0, 2.0 * np.pi, size=self.n_components)

        theta = np.mod(self.omega_, 2.0 * np.pi)
        lam = laplacian_symbol(theta, weights=self.lambda_weights_)
        effective_temp = self.temperature / max(self.input_dim, 1)
        kernel_weights = np.exp(-effective_temp * lam)
        self.scale_ = np.sqrt(np.maximum(kernel_weights, 1e-12) / self.n_components)

    def transform(self, x_values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(x_values, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.shape[1] != self.input_dim:
            raise ValueError(f"expected input_dim={self.input_dim}, got {matrix.shape[1]}")

        projections = np.mod(matrix @ self.omega_.T + self.bias_, 2.0 * np.pi)
        cos_proj = np.cos(projections)
        sin_proj = np.sin(projections)
        features = np.empty((matrix.shape[0], 2 * self.n_components), dtype=np.float32)
        features[:, 0::2] = self.scale_ * cos_proj
        features[:, 1::2] = self.scale_ * sin_proj
        return features

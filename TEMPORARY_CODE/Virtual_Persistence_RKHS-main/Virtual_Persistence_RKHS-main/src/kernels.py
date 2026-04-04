"""Kernel and random feature utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.utils import check_random_state


def laplacian_symbol(theta: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    theta = np.asarray(theta)
    if theta.shape[-1] == 2 and weights is None:
        theta1, theta2 = theta[..., 0], theta[..., 1]
        w1, w2, w3 = 0.75, 1.2, 0.95
        return 2.0 * (w1 * (1.0 - np.cos(theta1)) + w2 * (1.0 - np.cos(theta1 - theta2)) + w3 * (1.0 - np.cos(theta2)))
    
    if weights is None:
        raise ValueError("weights required for dimensions other than 2")
    weights = np.asarray(weights)
    if weights.shape[-1] != theta.shape[-1]:
        raise ValueError("weights dimensionality mismatch")
    return np.tensordot(1.0 - np.cos(theta), weights, axes=([-1], [0]))


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
        self.rng = check_random_state(self.random_state)
        self.lambda_weights_ = np.asarray(self.lambda_weights) if self.lambda_weights is not None else np.ones(self.input_dim)
        self.omega_ = self.rng.uniform(0.0, 2 * np.pi, size=(self.n_components, self.input_dim))
        self.bias_ = self.rng.uniform(0.0, 2 * np.pi, size=self.n_components)
        
        theta = np.mod(self.omega_, 2 * np.pi)
        lam = laplacian_symbol(theta, weights=self.lambda_weights_)
        effective_temp = self.temperature / self.input_dim if self.input_dim > 2 else self.temperature
        weights = np.exp(-effective_temp * lam)
        self.scale_ = np.sqrt(np.maximum(weights, 1e-12) / self.n_components)

    def transform(self, X: np.ndarray) -> np.ndarray:
        proj = np.mod(X @ self.omega_.T + self.bias_, 2 * np.pi)
        cos_proj = np.cos(proj)
        sin_proj = np.sin(proj)
        features = np.empty((X.shape[0], 2 * self.n_components), dtype=X.dtype)
        features[:, 0::2] = self.scale_ * cos_proj
        features[:, 1::2] = self.scale_ * sin_proj
        return features
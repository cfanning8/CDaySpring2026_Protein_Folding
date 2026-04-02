"""Heat kernel invariants via Monte Carlo integration."""

import numpy as np
from typing import Callable
from src.jump_kernel import LevyMeasure, lambda_symbol


def estimate_return_probability(
    t: float,
    levy_measure: LevyMeasure,
    N: int = 100000,
    seed: int = 14
) -> float:
    """Estimate p_t(0) = int exp(-t*lambda_H(theta)) dmu."""
    rng = np.random.default_rng(seed)
    m = levy_measure.m
    
    acc = 0.0
    for _ in range(N):
        theta = rng.uniform(-np.pi, np.pi, size=m)
        lam = lambda_symbol(theta, levy_measure)
        acc += np.exp(-t * lam)
    
    return acc / N


def estimate_collision_probability(
    t: float,
    levy_measure: LevyMeasure,
    N: int = 100000,
    seed: int = 14
) -> float:
    """Estimate sum_h p_t(h)^2 = int exp(-2*t*lambda_H(theta)) dmu."""
    rng = np.random.default_rng(seed)
    m = levy_measure.m
    
    acc = 0.0
    for _ in range(N):
        theta = rng.uniform(-np.pi, np.pi, size=m)
        lam = lambda_symbol(theta, levy_measure)
        acc += np.exp(-2.0 * t * lam)
    
    return acc / N


def estimate_energy_derivative(
    t: float,
    levy_measure: LevyMeasure,
    N: int = 50000,
    seed: int = 14
) -> float:
    """Estimate -d/dt p_t(0) = int lambda_H(theta) * exp(-t*lambda_H(theta)) dmu."""
    rng = np.random.default_rng(seed)
    m = levy_measure.m
    
    acc = 0.0
    for _ in range(N):
        theta = rng.uniform(-np.pi, np.pi, size=m)
        lam = lambda_symbol(theta, levy_measure)
        acc += lam * np.exp(-t * lam)
    
    return acc / N


def estimate_resolvent_diagonal(
    s: float,
    levy_measure: LevyMeasure,
    N: int = 50000,
    seed: int = 14
) -> float:
    """Estimate G_s(0,0) = int 1/(s + lambda_H(theta)) dmu."""
    rng = np.random.default_rng(seed)
    m = levy_measure.m
    
    acc = 0.0
    for _ in range(N):
        theta = rng.uniform(-np.pi, np.pi, size=m)
        lam = lambda_symbol(theta, levy_measure)
        acc += 1.0 / (s + lam)
    
    return acc / N


def estimate_kernel_section_energy(
    t: float,
    levy_measure: LevyMeasure,
    N: int = 50000,
    seed: int = 14
) -> float:
    """Estimate Dirichlet energy E_H^(2)(t) = int lambda_H(theta) * exp(-2*t*lambda_H(theta)) dmu."""
    rng = np.random.default_rng(seed)
    m = levy_measure.m
    
    acc = 0.0
    for _ in range(N):
        theta = rng.uniform(-np.pi, np.pi, size=m)
        lam = lambda_symbol(theta, levy_measure)
        acc += lam * np.exp(-2.0 * t * lam)
    
    return acc / N


def estimate_kernel_section_l2_squared(
    t: float,
    levy_measure: LevyMeasure,
    N: int = 50000,
    seed: int = 14
) -> float:
    """Estimate ||k_t(.,0)||_2^2 = int exp(-2*t*lambda_H(theta)) dmu."""
    rng = np.random.default_rng(seed)
    m = levy_measure.m
    
    acc = 0.0
    for _ in range(N):
        theta = rng.uniform(-np.pi, np.pi, size=m)
        lam = lambda_symbol(theta, levy_measure)
        acc += np.exp(-2.0 * t * lam)
    
    return acc / N


def estimate_Ht_norm_squared(
    f_values: np.ndarray,
    basis_points: list,
    levy_measure: LevyMeasure,
    t: float,
    N: int = 50000,
    seed: int = 14
) -> float:
    """Estimate RKHS norm |f|_{H_t}^2 = int |hat{f}(theta)|^2 * exp(t*lambda_H(theta)) dmu."""
    rng = np.random.default_rng(seed)
    m = levy_measure.m
    
    h_vectors = np.stack(basis_points, axis=0)
    
    acc = 0.0
    for _ in range(N):
        theta = rng.uniform(-np.pi, np.pi, size=m)
        
        phases = h_vectors @ theta
        f_hat = np.dot(f_values, np.exp(-1j * phases))
        
        lam = lambda_symbol(theta, levy_measure)
        acc += (np.abs(f_hat) ** 2) * np.exp(t * lam)
    
    return acc / N

"""Compound Poisson simulation for Levy process."""

import numpy as np
from typing import Tuple
from src.jump_kernel import LevyMeasure


def simulate_levy_process(
    levy_measure: LevyMeasure,
    t: float,
    rng: np.random.Generator
) -> np.ndarray:
    """Simulate X_t on H for finite-activity Levy measure."""
    if levy_measure.K == 0:
        return np.zeros(levy_measure.m, dtype=np.int64)
    
    total_rate = levy_measure.total_mass()
    if total_rate == 0:
        return np.zeros(levy_measure.m, dtype=np.int64)
    
    N_t = rng.poisson(total_rate * t)
    
    if N_t == 0:
        return np.zeros(levy_measure.m, dtype=np.int64)
    
    probs = levy_measure.jump_probabilities()
    jump_indices = rng.choice(levy_measure.K, size=N_t, p=probs)
    
    jumps = levy_measure.jumps[jump_indices]
    return jumps.sum(axis=0).astype(np.int64)


def sample_two_paths(
    levy_measure: LevyMeasure,
    t: float,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample two independent copies of the Levy process."""
    X_t = simulate_levy_process(levy_measure, t, rng)
    X_t_prime = simulate_levy_process(levy_measure, t, rng)
    return X_t, X_t_prime

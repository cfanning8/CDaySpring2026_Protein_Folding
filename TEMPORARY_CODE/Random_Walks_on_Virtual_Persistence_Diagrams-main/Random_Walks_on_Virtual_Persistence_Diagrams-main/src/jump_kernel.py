"""Levy measure on VPD group with truncation support."""

import numpy as np
from typing import Tuple, Callable, Optional


class LevyMeasure:
    """Levy measure on effective subgroup H."""
    
    def __init__(self, jumps: np.ndarray, rates: np.ndarray):
        if jumps.shape[0] != rates.shape[0]:
            raise ValueError("jumps and rates must have same length")
        if len(rates.shape) != 1:
            raise ValueError("rates must be 1D")
        if len(jumps.shape) != 2:
            raise ValueError("jumps must be 2D")
        
        self.jumps = jumps.astype(np.int64)
        self.rates = rates.astype(np.float64)
        self.m = jumps.shape[1]
        self.K = jumps.shape[0]
    
    def truncate_by_mass(self, mass_fn: Callable[[np.ndarray], float], R: float) -> 'LevyMeasure':
        """Truncate to jumps with mass <= R."""
        mask = np.array([mass_fn(kappa) <= R for kappa in self.jumps])
        return LevyMeasure(self.jumps[mask], self.rates[mask])
    
    def total_mass(self) -> float:
        """Total mass q = sum nu(kappa)."""
        return float(np.sum(self.rates))
    
    def jump_probabilities(self) -> np.ndarray:
        """Normalized jump probabilities."""
        q = self.total_mass()
        if q == 0:
            return np.zeros_like(self.rates)
        return self.rates / q


def lambda_symbol(theta: np.ndarray, levy_measure: LevyMeasure) -> float:
    """Levy-Khintchine exponent lambda_H(theta)."""
    if len(theta) != levy_measure.m:
        raise ValueError(f"theta dimension {len(theta)} != {levy_measure.m}")
    
    phases = levy_measure.jumps @ theta
    cos_vals = np.cos(phases)
    return float(np.sum(levy_measure.rates * (1.0 - cos_vals)))


def compute_d1_distance(bd1: Tuple[int, int], bd2: Tuple[int, int]) -> float:
    """1-strengthened metric d_1 between two BD points."""
    b1, d1 = bd1
    b2, d2 = bd2
    
    d_direct = abs(b1 - b2) + abs(d1 - d2)
    d_to_diag_1 = abs(b1 - d1) / 2.0
    d_to_diag_2 = abs(b2 - d2) / 2.0
    d_via_diagonal = d_to_diag_1 + d_to_diag_2
    
    return min(d_direct, d_via_diagonal)


def build_geometry_levy_measure(
    basis_bd_points: list,
    mass_fn: Callable[[np.ndarray], float],
    connectivity: str = "knn",
    k_neighbors: int = 3,
    distance_threshold: Optional[float] = None,
    lambda_val: float = 1.0
) -> LevyMeasure:
    """Build canonical geometry-induced Levy measure from VPD metric."""
    m = len(basis_bd_points)
    
    if m == 0:
        return LevyMeasure(np.zeros((0, 0), dtype=np.int64), np.array([], dtype=np.float64))
    
    dist_matrix = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i != j:
                dist_matrix[i, j] = compute_d1_distance(basis_bd_points[i], basis_bd_points[j])
    
    adjacency = np.zeros((m, m), dtype=bool)
    
    if connectivity == "knn":
        for i in range(m):
            distances = dist_matrix[i, :]
            distances[i] = np.inf
            nearest_indices = np.argsort(distances)[:k_neighbors]
            adjacency[i, nearest_indices] = True
            adjacency[nearest_indices, i] = True
    
    elif connectivity == "threshold":
        if distance_threshold is None:
            upper_tri = dist_matrix[np.triu_indices(m, k=1)]
            distance_threshold = np.median(upper_tri) if len(upper_tri) > 0 else 1.0
        
        for i in range(m):
            for j in range(m):
                if i != j and dist_matrix[i, j] <= distance_threshold:
                    adjacency[i, j] = True
    
    else:
        raise ValueError(f"Unknown connectivity: {connectivity}")
    
    edge_weights = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if adjacency[i, j]:
                edge_weights[i, j] = dist_matrix[i, j]
    
    degree = edge_weights.sum(axis=1)
    max_degree = np.max(degree) if np.max(degree) > 0 else 1.0
    avg_edge_weight = edge_weights.sum(axis=1) / (degree + 1e-10)
    
    jumps = []
    rates = []
    
    for i in range(m):
        e_i = np.zeros(m, dtype=np.int64)
        e_i[i] = 1
        
        if degree[i] > 0:
            rate = lambda_val * (degree[i] / max_degree) / (1.0 + avg_edge_weight[i])
        else:
            rate = lambda_val * 0.1
        
        jumps.append(e_i)
        rates.append(rate)
        
        e_i_neg = np.zeros(m, dtype=np.int64)
        e_i_neg[i] = -1
        jumps.append(e_i_neg)
        rates.append(rate)
    
    return LevyMeasure(np.array(jumps), np.array(rates))


def build_nearest_neighbor_levy_measure(m: int, lambda_val: float = 1.0) -> LevyMeasure:
    """Build nearest-neighbor Levy measure for testing."""
    jumps = []
    rates = []
    
    for i in range(m):
        e_i = np.zeros(m, dtype=np.int64)
        e_i[i] = 1
        jumps.append(e_i)
        rates.append(lambda_val)
        
        e_i_neg = np.zeros(m, dtype=np.int64)
        e_i_neg[i] = -1
        jumps.append(e_i_neg)
        rates.append(lambda_val)
    
    return LevyMeasure(np.array(jumps), np.array(rates))

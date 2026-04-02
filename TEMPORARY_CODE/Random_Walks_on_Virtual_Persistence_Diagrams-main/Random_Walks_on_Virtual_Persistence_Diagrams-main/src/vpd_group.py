"""VPD group construction: mass functional and d_1 metric."""

import numpy as np
from typing import List, Tuple, Dict, Callable, Optional


def compute_diagonal_distance(bd_point: Tuple[int, int]) -> float:
    """Distance from BD point to diagonal: |b - d| / 2."""
    b, d = bd_point
    return abs(b - d) / 2.0


def compute_mass(
    coeffs: np.ndarray,
    basis_bd_points: List[Tuple[int, int]],
    diag_distance_fn: Callable[[Tuple[int, int]], float] = None
) -> float:
    """Mass functional M(g) = sum_u |n_u| * d_1(u, [A])."""
    if diag_distance_fn is None:
        diag_distance_fn = compute_diagonal_distance
    
    if len(coeffs) != len(basis_bd_points):
        raise ValueError(f"coeffs length {len(coeffs)} != basis length {len(basis_bd_points)}")
    
    mass = 0.0
    for i, (n_u, bd_point) in enumerate(zip(coeffs, basis_bd_points)):
        d_to_diag = diag_distance_fn(bd_point)
        mass += abs(n_u) * d_to_diag
    
    return mass


def rho_upper_bound(
    coeffs: np.ndarray,
    basis_bd_points: List[Tuple[int, int]],
    diag_distance_fn: Callable[[Tuple[int, int]], float] = None
) -> float:
    """Upper bound rho(g, 0) <= M(g)."""
    return compute_mass(coeffs, basis_bd_points, diag_distance_fn)


def build_geometry_window(
    all_bd_points: List[Tuple[int, int]],
    max_birth: Optional[int] = None,
    max_death: Optional[int] = None
) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], int]]:
    """Build geometry window from BD points."""
    unique_points = sorted(set(all_bd_points))
    
    if max_birth is not None or max_death is not None:
        filtered = []
        for b, d in unique_points:
            if max_birth is not None and b > max_birth:
                continue
            if max_death is not None and d > max_death:
                continue
            filtered.append((b, d))
        unique_points = filtered
    
    index_of = {bd: i for i, bd in enumerate(unique_points)}
    return unique_points, index_of


def index_birth_death_pairs(
    alpha_A: List[Tuple[int, int]], 
    alpha_B: List[Tuple[int, int]]
) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], int]]:
    """Index distinct BD pairs from two diagrams. DEPRECATED: use build_geometry_window."""
    all_pairs = set(alpha_A) | set(alpha_B)
    S_points = sorted(list(all_pairs))
    index_of = {bd: i for i, bd in enumerate(S_points)}
    return S_points, index_of


def form_virtual_difference(
    alpha_A: List[Tuple[int, int]],
    alpha_B: List[Tuple[int, int]],
    index_of: Dict[Tuple[int, int], int]
) -> np.ndarray:
    """Form virtual difference beta = alpha^A - alpha^B."""
    m = len(index_of)
    beta = np.zeros(m, dtype=np.int64)
    
    for bd in alpha_A:
        if bd in index_of:
            beta[index_of[bd]] += 1
    
    for bd in alpha_B:
        if bd in index_of:
            beta[index_of[bd]] -= 1
    
    return beta


def compute_effective_subgroup(
    alpha_A: List[Tuple[int, int]],
    alpha_B: List[Tuple[int, int]]
) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], int], np.ndarray]:
    """Construct effective subgroup H. DEPRECATED: use build_geometry_window + form_virtual_difference."""
    S_points, index_of = index_birth_death_pairs(alpha_A, alpha_B)
    beta = form_virtual_difference(alpha_A, alpha_B, index_of)
    return S_points, index_of, beta

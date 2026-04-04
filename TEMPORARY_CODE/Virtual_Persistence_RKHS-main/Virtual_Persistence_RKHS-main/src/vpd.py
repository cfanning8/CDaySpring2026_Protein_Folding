"""Utilities for converting persistence diagrams to VPD vectors."""

from __future__ import annotations

import numpy as np


def persistence_diagram_to_vpd_vector(
    diagram: list[tuple[float, float]] | np.ndarray,
    grid_size: int = 50,
    birth_range: tuple[float, float] | None = None,
    death_range: tuple[float, float] | None = None,
) -> np.ndarray:
    diagram = np.asarray(diagram)
    if diagram.ndim == 1 and len(diagram) == 2:
        diagram = diagram.reshape(1, 2)
    if diagram.ndim != 2 or diagram.shape[1] != 2:
        raise ValueError(f"Diagram must be shape (n_points, 2), got {diagram.shape}")
    
    births, deaths = diagram[:, 0], diagram[:, 1]
    
    if birth_range is None:
        bmin, bmax = float(births.min()), float(births.max())
        birth_range = (bmin - 0.1, bmax + 0.1) if bmin == bmax else (bmin, bmax)
    
    if death_range is None:
        dmin, dmax = float(deaths.min()), float(deaths.max())
        death_range = (dmin - 0.1, dmax + 0.1) if dmin == dmax else (dmin, dmax)
    
    birth_bins = np.linspace(birth_range[0], birth_range[1], grid_size + 1)
    death_bins = np.linspace(death_range[0], death_range[1], grid_size + 1)
    
    vpd_vec = np.zeros(grid_size * grid_size, dtype=np.int32)
    for birth, death in diagram:
        birth_idx = np.clip(np.searchsorted(birth_bins, birth, side='right') - 1, 0, grid_size - 1)
        death_idx = np.clip(np.searchsorted(death_bins, death, side='right') - 1, 0, grid_size - 1)
        vpd_vec[death_idx * grid_size + birth_idx] += 1
    return vpd_vec


def gudhi_persistence_to_vpd_vector(
    persistence_pairs: list[tuple[int, tuple[float, float]]],
    grid_size: int = 50,
    birth_range: tuple[float, float] | None = None,
    death_range: tuple[float, float] | None = None,
    dimension: int | None = None,
) -> np.ndarray:
    diagram = []
    for pair in persistence_pairs:
        if len(pair) != 2:
            continue
        dim, (birth, death) = pair
        if dimension is not None and dim != dimension:
            continue
        if death == float('inf'):
            death = birth + 1.0
        diagram.append((float(birth), float(death)))
    
    if not diagram:
        return np.zeros(grid_size * grid_size, dtype=np.int32)
    
    return persistence_diagram_to_vpd_vector(diagram, grid_size=grid_size, birth_range=birth_range, death_range=death_range)
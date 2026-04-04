from __future__ import annotations

import numpy as np


def persistence_diagram_to_vpd_vector(
    diagram: list[tuple[float, float]] | np.ndarray,
    grid_size: int = 50,
    birth_range: tuple[float, float] | None = None,
    death_range: tuple[float, float] | None = None,
    *,
    require_ranges: bool = False,
) -> np.ndarray:
    points = np.asarray(diagram, dtype=float)
    if points.ndim == 1 and len(points) == 2:
        points = points.reshape(1, 2)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"diagram must be shape (n_points, 2), got {points.shape}")

    births, deaths = points[:, 0], points[:, 1]
    if require_ranges and (birth_range is None or death_range is None):
        raise ValueError("birth_range and death_range must be provided when require_ranges=True")
    if birth_range is None:
        birth_range = _safe_range(births)
    if death_range is None:
        death_range = _safe_range(deaths)

    birth_bins = np.linspace(birth_range[0], birth_range[1], grid_size + 1)
    death_bins = np.linspace(death_range[0], death_range[1], grid_size + 1)

    vector = np.zeros(grid_size * grid_size, dtype=np.int32)
    for birth, death in points:
        birth_index = np.clip(np.searchsorted(birth_bins, birth, side="right") - 1, 0, grid_size - 1)
        death_index = np.clip(np.searchsorted(death_bins, death, side="right") - 1, 0, grid_size - 1)
        vector[death_index * grid_size + birth_index] += 1
    return vector


def gudhi_persistence_to_vpd_vector(
    persistence_pairs: list[tuple[int, tuple[float, float]]],
    grid_size: int = 50,
    birth_range: tuple[float, float] | None = None,
    death_range: tuple[float, float] | None = None,
    dimension: int | None = None,
    *,
    require_ranges: bool = False,
) -> np.ndarray:
    points: list[tuple[float, float]] = []
    for pair in persistence_pairs:
        if len(pair) != 2:
            continue
        dim, interval = pair
        if len(interval) != 2:
            continue
        if dimension is not None and dim != dimension:
            continue
        birth, death = float(interval[0]), float(interval[1])
        if death == float("inf"):
            death = birth + 1.0
        points.append((birth, death))

    if not points:
        return np.zeros(grid_size * grid_size, dtype=np.int32)

    return persistence_diagram_to_vpd_vector(
        points,
        grid_size=grid_size,
        birth_range=birth_range,
        death_range=death_range,
        require_ranges=require_ranges,
    )


def virtual_difference_vector(vpd_next: np.ndarray, vpd_current: np.ndarray) -> np.ndarray:
    if vpd_next.shape != vpd_current.shape:
        raise ValueError("virtual difference requires equal-sized vectors")
    return vpd_next.astype(np.int64) - vpd_current.astype(np.int64)


def _safe_range(values: np.ndarray) -> tuple[float, float]:
    lower = float(values.min())
    upper = float(values.max())
    if lower == upper:
        delta = 0.1 if lower == 0 else abs(lower) * 0.1
        return lower - delta, upper + delta
    return lower, upper

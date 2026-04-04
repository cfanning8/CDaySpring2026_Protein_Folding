from __future__ import annotations

import numpy as np
import pandas as pd


def topology_residue_graph_edges(
    coords: np.ndarray,
    radius_max_a: float,
    *,
    backbone_filtration: float = 0.0,
) -> pd.DataFrame:
    """
    Primary topology graph: mandatory backbone edges (filtration = backbone_filtration,
    default 0) plus spatial contacts for |i-j|>1 with Euclidean distance <= radius_max_a
    (filtration value = distance).
    """
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (n, 3)")
    if coords.shape[0] == 0:
        raise ValueError("coords is empty")
    if radius_max_a <= 0:
        raise ValueError("radius_max_a must be positive")
    bf = float(backbone_filtration)
    if bf < 0 or not np.isfinite(bf):
        raise ValueError("backbone_filtration must be finite and non-negative")

    n = int(coords.shape[0])
    rows: list[tuple[int, int, float]] = []

    for i in range(n - 1):
        rows.append((i, i + 1, bf))

    for i in range(n):
        for j in range(i + 2, n):
            delta = coords[i] - coords[j]
            dist = float(np.linalg.norm(delta))
            if dist <= radius_max_a:
                rows.append((i, j, dist))

    return pd.DataFrame(rows, columns=["source", "target", "filtration"])


def residue_contact_edges(coords: np.ndarray, radius_max_a: float) -> pd.DataFrame:
    """
    Legacy C_alpha path: backbone edges weighted by Euclidean distance (not filtration-0).
    Prefer topology_residue_graph_edges for the frozen NeRIPS topology graph.
    """
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (n, 3)")
    if coords.shape[0] == 0:
        raise ValueError("coords is empty")
    if radius_max_a <= 0:
        raise ValueError("radius_max_a must be positive")

    n = int(coords.shape[0])
    rows: list[tuple[int, int, float]] = []

    for i in range(n - 1):
        delta = coords[i] - coords[i + 1]
        dist = float(np.linalg.norm(delta))
        rows.append((i, i + 1, dist))

    for i in range(n):
        for j in range(i + 2, n):
            delta = coords[i] - coords[j]
            dist = float(np.linalg.norm(delta))
            if dist <= radius_max_a:
                rows.append((i, j, dist))

    return pd.DataFrame(rows, columns=["source", "target", "distance"])

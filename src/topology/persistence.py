from __future__ import annotations

import math

import gudhi as gd
import numpy as np
import pandas as pd


def weighted_clique_persistence_pairs(
    edges: pd.DataFrame,
    max_dimension: int = 1,
) -> list[tuple[int, tuple[float, float]]]:
    required = {"source", "target", "duration_seconds"}
    missing = required.difference(set(edges.columns))
    if missing:
        raise ValueError(f"edges missing required columns: {sorted(missing)}")
    if edges.empty:
        return []

    nodes = sorted(set(edges["source"]).union(set(edges["target"])))
    node_index = {node: idx for idx, node in enumerate(nodes)}

    max_weight = float(pd.to_numeric(edges["duration_seconds"], errors="coerce").max())
    if not math.isfinite(max_weight) or max_weight <= 0:
        raise ValueError("duration_seconds must have positive finite values")

    simplex = gd.SimplexTree()
    for node in nodes:
        simplex.insert([node_index[node]], filtration=0.0)

    for row in edges.itertuples(index=False):
        weight = float(row.duration_seconds)
        if weight <= 0:
            continue
        filtration = 1.0 - (weight / max_weight)
        simplex.insert([node_index[str(row.source)], node_index[str(row.target)]], filtration=filtration)

    simplex.expansion(max_dimension + 1)
    simplex.initialize_filtration()
    return simplex.persistence()


def persistence_pairs_for_dimension(
    pairs: list[tuple[int, tuple[float, float]]],
    homology_dimension: int,
) -> list[tuple[int, tuple[float, float]]]:
    filtered = []
    for dim, interval in pairs:
        if dim != homology_dimension:
            continue
        birth, death = float(interval[0]), float(interval[1])
        if death == float("inf"):
            death = birth + 1.0
        filtered.append((dim, (birth, death)))
    return filtered

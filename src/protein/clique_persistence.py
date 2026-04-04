from __future__ import annotations

import numpy as np
import pandas as pd
import gudhi as gd


def clique_persistence_from_distance_edges(
    edges: pd.DataFrame,
    max_dimension: int = 1,
) -> list[tuple[int, tuple[float, float]]]:
    """
    Clique filtration from edge filtrations. Edge column must be 'filtration' (non-negative
    finite; 0 allowed for backbone edges). Legacy tables with column 'distance' are accepted.
    """
    if "filtration" in edges.columns:
        weight_col = "filtration"
    elif "distance" in edges.columns:
        weight_col = "distance"
    else:
        raise ValueError("edges must contain column 'filtration' or 'distance'")

    required = {"source", "target", weight_col}
    missing = required.difference(set(edges.columns))
    if missing:
        raise ValueError(f"edges missing required columns: {sorted(missing)}")
    if edges.empty:
        return []

    sources = pd.to_numeric(edges["source"], errors="raise").astype(int)
    targets = pd.to_numeric(edges["target"], errors="raise").astype(int)
    weights = pd.to_numeric(edges[weight_col], errors="coerce")
    if weights.isna().any():
        raise ValueError(f"{weight_col} contains non-numeric values")

    nodes = sorted(set(sources.tolist()).union(set(targets.tolist())))
    for node in nodes:
        if node < 0:
            raise ValueError("negative vertex index")

    tree = gd.SimplexTree()
    for node in nodes:
        tree.insert([node], filtration=0.0)

    for s, t, w in zip(sources.tolist(), targets.tolist(), weights.tolist(), strict=True):
        val = float(w)
        if val < 0 or not np.isfinite(val):
            raise ValueError("edge filtration must be finite and non-negative")
        if s == t:
            continue
        a, b = (s, t) if s < t else (t, s)
        tree.insert([a, b], filtration=val)

    tree.expansion(max_dimension + 1)
    tree.initialize_filtration()
    return tree.persistence()

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.topology import (  # noqa: E402
    HeatRandomFeatures,
    gudhi_persistence_to_vpd_vector,
    persistence_pairs_for_dimension,
    virtual_difference_vector,
    weighted_clique_persistence_pairs,
)
from src.window_cache import aggregate_window_edges, load_cached_windows  # noqa: E402


def main() -> None:
    args = parse_args()
    cache_dir = PROJECT_ROOT / "data" / "preprocessed" / args.dataset
    windows_path = cache_dir / "windows.npz"
    output_npz = cache_dir / "persistence_features.npz"
    output_graph = cache_dir / "graph_features.csv"
    output_raw_diag = cache_dir / "raw_persistence_diagrams.jsonl"
    if not windows_path.exists():
        raise ValueError(f"missing windows cache: {windows_path}")
    if output_npz.exists() and output_graph.exists() and output_raw_diag.exists() and not args.force:
        print(f"[cache] persistence already cached for {args.dataset}")
        return

    windows = load_cached_windows(cache_dir)
    if len(windows) < 2:
        raise ValueError(f"need at least 2 windows for persistence drift cache, got {len(windows)}")

    # First pass: compute persistence pairs and gather global ranges per homology dimension.
    window_h0_pairs: list[list[tuple[int, tuple[float, float]]]] = []
    window_h1_pairs: list[list[tuple[int, tuple[float, float]]]] = []
    h0_points: list[tuple[float, float]] = []
    h1_points: list[tuple[float, float]] = []
    graph_rows: list[dict[str, float | int]] = []
    raw_rows: list[dict[str, object]] = []
    for window_id, window in enumerate(tqdm(windows, desc=f"{args.dataset}:persistence", unit="window")):
        edges = aggregate_window_edges(window)
        pairs = weighted_clique_persistence_pairs(edges, max_dimension=1)
        h0_pairs = persistence_pairs_for_dimension(pairs, homology_dimension=0)
        h1_pairs = persistence_pairs_for_dimension(pairs, homology_dimension=1)
        window_h0_pairs.append(h0_pairs)
        window_h1_pairs.append(h1_pairs)
        h0_points.extend(_pairs_to_points(h0_pairs))
        h1_points.extend(_pairs_to_points(h1_pairs))
        raw_rows.append(
            {
                "window_id": int(window_id),
                "h0": [[float(b), float(d)] for b, d in _pairs_to_points(h0_pairs)],
                "h1": [[float(b), float(d)] for b, d in _pairs_to_points(h1_pairs)],
            }
        )
        graph_rows.append(
            {
                "window_id": window_id,
                "nodes": int(len(set(edges["source"]).union(set(edges["target"])))),
                "edges": int(len(edges)),
                "total_duration_seconds": float(edges["duration_seconds"].sum()),
                "mean_edge_duration_seconds": float(edges["duration_seconds"].mean()),
                "window_start": float(window["t_start"].min()),
                "window_end": float(window["t_start"].max()),
            }
        )

    birth_range_h0, death_range_h0 = _ranges_from_points(h0_points)
    birth_range_h1, death_range_h1 = _ranges_from_points(h1_points)

    # Second pass: vectorize every diagram on a common basis.
    vectors: list[np.ndarray] = []
    for h0_pairs, h1_pairs in zip(window_h0_pairs, window_h1_pairs):
        v0 = gudhi_persistence_to_vpd_vector(
            h0_pairs,
            grid_size=args.grid_size,
            birth_range=birth_range_h0,
            death_range=death_range_h0,
            dimension=0,
            require_ranges=True,
        )
        v1 = gudhi_persistence_to_vpd_vector(
            h1_pairs,
            grid_size=args.grid_size,
            birth_range=birth_range_h1,
            death_range=death_range_h1,
            dimension=1,
            require_ranges=True,
        )
        vectors.append(np.concatenate([v0, v1]).astype(np.int64))

    d_matrix = np.vstack(vectors)
    g_matrix = np.vstack([virtual_difference_vector(d_matrix[idx + 1], d_matrix[idx]) for idx in range(len(windows) - 1)])
    rff = HeatRandomFeatures(
        input_dim=g_matrix.shape[1],
        n_components=args.rff_components,
        temperature=args.rff_temperature,
        random_state=args.seed,
    )
    rkhs = rff.transform(g_matrix)

    np.savez_compressed(
        output_npz,
        d_t=d_matrix,
        g_t=g_matrix,
        rkhs_g_t=rkhs,
        birth_range_h0=np.asarray(birth_range_h0, dtype=np.float64),
        death_range_h0=np.asarray(death_range_h0, dtype=np.float64),
        birth_range_h1=np.asarray(birth_range_h1, dtype=np.float64),
        death_range_h1=np.asarray(death_range_h1, dtype=np.float64),
    )
    pd.DataFrame(graph_rows).to_csv(output_graph, index=False)
    with output_raw_diag.open("w", encoding="utf-8") as handle:
        for row in raw_rows:
            handle.write(json.dumps(row) + "\n")
    print(f"[ok] cached persistence features: {output_npz}")
    print(f"[ok] cached graph features: {output_graph}")
    print(f"[ok] cached raw diagrams: {output_raw_diag}")


def _pairs_to_points(pairs: list[tuple[int, tuple[float, float]]]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for _, interval in pairs:
        if len(interval) != 2:
            continue
        birth = float(interval[0])
        death = float(interval[1])
        if not np.isfinite(birth):
            continue
        if not np.isfinite(death):
            death = birth + 1.0
        points.append((birth, death))
    return points


def _ranges_from_points(points: list[tuple[float, float]]) -> tuple[tuple[float, float], tuple[float, float]]:
    if not points:
        return (0.0, 1.0), (0.0, 1.0)
    births = np.asarray([b for b, _ in points], dtype=np.float64)
    deaths = np.asarray([d for _, d in points], dtype=np.float64)
    return _safe_range(births), _safe_range(deaths)


def _safe_range(values: np.ndarray) -> tuple[float, float]:
    lower = float(np.min(values))
    upper = float(np.max(values))
    if lower == upper:
        delta = 0.1 if lower == 0.0 else abs(lower) * 0.1
        return lower - delta, upper + delta
    return lower, upper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache persistence features for one dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--rff-components", type=int, default=128)
    parser.add_argument("--rff-temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

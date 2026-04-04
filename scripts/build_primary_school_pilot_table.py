from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataloaders import load_all_datasets  # noqa: E402
from src.edge_preparation import extract_temporal_events_for_dataset  # noqa: E402
from src.episim import SIRSimulationConfig, estimate_large_outbreak_probability  # noqa: E402
from src.window_cache import aggregate_window_edges as aggregate_edges_from_events  # noqa: E402
from src.window_cache import load_cached_windows as load_cached_windows_from_npz  # noqa: E402
from src.topology import (  # noqa: E402
    HeatRandomFeatures,
    gudhi_persistence_to_vpd_vector,
    persistence_pairs_for_dimension,
    virtual_difference_vector,
    weighted_clique_persistence_pairs,
)

DEFAULT_DATASET_KEY = r"primary_school\primaryschool.csv\primaryschool.csv"


def main() -> None:
    args = parse_args()
    if args.preprocessed_cache_dir is not None:
        windows = load_cached_windows_from_npz(args.preprocessed_cache_dir)
    else:
        datasets = load_all_datasets(PROJECT_ROOT / "data")
        if args.dataset_key not in datasets:
            raise ValueError(f"dataset_key not found: {args.dataset_key}")

        temporal_result = extract_temporal_events_for_dataset(args.dataset_key, datasets[args.dataset_key])
        if temporal_result is None:
            raise ValueError(f"no temporal extraction rule for dataset: {args.dataset_key}")

        windows = build_windows(
            temporal_result.events,
            window_seconds=args.window_seconds,
            stride_seconds=args.stride_seconds,
            min_events=args.min_events_per_window,
        )
    if len(windows) < 2:
        raise ValueError("need at least 2 windows to compute topology drift g_t")

    if args.persistence_cache_path is not None and args.sir_cache_path is not None and args.graph_cache_csv is not None:
        persistence_arrays = np.load(args.persistence_cache_path)
        d_matrix = persistence_arrays["d_t"]
        g_matrix = persistence_arrays["g_t"]
        rkhs_features = persistence_arrays["rkhs_g_t"]
        sir_table = pd.read_csv(args.sir_cache_path).sort_values("window_id")
        graph_table = pd.read_csv(args.graph_cache_csv).sort_values("window_id")
        labels = sir_table["y_large_outbreak_prob"].to_list()
        mean_attacks = sir_table["mean_attack_rate"].to_list()
        mean_peaks = sir_table["mean_peak_prevalence"].to_list()
        graph_features = graph_table.to_dict("records")
        if len(labels) != len(windows):
            raise ValueError("sir cache size does not match number of windows")
        if len(graph_features) != len(windows):
            raise ValueError("graph cache size does not match number of windows")
        if d_matrix.shape[0] != len(windows):
            raise ValueError("persistence d_t rows do not match number of windows")
        if g_matrix.shape[0] != len(windows) - 1:
            raise ValueError("persistence g_t rows do not match number of transitions")
    else:
        per_window_pairs: list[tuple[list[tuple[int, tuple[float, float]]], list[tuple[int, tuple[float, float]]]]] = []
        h0_points: list[tuple[float, float]] = []
        h1_points: list[tuple[float, float]] = []
        labels = []
        mean_attacks = []
        mean_peaks = []
        graph_features = []
        for window_id, window_events in enumerate(tqdm(windows, desc="build_windows", unit="window")):
            canonical_edges = aggregate_edges_from_events(window_events)
            persistence_pairs = weighted_clique_persistence_pairs(canonical_edges, max_dimension=1)
            h0_pairs = persistence_pairs_for_dimension(persistence_pairs, homology_dimension=0)
            h1_pairs = persistence_pairs_for_dimension(persistence_pairs, homology_dimension=1)
            per_window_pairs.append((h0_pairs, h1_pairs))
            h0_points.extend(normalize_points(h0_pairs))
            h1_points.extend(normalize_points(h1_pairs))

            simulation_config = SIRSimulationConfig(
                beta_per_second=args.beta_per_second,
                gamma_per_second=args.gamma_per_second,
                tau=args.tau,
                num_simulations=args.num_simulations,
                horizon_seconds=args.horizon_seconds,
                seed=args.seed + window_id,
            )
            large_prob, mean_attack, mean_peak = estimate_large_outbreak_probability(window_events, simulation_config)
            labels.append(large_prob)
            mean_attacks.append(mean_attack)
            mean_peaks.append(mean_peak)

            graph_features.append(
                {
                    "nodes": int(len(set(canonical_edges["source"]).union(set(canonical_edges["target"])))),
                    "edges": int(len(canonical_edges)),
                    "total_duration_seconds": float(canonical_edges["duration_seconds"].sum()),
                    "mean_edge_duration_seconds": float(canonical_edges["duration_seconds"].mean()),
                }
            )
        birth_range_h0, death_range_h0 = ranges_from_points(h0_points)
        birth_range_h1, death_range_h1 = ranges_from_points(h1_points)
        diagram_vectors = []
        for h0_pairs, h1_pairs in per_window_pairs:
            vector_h0 = gudhi_persistence_to_vpd_vector(
                h0_pairs,
                grid_size=args.grid_size,
                birth_range=birth_range_h0,
                death_range=death_range_h0,
                dimension=0,
                require_ranges=True,
            )
            vector_h1 = gudhi_persistence_to_vpd_vector(
                h1_pairs,
                grid_size=args.grid_size,
                birth_range=birth_range_h1,
                death_range=death_range_h1,
                dimension=1,
                require_ranges=True,
            )
            diagram_vectors.append(np.concatenate([vector_h0, vector_h1]).astype(np.int64))
        assert len(diagram_vectors) == len(labels), "diagram/label count mismatch"

        d_matrix = np.vstack(diagram_vectors)
        g_matrix = np.vstack([virtual_difference_vector(d_matrix[t + 1], d_matrix[t]) for t in range(len(windows) - 1)])

        rff = HeatRandomFeatures(
            input_dim=g_matrix.shape[1],
            n_components=args.rff_components,
            temperature=args.rff_temperature,
            random_state=args.seed,
        )
        rkhs_features = rff.transform(g_matrix)

    table_rows = []
    for t in range(len(windows) - 1):
        window_start = float(windows[t]["t_start"].min())
        window_end = float(windows[t]["t_start"].max())
        g_norm = float(np.linalg.norm(g_matrix[t], ord=2))
        d_norm = float(np.linalg.norm(d_matrix[t], ord=2))
        table_rows.append(
            {
                "t": t,
                "window_start": window_start,
                "window_end": window_end,
                "day_index": int((window_start - float(windows[0]["t_start"].min())) // 86400),
                "context_label": derive_context_label(window_start),
                "y_large_outbreak_prob": float(labels[t]),
                "y_next_large_outbreak_prob": float(labels[t + 1]),
                "delta_y_large_outbreak_prob": float(labels[t + 1] - labels[t]),
                "mean_attack_rate": float(mean_attacks[t]),
                "mean_peak_prevalence": float(mean_peaks[t]),
                "d_l2_norm": d_norm,
                "g_l2_norm": g_norm,
                "nodes": graph_features[t]["nodes"],
                "edges": graph_features[t]["edges"],
                "total_duration_seconds": graph_features[t]["total_duration_seconds"],
                "mean_edge_duration_seconds": graph_features[t]["mean_edge_duration_seconds"],
            }
        )

    output_dir = PROJECT_ROOT / "temp" / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(table_rows)
    table_path = output_dir / "table.csv"
    table.to_csv(table_path, index=False)

    npz_path = output_dir / "features.npz"
    np.savez_compressed(
        npz_path,
        d_t=d_matrix[:-1],
        g_t=g_matrix,
        rkhs_g_t=rkhs_features,
        y_t=np.asarray(labels[:-1], dtype=np.float32),
    )

    print(f"Windows used: {len(windows)}")
    print(f"Training rows (t -> t+1): {len(table_rows)}")
    print(f"Saved table: {table_path}")
    print(f"Saved features: {npz_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build primary_school pilot training table for claim-B pipeline.")
    parser.add_argument("--dataset-key", default=DEFAULT_DATASET_KEY)
    parser.add_argument("--window-seconds", type=float, default=2700.0)
    parser.add_argument("--stride-seconds", type=float, default=1350.0)
    parser.add_argument("--min-events-per-window", type=int, default=100)
    parser.add_argument("--tau", type=float, default=0.20)
    parser.add_argument("--num-simulations", type=int, default=200)
    parser.add_argument("--beta-per-second", type=float, required=True)
    parser.add_argument("--gamma-per-second", type=float, required=True)
    parser.add_argument("--horizon-seconds", type=float, required=True)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--rff-components", type=int, default=128)
    parser.add_argument("--rff-temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--output-subdir", type=str, default="pilot_primary_school")
    parser.add_argument("--preprocessed-cache-dir", type=Path, default=None)
    parser.add_argument("--persistence-cache-path", type=Path, default=None)
    parser.add_argument("--graph-cache-csv", type=Path, default=None)
    parser.add_argument("--sir-cache-path", type=Path, default=None)
    return parser.parse_args()


def build_windows(
    events: pd.DataFrame,
    window_seconds: float,
    stride_seconds: float,
    min_events: int,
) -> list[pd.DataFrame]:
    t_min = float(events["t_start"].min())
    t_max = float(events["t_start"].max())
    windows = []
    cursor = t_min
    while cursor + window_seconds <= t_max + 1e-9:
        mask = (events["t_start"] >= cursor) & (events["t_start"] < cursor + window_seconds)
        window_events = events.loc[mask, ["source", "target", "t_start", "duration_seconds"]].copy()
        if len(window_events) >= min_events:
            windows.append(window_events)
        cursor += stride_seconds
    return windows


def derive_context_label(window_start_seconds: float) -> str:
    local_seconds = window_start_seconds % 86400.0
    hour = local_seconds / 3600.0
    if 8.0 <= hour < 11.0:
        return "morning_class"
    if 11.0 <= hour < 13.0:
        return "lunch_or_transition"
    if 13.0 <= hour < 16.5:
        return "afternoon_class"
    return "off_schedule"


def ranges_from_points(points: list[tuple[float, float]]) -> tuple[tuple[float, float], tuple[float, float]]:
    if not points:
        return (0.0, 1.0), (0.0, 1.0)
    births = np.asarray([b for b, _ in points], dtype=np.float64)
    deaths = np.asarray([d for _, d in points], dtype=np.float64)
    return safe_range(births), safe_range(deaths)


def safe_range(values: np.ndarray) -> tuple[float, float]:
    low = float(np.min(values))
    high = float(np.max(values))
    if low == high:
        delta = 0.1 if low == 0.0 else abs(low) * 0.1
        return low - delta, high + delta
    return low, high


def normalize_points(pairs: list[tuple[int, tuple[float, float]]]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for _, (birth, death) in pairs:
        b = float(birth)
        d = float(death)
        if not np.isfinite(b):
            continue
        if not np.isfinite(d):
            d = b + 1.0
        points.append((b, d))
    return points


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataloaders import load_all_datasets  # noqa: E402
from src.edge_preparation import extract_temporal_events_for_dataset  # noqa: E402
from src.episim import SIRSimulationConfig, estimate_large_outbreak_probability  # noqa: E402

DEFAULT_DATASET_KEY = r"primary_school\primaryschool.csv\primaryschool.csv"


def main() -> None:
    args = parse_args()
    datasets = load_all_datasets(PROJECT_ROOT / "data")
    if args.dataset_key not in datasets:
        raise ValueError(f"dataset_key not found: {args.dataset_key}")
    temporal = extract_temporal_events_for_dataset(args.dataset_key, datasets[args.dataset_key])
    if temporal is None:
        raise ValueError("dataset is not temporal under current extraction rules")

    windows = build_windows(
        temporal.events,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        min_events=args.min_events_per_window,
    )
    if not windows:
        raise ValueError("no windows were produced")

    beta_values = [float(value) for value in args.beta_grid.split(",")]
    gamma_values = [float(value) for value in args.gamma_grid.split(",")]

    rows: list[dict[str, float]] = []
    for beta in beta_values:
        for gamma in gamma_values:
            labels = []
            for index, window in enumerate(windows):
                config = SIRSimulationConfig(
                    beta_per_second=beta,
                    gamma_per_second=gamma,
                    tau=args.tau,
                    num_simulations=args.num_simulations,
                    horizon_seconds=args.horizon_seconds,
                    seed=args.seed + index,
                )
                y_value, _, _ = estimate_large_outbreak_probability(window, config)
                labels.append(y_value)

            y_array = np.asarray(labels, dtype=float)
            rows.append(
                {
                    "beta_per_second": beta,
                    "gamma_per_second": gamma,
                    "median_y": float(np.median(y_array)),
                    "std_y": float(np.std(y_array)),
                    "prop_low_y": float(np.mean(y_array < 0.10)),
                    "prop_high_y": float(np.mean(y_array > 0.50)),
                    "mean_y": float(np.mean(y_array)),
                    "min_y": float(np.min(y_array)),
                    "max_y": float(np.max(y_array)),
                }
            )

    output_dir = PROJECT_ROOT / "temp" / "pilot_primary_school"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "calibration_grid.csv"
    result = pd.DataFrame(rows).sort_values(["beta_per_second", "gamma_per_second"])
    result.to_csv(output_path, index=False)

    print(f"Windows used: {len(windows)}")
    print(f"Saved calibration grid: {output_path}")


def build_windows(
    events: pd.DataFrame,
    window_seconds: float,
    stride_seconds: float,
    min_events: int,
) -> list[pd.DataFrame]:
    t_min = float(events["t_start"].min())
    t_max = float(events["t_start"].max())
    windows: list[pd.DataFrame] = []
    cursor = t_min
    while cursor + window_seconds <= t_max + 1e-9:
        mask = (events["t_start"] >= cursor) & (events["t_start"] < cursor + window_seconds)
        window_events = events.loc[mask, ["source", "target", "t_start", "duration_seconds"]].copy()
        if len(window_events) >= min_events:
            windows.append(window_events)
        cursor += stride_seconds
    return windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate SIR parameters for non-degenerate Y_t labels.")
    parser.add_argument("--dataset-key", default=DEFAULT_DATASET_KEY)
    parser.add_argument("--window-seconds", type=float, default=2700.0)
    parser.add_argument("--stride-seconds", type=float, default=1350.0)
    parser.add_argument("--min-events-per-window", type=int, default=100)
    parser.add_argument("--tau", type=float, default=0.20)
    parser.add_argument("--num-simulations", type=int, default=80)
    parser.add_argument("--horizon-seconds", type=float, default=86400.0)
    parser.add_argument("--beta-grid", default="0.001,0.003,0.005,0.01")
    parser.add_argument("--gamma-grid", default="0.000011574,0.000023148,0.000046296")
    parser.add_argument("--seed", type=int, default=14)
    return parser.parse_args()


if __name__ == "__main__":
    main()

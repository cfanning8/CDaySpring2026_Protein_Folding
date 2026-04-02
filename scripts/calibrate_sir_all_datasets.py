from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXE = str(PROJECT_ROOT / ".venv" / "Scripts" / "python")
sys.path.insert(0, str(PROJECT_ROOT))

from src.episim import SIRSimulationConfig, estimate_large_outbreak_probability  # noqa: E402
from src.window_cache import load_cached_windows  # noqa: E402

DATASETS = ["Thiers13", "LyonSchool", "InVS15", "HT2009", "Infectious"]
TARGET_MEDIAN_LO = 0.15
TARGET_MEDIAN_HI = 0.35
TARGET_STD = 0.15
TARGET_LOW = 0.20
TARGET_HIGH = 0.20


def main() -> None:
    args = parse_args()
    output_dir = PROJECT_ROOT / "results" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_rows: list[dict[str, float | str | int]] = []
    best_rows: list[dict[str, float | str | int]] = []

    beta_scales = [float(x) for x in args.beta_scales.split(",")]
    gamma_scales = [float(x) for x in args.gamma_scales.split(",")]
    baseline = pd.read_csv(args.baseline_csv) if args.baseline_csv.exists() else pd.DataFrame()
    baseline_map = {
        str(row["dataset"]): (float(row["beta_per_second"]), float(row["gamma_per_second"]))
        for _, row in baseline.iterrows()
    }

    for dataset in tqdm(DATASETS, desc="calibrate_datasets", unit="dataset"):
        windows = load_cached_windows(PROJECT_ROOT / "data" / "preprocessed" / dataset)
        sample_ids = pick_window_sample_ids(len(windows), args.sample_windows)
        sample_windows = [windows[idx] for idx in sample_ids]
        beta_cur = float(baseline_map.get(dataset, (args.default_beta, args.default_gamma))[0])
        gamma_cur = float(baseline_map.get(dataset, (args.default_beta, args.default_gamma))[1])

        best = evaluate_pair(
            dataset=dataset,
            windows=sample_windows,
            beta=beta_cur,
            gamma=gamma_cur,
            tau=args.tau,
            num_simulations=args.calibration_num_simulations,
            horizon_seconds=args.horizon_seconds,
            seed=args.seed,
            sample_ids=sample_ids,
        )
        best["round"] = 0
        grid_rows.append(best.copy())

        for round_idx in range(1, args.search_rounds + 1):
            candidates = []
            for b_scale in beta_scales:
                for g_scale in gamma_scales:
                    beta = float(np.clip(beta_cur * b_scale, args.beta_min, args.beta_max))
                    gamma = float(np.clip(gamma_cur * g_scale, args.gamma_min, args.gamma_max))
                    candidates.append((beta, gamma))
            candidates = sorted(set(candidates))

            improved = False
            for beta, gamma in tqdm(candidates, desc=f"{dataset}:round{round_idx}", unit="pair", leave=False):
                row = evaluate_pair(
                    dataset=dataset,
                    windows=sample_windows,
                    beta=beta,
                    gamma=gamma,
                    tau=args.tau,
                    num_simulations=args.calibration_num_simulations,
                    horizon_seconds=args.horizon_seconds,
                    seed=args.seed,
                    sample_ids=sample_ids,
                )
                row["round"] = round_idx
                grid_rows.append(row)
                if float(row["objective"]) + 1e-12 < float(best["objective"]):
                    best = row
                    improved = True

            beta_cur = float(best["beta_per_second"])
            gamma_cur = float(best["gamma_per_second"])
            if not improved:
                break

        best_rows.append(best)

    grid_df = pd.DataFrame(grid_rows).sort_values(["dataset", "round", "objective", "beta_per_second", "gamma_per_second"])
    best_df = pd.DataFrame(best_rows).sort_values("dataset")
    grid_path = output_dir / "sir_calibration_search.csv"
    best_path = output_dir / "sir_calibration_by_dataset.csv"
    grid_df.to_csv(grid_path, index=False)
    best_df.to_csv(best_path, index=False)
    print(f"[ok] wrote calibration search: {grid_path}")
    print(f"[ok] wrote calibration winners: {best_path}")

    if args.apply_full_cache:
        recache_all(best_df, args)


def evaluate_pair(
    dataset: str,
    windows: list[pd.DataFrame],
    beta: float,
    gamma: float,
    tau: float,
    num_simulations: int,
    horizon_seconds: float,
    seed: int,
    sample_ids: list[int],
) -> dict[str, float | str | int]:
    labels = []
    for idx, window in enumerate(windows):
        cfg = SIRSimulationConfig(
            beta_per_second=beta,
            gamma_per_second=gamma,
            tau=tau,
            num_simulations=num_simulations,
            horizon_seconds=horizon_seconds,
            seed=seed + idx,
        )
        y, _, _ = estimate_large_outbreak_probability(window, cfg)
        labels.append(float(y))

    y_arr = np.asarray(labels, dtype=float)
    median_y = float(np.median(y_arr))
    std_y = float(np.std(y_arr))
    prop_low = float(np.mean(y_arr < 0.1))
    prop_high = float(np.mean(y_arr > 0.5))
    objective = calibration_objective(median_y, std_y, prop_low, prop_high)
    return {
        "dataset": dataset,
        "beta_per_second": beta,
        "gamma_per_second": gamma,
        "median_y": median_y,
        "std_y": std_y,
        "prop_low_y": prop_low,
        "prop_high_y": prop_high,
        "mean_y": float(np.mean(y_arr)),
        "min_y": float(np.min(y_arr)),
        "max_y": float(np.max(y_arr)),
        "objective": float(objective),
        "sample_windows": int(len(windows)),
        "sample_window_ids": "|".join(str(i) for i in sample_ids),
    }


def calibration_objective(median_y: float, std_y: float, prop_low: float, prop_high: float) -> float:
    penalty = 0.0
    if median_y < TARGET_MEDIAN_LO:
        penalty += 6.0 * (TARGET_MEDIAN_LO - median_y)
    elif median_y > TARGET_MEDIAN_HI:
        penalty += 6.0 * (median_y - TARGET_MEDIAN_HI)
    else:
        penalty += abs(median_y - 0.25)

    if std_y < TARGET_STD:
        penalty += 4.0 * (TARGET_STD - std_y)
    if prop_low < TARGET_LOW:
        penalty += 3.0 * (TARGET_LOW - prop_low)
    if prop_high < TARGET_HIGH:
        penalty += 3.0 * (TARGET_HIGH - prop_high)
    return penalty


def pick_window_sample_ids(num_windows: int, sample_windows: int) -> list[int]:
    if num_windows <= sample_windows:
        return list(range(num_windows))
    indices = np.linspace(0, num_windows - 1, sample_windows)
    return sorted({int(round(x)) for x in indices})


def recache_all(calibration_table: pd.DataFrame, args: argparse.Namespace) -> None:
    for _, row in tqdm(calibration_table.iterrows(), total=len(calibration_table), desc="apply_calibrated_cache", unit="dataset"):
        dataset = str(row["dataset"])
        cmd = [
            PYTHON_EXE,
            "scripts/cache_sir_labels.py",
            "--dataset",
            dataset,
            "--beta-per-second",
            str(float(row["beta_per_second"])),
            "--gamma-per-second",
            str(float(row["gamma_per_second"])),
            "--horizon-seconds",
            str(args.horizon_seconds),
            "--num-simulations",
            str(args.full_num_simulations),
            "--workers",
            str(args.full_workers),
            "--flush-every",
            str(args.full_flush_every),
            "--resume",
            "--force",
        ]
        completed = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"full cache failed for dataset: {dataset}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate beta/gamma by dataset and optionally recache SIR labels.")
    parser.add_argument("--tau", type=float, default=0.20)
    parser.add_argument("--horizon-seconds", type=float, default=86400.0)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--sample-windows", type=int, default=10)
    parser.add_argument("--calibration-num-simulations", type=int, default=16)
    parser.add_argument("--search-rounds", type=int, default=3)
    parser.add_argument("--default-beta", type=float, default=0.01)
    parser.add_argument("--default-gamma", type=float, default=0.000046296)
    parser.add_argument("--beta-scales", type=str, default="0.5,0.8,1.0,1.25,1.6")
    parser.add_argument("--gamma-scales", type=str, default="0.5,0.8,1.0,1.25,1.6")
    parser.add_argument("--beta-min", type=float, default=1e-5)
    parser.add_argument("--beta-max", type=float, default=0.05)
    parser.add_argument("--gamma-min", type=float, default=1e-6)
    parser.add_argument("--gamma-max", type=float, default=0.001)
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "output" / "sir_calibration_by_dataset.csv",
    )
    parser.add_argument("--apply-full-cache", action="store_true")
    parser.add_argument("--full-num-simulations", type=int, default=120)
    parser.add_argument("--full-workers", type=int, default=8)
    parser.add_argument("--full-flush-every", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    main()

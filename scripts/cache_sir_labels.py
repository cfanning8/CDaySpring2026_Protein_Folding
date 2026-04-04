from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
import time

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.episim import SIRSimulationConfig, estimate_large_outbreak_probability  # noqa: E402
from src.window_cache import load_cached_windows  # noqa: E402


def _simulate_window(
    window_id: int,
    window: pd.DataFrame,
    beta_per_second: float,
    gamma_per_second: float,
    tau: float,
    num_simulations: int,
    horizon_seconds: float,
    seed: int,
) -> dict[str, float | int]:
    cfg = SIRSimulationConfig(
        beta_per_second=beta_per_second,
        gamma_per_second=gamma_per_second,
        tau=tau,
        num_simulations=num_simulations,
        horizon_seconds=horizon_seconds,
        seed=seed + window_id,
    )
    t0 = time.perf_counter()
    large_prob, mean_attack, mean_peak = estimate_large_outbreak_probability(window, cfg)
    elapsed_s = float(time.perf_counter() - t0)
    return {
        "window_id": window_id,
        "y_large_outbreak_prob": float(large_prob),
        "mean_attack_rate": float(mean_attack),
        "mean_peak_prevalence": float(mean_peak),
        "elapsed_s": elapsed_s,
    }


def main() -> None:
    args = parse_args()
    cache_dir = PROJECT_ROOT / "data" / "preprocessed" / args.dataset
    windows_path = cache_dir / "windows.npz"
    output_csv = cache_dir / "sir_labels.csv"
    if not windows_path.exists():
        raise ValueError(f"missing windows cache: {windows_path}")
    if output_csv.exists() and not args.force:
        print(f"[cache] SIR labels already cached for {args.dataset}")
        return

    windows = load_cached_windows(cache_dir)
    if len(windows) < 2:
        raise ValueError(f"need at least 2 windows for SIR label cache, got {len(windows)}")

    rows: list[dict[str, float | int]] = []
    done_ids: set[int] = set()
    if output_csv.exists() and args.resume and not args.force:
        existing = pd.read_csv(output_csv)
        for row in existing.to_dict("records"):
            wid = int(row["window_id"])
            rows.append(
                {
                    "window_id": wid,
                    "y_large_outbreak_prob": float(row["y_large_outbreak_prob"]),
                    "mean_attack_rate": float(row["mean_attack_rate"]),
                    "mean_peak_prevalence": float(row["mean_peak_prevalence"]),
                    "elapsed_s": float(row.get("elapsed_s", 0.0)),
                }
            )
            done_ids.add(wid)
        print(f"[resume] loaded {len(done_ids)} cached windows from {output_csv}")

    pending = [(wid, windows[wid]) for wid in range(len(windows)) if wid not in done_ids]
    if not pending:
        print(f"[cache] no pending windows for {args.dataset}")
        return

    if args.workers <= 1:
        for window_id, window in tqdm(pending, desc=f"{args.dataset}:sir", unit="window"):
            rows.append(
                _simulate_window(
                    window_id=window_id,
                    window=window,
                    beta_per_second=args.beta_per_second,
                    gamma_per_second=args.gamma_per_second,
                    tau=args.tau,
                    num_simulations=args.num_simulations,
                    horizon_seconds=args.horizon_seconds,
                    seed=args.seed,
                )
            )
            if len(rows) % args.flush_every == 0:
                _flush_rows(rows, output_csv)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(
                    _simulate_window,
                    window_id,
                    window,
                    args.beta_per_second,
                    args.gamma_per_second,
                    args.tau,
                    args.num_simulations,
                    args.horizon_seconds,
                    args.seed,
                ): window_id
                for window_id, window in pending
            }
            for future in tqdm(as_completed(future_map), total=len(future_map), desc=f"{args.dataset}:sir", unit="window"):
                rows.append(future.result())
                if len(rows) % args.flush_every == 0:
                    _flush_rows(rows, output_csv)

    _flush_rows(rows, output_csv)
    print(f"[ok] cached SIR labels: {output_csv}")
    elapsed_series = pd.DataFrame(rows)["elapsed_s"].to_numpy(dtype=float)
    print(
        f"[timing] window mean={elapsed_series.mean():.3f}s median={float(pd.Series(elapsed_series).median()):.3f}s "
        f"max={elapsed_series.max():.3f}s"
    )


def _flush_rows(rows: list[dict[str, float | int]], output_csv: Path) -> None:
    frame = pd.DataFrame(rows).drop_duplicates(subset=["window_id"], keep="last").sort_values("window_id")
    frame.to_csv(output_csv, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache SIR labels for one dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tau", type=float, default=0.20)
    parser.add_argument("--num-simulations", type=int, default=100)
    parser.add_argument("--beta-per-second", type=float, required=True)
    parser.add_argument("--gamma-per-second", type=float, required=True)
    parser.add_argument("--horizon-seconds", type=float, required=True)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--flush-every", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

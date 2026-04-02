from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXE = str(PROJECT_ROOT / ".venv" / "Scripts" / "python")

DATASET_CONFIGS = {
    "Thiers13": {
        "dataset_key": r"high_school\HighSchool2013_proximity_net.csv\High-School_data_2013.csv",
        "window_seconds": 45 * 60,
        "stride_seconds": 45 * 60,
        "min_events": 80,
    },
    "LyonSchool": {
        "dataset_key": r"primary_school\primaryschool.csv\primaryschool.csv",
        "window_seconds": 45 * 60,
        "stride_seconds": 45 * 60,
        "min_events": 80,
    },
    "InVS15": {
        "dataset_key": r"workplace\workplace_InVS15_tij.dat\tij_InVS15.dat",
        "window_seconds": 60 * 60,
        "stride_seconds": 60 * 60,
        "min_events": 60,
    },
    "HT2009": {
        "dataset_key": r"hypertext\ht2009_contact_list.dat\ht09_contact_list.dat",
        "window_seconds": 30 * 60,
        "stride_seconds": 30 * 60,
        "min_events": 60,
    },
    "Infectious": {
        "dataset_key": r"infectious\sciencegallery_infectious_contacts\listcontacts_2009_06_10.txt",
        "window_seconds": 24 * 60 * 60,
        "stride_seconds": 24 * 60 * 60,
        "min_events": 20,
    },
}


def main() -> None:
    args = parse_args()
    cfg = DATASET_CONFIGS[args.dataset]
    slug = args.dataset.lower()
    feature_npz = PROJECT_ROOT / "temp" / f"pilot_{slug}" / "features.npz"
    cache_dir = PROJECT_ROOT / "data" / "preprocessed" / args.dataset
    output_dir = PROJECT_ROOT / "results" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not feature_npz.exists():
        raise ValueError(f"missing feature cache for dataset {args.dataset}: {feature_npz}")

    rows: list[dict[str, float | str]] = []
    base_tag = f"tune_{slug}"

    for idx, value in enumerate(_parse_values(args.lambda_d_grid)):
        tag = f"{base_tag}_perslay_{idx}"
        cmd = [
            PYTHON_EXE,
            "scripts/train_tgn_perslay_constraint.py",
            "--dataset-key",
            str(cfg["dataset_key"]),
            "--window-seconds",
            str(cfg["window_seconds"]),
            "--stride-seconds",
            str(cfg["stride_seconds"]),
            "--min-events-per-window",
            str(cfg["min_events"]),
            "--feature-npz",
            str(feature_npz),
            "--preprocessed-cache-dir",
            str(cache_dir),
            "--epochs",
            str(args.epochs),
            "--early-stopping-patience",
            str(args.early_stopping_patience),
            "--early-stopping-min-delta",
            str(args.early_stopping_min_delta),
            "--lambda-d",
            str(value),
            "--output-tag",
            tag,
            "--device",
            "cuda",
        ]
        run_command(cmd, f"perslay_lambda_d={value}")
        pred_path = output_dir / f"tgn_perslay_constraint_predictions_{tag}.csv"
        rmse = evaluate_rmse(pred_path)
        rows.append({"dataset": args.dataset, "model": "PersLay", "lambda_value": value, "rmse": rmse, "tag": tag})

    for idx, value in enumerate(_parse_values(args.lambda_g_grid)):
        tag = f"{base_tag}_rkhs_{idx}"
        cmd = [
            PYTHON_EXE,
            "scripts/train_tgn_rkhs_constraint.py",
            "--dataset-key",
            str(cfg["dataset_key"]),
            "--window-seconds",
            str(cfg["window_seconds"]),
            "--stride-seconds",
            str(cfg["stride_seconds"]),
            "--min-events-per-window",
            str(cfg["min_events"]),
            "--feature-npz",
            str(feature_npz),
            "--preprocessed-cache-dir",
            str(cache_dir),
            "--epochs",
            str(args.epochs),
            "--early-stopping-patience",
            str(args.early_stopping_patience),
            "--early-stopping-min-delta",
            str(args.early_stopping_min_delta),
            "--lambda-g",
            str(value),
            "--output-tag",
            tag,
            "--device",
            "cuda",
        ]
        run_command(cmd, f"rkhs_lambda_g={value}")
        pred_path = output_dir / f"tgn_rkhs_constraint_predictions_{tag}.csv"
        rmse = evaluate_rmse(pred_path)
        rows.append({"dataset": args.dataset, "model": "RKHS", "lambda_value": value, "rmse": rmse, "tag": tag})

    result = pd.DataFrame(rows).sort_values(["model", "rmse", "lambda_value"]).reset_index(drop=True)
    out_csv = output_dir / f"topology_loss_tuning_{slug}.csv"
    best_csv = output_dir / f"topology_loss_tuning_{slug}_best.csv"
    result.to_csv(out_csv, index=False)
    best = result.groupby("model", as_index=False).first()
    best.to_csv(best_csv, index=False)
    print(f"[ok] wrote tuning results: {out_csv}")
    print(f"[ok] wrote best rows: {best_csv}")


def evaluate_rmse(predictions_csv: Path) -> float:
    frame = pd.read_csv(predictions_csv)
    if "split" in frame.columns and (frame["split"] == "test").any():
        frame = frame[frame["split"] == "test"].copy()
    if len(frame) == 0:
        raise RuntimeError(f"empty evaluation slice for tuning file: {predictions_csv}")
    y_true = frame["y_true"].to_numpy(dtype=float)
    y_pred = frame["y_pred"].to_numpy(dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def _parse_values(grid: str) -> list[float]:
    values = [float(item.strip()) for item in grid.split(",") if item.strip()]
    if len(values) == 0:
        raise ValueError("lambda grid must contain at least one numeric value")
    return values


def run_command(command: list[str], name: str) -> None:
    print(f"[run] {name}: {' '.join(command)}")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gentle tuning for topology loss terms on one active dataset.")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--lambda-d-grid", type=str, default="0.1,0.3,0.5")
    parser.add_argument("--lambda-g-grid", type=str, default="0.3,0.7,1.0")
    return parser.parse_args()


if __name__ == "__main__":
    main()

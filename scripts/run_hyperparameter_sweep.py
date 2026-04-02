from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    args = parse_args()
    output_dir = PROJECT_ROOT / "results" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        {"learning_rate": 0.001, "lambda_d": 0.10, "lambda_g": 0.30},
        {"learning_rate": 0.001, "lambda_d": 0.30, "lambda_g": 0.70},
        {"learning_rate": 0.001, "lambda_d": 0.50, "lambda_g": 1.00},
        {"learning_rate": 0.0005, "lambda_d": 0.10, "lambda_g": 0.30},
        {"learning_rate": 0.0005, "lambda_d": 0.30, "lambda_g": 0.70},
        {"learning_rate": 0.0005, "lambda_d": 0.50, "lambda_g": 1.00},
    ]

    rows: list[dict[str, float | str]] = []
    baseline_tag = "sweep_baseline"
    run_command(
        [
            str(PROJECT_ROOT / ".venv" / "Scripts" / "python"),
            "scripts/train_tgn_baseline.py",
            "--epochs",
            str(args.epochs),
            "--output-tag",
            baseline_tag,
            "--device",
            "cuda",
        ]
    )
    baseline_metrics = evaluate_predictions(output_dir / f"tgn_baseline_predictions_{baseline_tag}.csv")
    rows.append({"model": "TGN", "tag": baseline_tag, **baseline_metrics, "learning_rate": 0.001, "lambda_d": 0.0, "lambda_g": 0.0})

    for idx, cfg in enumerate(configs):
        tag_p = f"sweep_perslay_{idx}"
        run_command(
            [
                str(PROJECT_ROOT / ".venv" / "Scripts" / "python"),
                "scripts/train_tgn_perslay_constraint.py",
                "--epochs",
                str(args.epochs),
                "--learning-rate",
                str(cfg["learning_rate"]),
                "--lambda-d",
                str(cfg["lambda_d"]),
                "--lambda-g",
                str(cfg["lambda_g"]),
                "--output-tag",
                tag_p,
                "--device",
                "cuda",
            ]
        )
        metrics_p = evaluate_predictions(output_dir / f"tgn_perslay_constraint_predictions_{tag_p}.csv")
        rows.append({"model": "PersLay", "tag": tag_p, **metrics_p, **cfg})

        tag_r = f"sweep_rkhs_{idx}"
        run_command(
            [
                str(PROJECT_ROOT / ".venv" / "Scripts" / "python"),
                "scripts/train_tgn_rkhs_constraint.py",
                "--epochs",
                str(args.epochs),
                "--learning-rate",
                str(cfg["learning_rate"]),
                "--lambda-d",
                str(cfg["lambda_d"]),
                "--lambda-g",
                str(cfg["lambda_g"]),
                "--output-tag",
                tag_r,
                "--device",
                "cuda",
            ]
        )
        metrics_r = evaluate_predictions(output_dir / f"tgn_rkhs_constraint_predictions_{tag_r}.csv")
        rows.append({"model": "RKHS", "tag": tag_r, **metrics_r, **cfg})

    result_df = pd.DataFrame(rows).sort_values(["model", "rmse"])
    result_df.to_csv(output_dir / "hyperparameter_sweep_primary_school.csv", index=False)
    best_df = result_df.sort_values("rmse").groupby("model", as_index=False).first()
    best_df.to_csv(output_dir / "hyperparameter_sweep_primary_school_best.csv", index=False)
    print(f"Saved sweep table: {output_dir / 'hyperparameter_sweep_primary_school.csv'}")
    print(f"Saved best configs: {output_dir / 'hyperparameter_sweep_primary_school_best.csv'}")


def evaluate_predictions(predictions_path: Path) -> dict[str, float]:
    df = pd.read_csv(predictions_path)
    y_true = df["y_true"].to_numpy(dtype=float)
    y_pred = df["y_pred"].to_numpy(dtype=float)
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    return {"rmse": rmse}


def run_command(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        raise RuntimeError(f"command failed: {' '.join(command)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hyperparameter sweep for PersLay and RKHS constraints.")
    parser.add_argument("--epochs", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import itertools
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
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for d in datasets:
        if d not in DATASET_CONFIGS:
            raise ValueError(f"unknown dataset in --datasets: {d}")

    combos = list(
        itertools.product(
            parse_float_grid(args.lambda_g_grid),
            parse_float_grid(args.learning_rate_grid),
            parse_int_grid(args.fusion_hidden_grid),
            parse_int_grid(args.rkhs_proj_hidden_grid),
            parse_float_grid(args.weight_decay_grid),
        )
    )
    if len(combos) == 0:
        raise ValueError("empty tuning grid")
    if args.max_combos > 0:
        combos = combos[: args.max_combos]

    rows: list[dict[str, float | str | int]] = []
    output_dir = PROJECT_ROOT / "results" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        cfg = DATASET_CONFIGS[dataset]
        slug = dataset.lower()
        feature_npz = PROJECT_ROOT / "temp" / f"pilot_{slug}" / "features.npz"
        cache_dir = PROJECT_ROOT / "data" / "preprocessed" / dataset
        if not feature_npz.exists():
            raise ValueError(f"missing feature cache for dataset {dataset}: {feature_npz}")
        if not cache_dir.exists():
            raise ValueError(f"missing preprocessed cache for dataset {dataset}: {cache_dir}")

        for idx, (lambda_g, learning_rate, fusion_hidden, rkhs_proj_hidden, weight_decay) in enumerate(combos):
            tag = f"tune_{slug}_{idx:03d}"
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
                str(lambda_g),
                "--learning-rate",
                str(learning_rate),
                "--fusion-hidden-dim",
                str(fusion_hidden),
                "--rkhs-proj-hidden-dim",
                str(rkhs_proj_hidden),
                "--weight-decay",
                str(weight_decay),
                "--output-tag",
                tag,
                "--device",
                "cuda",
            ]
            run_command(cmd, f"{dataset}:combo_{idx}")
            pred_path = output_dir / f"tgn_rkhs_constraint_predictions_{tag}.csv"
            rmse, brier, ece = evaluate_predictions(pred_path, risk_threshold=args.risk_threshold)
            rows.append(
                {
                    "dataset": dataset,
                    "combo_id": idx,
                    "lambda_g": lambda_g,
                    "learning_rate": learning_rate,
                    "fusion_hidden_dim": fusion_hidden,
                    "rkhs_proj_hidden_dim": rkhs_proj_hidden,
                    "weight_decay": weight_decay,
                    "rmse": rmse,
                    "brier": brier,
                    "ece": ece,
                    "tag": tag,
                }
            )

    result_df = pd.DataFrame(rows).sort_values(["dataset", "rmse", "brier", "ece", "combo_id"]).reset_index(drop=True)
    result_csv = output_dir / "rkhs_tuning_grid_results.csv"
    result_df.to_csv(result_csv, index=False)
    best_df = result_df.groupby("dataset", as_index=False).first()
    best_csv = output_dir / "rkhs_tuning_best_by_dataset.csv"
    best_df.to_csv(best_csv, index=False)
    print(f"[ok] wrote grid results: {result_csv}")
    print(f"[ok] wrote best configs: {best_csv}")

    if args.apply_best:
        apply_best_configs(best_df, args)


def apply_best_configs(best_df: pd.DataFrame, args: argparse.Namespace) -> None:
    for _, row in best_df.iterrows():
        dataset = str(row["dataset"])
        cfg = DATASET_CONFIGS[dataset]
        slug = dataset.lower()
        feature_npz = PROJECT_ROOT / "temp" / f"pilot_{slug}" / "features.npz"
        cache_dir = PROJECT_ROOT / "data" / "preprocessed" / dataset
        canonical_tag = f"{slug}_rkhs"
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
            str(float(row["lambda_g"])),
            "--learning-rate",
            str(float(row["learning_rate"])),
            "--fusion-hidden-dim",
            str(int(row["fusion_hidden_dim"])),
            "--rkhs-proj-hidden-dim",
            str(int(row["rkhs_proj_hidden_dim"])),
            "--weight-decay",
            str(float(row["weight_decay"])),
            "--output-tag",
            canonical_tag,
            "--device",
            "cuda",
        ]
        run_command(cmd, f"{dataset}:apply_best")


def evaluate_predictions(predictions_path: Path, risk_threshold: float) -> tuple[float, float, float]:
    df = pd.read_csv(predictions_path)
    if "split" in df.columns and (df["split"] == "test").any():
        df = df[df["split"] == "test"].copy()
    if len(df) == 0:
        raise RuntimeError(f"empty evaluation slice for predictions file: {predictions_path}")
    y_true = df["y_true"].to_numpy(dtype=float)
    y_pred = np.clip(df["y_pred"].to_numpy(dtype=float), 0.0, 1.0)
    y_true_bin = (y_true >= risk_threshold).astype(np.int64)
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    brier = float(np.mean((y_pred - y_true_bin.astype(float)) ** 2))
    ece = expected_calibration_error(y_true_bin, y_pred, num_bins=10)
    return rmse, brier, ece


def expected_calibration_error(y_true_binary: np.ndarray, y_prob: np.ndarray, num_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    total = max(1, y_true_binary.size)
    ece = 0.0
    for idx in range(num_bins):
        lo = edges[idx]
        hi = edges[idx + 1]
        if idx == num_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true_binary[mask]))
        ece += (float(np.sum(mask)) / float(total)) * abs(acc - conf)
    return float(ece)


def run_command(command: list[str], name: str) -> None:
    print(f"[run] {name}: {' '.join(command)}")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed for {name}")


def parse_float_grid(raw: str) -> list[float]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("float grid cannot be empty")
    return values


def parse_int_grid(raw: str) -> list[int]:
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("int grid cannot be empty")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune RKHS fusion+constraint hyperparameters across active datasets.")
    parser.add_argument("--datasets", type=str, default="Thiers13,LyonSchool,InVS15,HT2009,Infectious")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--risk-threshold", type=float, default=0.20)
    parser.add_argument("--lambda-g-grid", type=str, default="0.3,0.7,1.2")
    parser.add_argument("--learning-rate-grid", type=str, default="0.0005,0.001")
    parser.add_argument("--fusion-hidden-grid", type=str, default="64,128")
    parser.add_argument("--rkhs-proj-hidden-grid", type=str, default="64,128")
    parser.add_argument("--weight-decay-grid", type=str, default="0.0,0.0001")
    parser.add_argument("--max-combos", type=int, default=12)
    parser.add_argument("--apply-best", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

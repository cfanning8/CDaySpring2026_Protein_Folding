from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm

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
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"dataset must be one of: {', '.join(DATASET_CONFIGS.keys())}")

    cfg = DATASET_CONFIGS[args.dataset]
    if args.reset_cache:
        reset_dataset_cache(args.dataset)
    beta_value, gamma_value = resolve_beta_gamma_for_dataset(args.dataset, args)
    output_dir = PROJECT_ROOT / "results" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    slug = args.dataset.lower()
    subdir = f"pilot_{slug}"
    feature_npz = PROJECT_ROOT / "temp" / subdir / "features.npz"
    table_csv = PROJECT_ROOT / "temp" / subdir / "table.csv"
    cache_dir = PROJECT_ROOT / "data" / "preprocessed" / args.dataset
    persistence_npz = cache_dir / "persistence_features.npz"
    graph_csv = cache_dir / "graph_features.csv"
    sir_csv = cache_dir / "sir_labels.csv"

    steps = [
        "cache_windows",
        "cache_persistence",
        "cache_sir",
        "build_table",
        "train_tgn",
        "train_perslay",
        "train_rkhs",
        "update_collective_csvs",
    ]
    for step in tqdm(steps, desc=f"{args.dataset} pipeline", unit="step"):
        if step == "cache_windows":
            if (cache_dir / "windows.npz").exists() and not args.force_rebuild:
                print(f"[cache] using existing preprocessed windows for {args.dataset}")
            else:
                cache_cmd = [PYTHON_EXE, "scripts/cache_dataset_windows.py", "--dataset", args.dataset]
                if args.force_rebuild:
                    cache_cmd.append("--force")
                run_command(cache_cmd, f"{args.dataset}:cache_windows")
            assert (cache_dir / "windows.npz").exists(), f"missing windows cache for {args.dataset}: {cache_dir / 'windows.npz'}"

        elif step == "cache_persistence":
            if persistence_npz.exists() and graph_csv.exists() and not args.force_rebuild:
                print(f"[cache] using existing persistence cache for {args.dataset}")
            else:
                cmd = [PYTHON_EXE, "scripts/cache_persistence_features.py", "--dataset", args.dataset]
                if args.force_rebuild:
                    cmd.append("--force")
                run_command(cmd, f"{args.dataset}:cache_persistence")
            assert persistence_npz.exists(), f"missing persistence cache: {persistence_npz}"
            assert graph_csv.exists(), f"missing graph cache: {graph_csv}"

        elif step == "cache_sir":
            if sir_csv.exists() and not args.force_rebuild:
                print(f"[cache] using existing SIR cache for {args.dataset}")
            else:
                cmd = [
                    PYTHON_EXE,
                    "scripts/cache_sir_labels.py",
                    "--dataset",
                    args.dataset,
                    "--beta-per-second",
                    str(beta_value),
                    "--gamma-per-second",
                    str(gamma_value),
                    "--horizon-seconds",
                    str(args.horizon_seconds),
                    "--num-simulations",
                    str(args.num_simulations),
                    "--workers",
                    str(args.sir_workers),
                    "--flush-every",
                    str(args.sir_flush_every),
                ]
                if args.force_rebuild:
                    cmd.append("--force")
                run_command(cmd, f"{args.dataset}:cache_sir")
            assert sir_csv.exists(), f"missing SIR label cache: {sir_csv}"

        elif step == "build_table":
            if feature_npz.exists() and table_csv.exists() and not args.force_rebuild:
                print(f"[cache] using existing table/features for {args.dataset}")
            else:
                build_cmd = [
                    PYTHON_EXE,
                    "scripts/build_primary_school_pilot_table.py",
                    "--dataset-key",
                    cfg["dataset_key"],
                    "--window-seconds",
                    str(cfg["window_seconds"]),
                    "--stride-seconds",
                    str(cfg["window_seconds"]),
                    "--min-events-per-window",
                    str(cfg["min_events"]),
                    "--beta-per-second",
                    str(beta_value),
                    "--gamma-per-second",
                    str(gamma_value),
                    "--horizon-seconds",
                    str(args.horizon_seconds),
                    "--num-simulations",
                    str(args.num_simulations),
                    "--output-subdir",
                    subdir,
                    "--preprocessed-cache-dir",
                    str(cache_dir),
                    "--persistence-cache-path",
                    str(persistence_npz),
                    "--graph-cache-csv",
                    str(graph_csv),
                    "--sir-cache-path",
                    str(sir_csv),
                ]
                run_command(build_cmd, f"{args.dataset}:build")
            assert feature_npz.exists(), f"missing feature file: {feature_npz}"
            assert table_csv.exists(), f"missing table file: {table_csv}"

        elif step == "train_tgn":
            train_and_cache_model(
                dataset_name=args.dataset,
                model_name="TGN",
                base_cmd=[PYTHON_EXE, "scripts/train_tgn_baseline.py"],
                cfg=cfg,
                feature_npz=feature_npz,
                preprocessed_cache_dir=cache_dir,
                epochs=args.epochs,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_min_delta=args.early_stopping_min_delta,
                force=args.force_train,
            )
        elif step == "train_perslay":
            train_and_cache_model(
                dataset_name=args.dataset,
                model_name="PersLay",
                base_cmd=[PYTHON_EXE, "scripts/train_tgn_perslay_constraint.py"],
                cfg=cfg,
                feature_npz=feature_npz,
                preprocessed_cache_dir=cache_dir,
                epochs=args.epochs,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_min_delta=args.early_stopping_min_delta,
                force=args.force_train,
            )
        elif step == "train_rkhs":
            train_and_cache_model(
                dataset_name=args.dataset,
                model_name="RKHS",
                base_cmd=[PYTHON_EXE, "scripts/train_tgn_rkhs_constraint.py"],
                cfg=cfg,
                feature_npz=feature_npz,
                preprocessed_cache_dir=cache_dir,
                epochs=args.epochs,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_min_delta=args.early_stopping_min_delta,
                force=args.force_train,
            )
        else:
            update_collective_csvs(args.dataset, table_csv, output_dir)


def evaluate_predictions(predictions_path: Path, risk_threshold: float) -> dict[str, float | str]:
    df = pd.read_csv(predictions_path)
    eval_scope = "all"
    if "split" in df.columns:
        test_df = df[df["split"] == "test"].copy()
        if len(test_df) > 0:
            df = test_df
            eval_scope = "test"
        else:
            raise RuntimeError(
                f"missing test split rows in predictions file: {predictions_path}. "
                "Chronological test evaluation is required."
            )
    if len(df) == 0:
        raise RuntimeError(f"empty evaluation slice in predictions file: {predictions_path}")
    y_true = df["y_true"].to_numpy(dtype=float)
    y_pred = np.clip(df["y_pred"].to_numpy(dtype=float), 0.0, 1.0)
    y_true_bin = (y_true >= risk_threshold).astype(np.int64)
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    brier = float(np.mean((y_pred - y_true_bin.astype(float)) ** 2))
    ece = expected_calibration_error(y_true_bin, y_pred, num_bins=10)
    positive_count = int(np.sum(y_true_bin == 1))
    negative_count = int(np.sum(y_true_bin == 0))
    return {
        "rmse": rmse,
        "brier": brier,
        "ece": ece,
        "eval_count": float(len(y_true_bin)),
        "positive_count": float(positive_count),
        "negative_count": float(negative_count),
        "eval_scope": eval_scope,
    }


def train_and_cache_model(
    dataset_name: str,
    model_name: str,
    base_cmd: list[str],
    cfg: dict[str, int | str],
    feature_npz: Path,
    preprocessed_cache_dir: Path,
    epochs: int,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    force: bool,
) -> None:
    output_dir = PROJECT_ROOT / "results" / "output"
    tag = f"{dataset_name.lower()}_{model_name.lower()}"
    if model_name == "TGN":
        pred_path = output_dir / f"tgn_baseline_predictions_{tag}.csv"
    elif model_name == "PersLay":
        pred_path = output_dir / f"tgn_perslay_constraint_predictions_{tag}.csv"
    else:
        pred_path = output_dir / f"tgn_rkhs_constraint_predictions_{tag}.csv"
    if pred_path.exists() and not force:
        print(f"[cache] using existing predictions for {dataset_name} / {model_name}")
        return
    train_cmd = (
        base_cmd
        + [
            "--dataset-key",
            str(cfg["dataset_key"]),
            "--window-seconds",
            str(cfg["window_seconds"]),
            "--stride-seconds",
            str(cfg.get("stride_seconds", cfg["window_seconds"])),
            "--min-events-per-window",
            str(cfg["min_events"]),
            "--feature-npz",
            str(feature_npz),
            "--preprocessed-cache-dir",
            str(preprocessed_cache_dir),
            "--epochs",
            str(epochs),
            "--early-stopping-patience",
            str(early_stopping_patience),
            "--early-stopping-min-delta",
            str(early_stopping_min_delta),
            "--output-tag",
            tag,
            "--device",
            "cuda",
        ]
    )
    run_command(train_cmd, f"{dataset_name}:train:{model_name}")
    assert pred_path.exists(), f"missing predictions after training: {pred_path}"


def update_collective_csvs(dataset_name: str, table_csv: Path, output_dir: Path) -> None:
    model_files = {
        "TGN": output_dir / f"tgn_baseline_predictions_{dataset_name.lower()}_tgn.csv",
        "PersLay": output_dir / f"tgn_perslay_constraint_predictions_{dataset_name.lower()}_perslay.csv",
        "RKHS": output_dir / f"tgn_rkhs_constraint_predictions_{dataset_name.lower()}_rkhs.csv",
    }
    for model_name, path in model_files.items():
        assert path.exists(), f"missing cached prediction for {dataset_name}/{model_name}: {path}"

    metrics_rows = []
    for model_name, path in model_files.items():
        metrics = evaluate_predictions(path, risk_threshold=0.20)
        metrics_rows.append({"dataset": dataset_name, "model": model_name, **metrics})
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = output_dir / "collective_metrics.csv"
    if metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        existing = existing[existing["dataset"] != dataset_name]
        metrics_df = pd.concat([existing, metrics_df], ignore_index=True)
    finite_rmse = metrics_df["rmse"].to_numpy(dtype=float)
    finite_rmse = finite_rmse[np.isfinite(finite_rmse)]
    max_rmse = float(np.max(finite_rmse)) if finite_rmse.size > 0 else 1.0
    metrics_df["rmse_norm"] = metrics_df["rmse"] / max(max_rmse, 1e-12)
    metrics_df = metrics_df.sort_values(["dataset", "model"])
    metrics_df.to_csv(metrics_path, index=False)

    table = pd.read_csv(table_csv)
    assert "y_large_outbreak_prob" in table.columns, "missing y_large_outbreak_prob in table"
    y = table["y_large_outbreak_prob"].to_numpy(dtype=float)
    y_ci = 1.96 * np.sqrt(np.maximum(y * (1.0 - y), 0.0) / 100.0)
    ts_rows = []
    for idx in range(len(table)):
        ts_rows.append(
            {
                "dataset": dataset_name,
                "t": int(table.loc[idx, "t"]),
                "y_t": float(table.loc[idx, "y_large_outbreak_prob"]),
                "g_l2_norm": float(table.loc[idx, "g_l2_norm"]),
                "y_ci": float(y_ci[idx]),
            }
        )
    ts_df = pd.DataFrame(ts_rows)
    ts_path = output_dir / "collective_timeseries.csv"
    if ts_path.exists():
        existing_ts = pd.read_csv(ts_path)
        existing_ts = existing_ts[existing_ts["dataset"] != dataset_name]
        ts_df = pd.concat([existing_ts, ts_df], ignore_index=True)
    ts_df = ts_df.sort_values(["dataset", "t"])
    ts_df.to_csv(ts_path, index=False)
    print(f"[ok] updated collective cache for {dataset_name}")


def run_command(command: list[str], name: str) -> None:
    print(f"[run] {name}: {' '.join(command)}")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        print(f"[error] {name}")
        raise RuntimeError(f"command failed for step: {name}")


def reset_dataset_cache(dataset_name: str) -> None:
    slug = dataset_name.lower()
    cache_dir = PROJECT_ROOT / "data" / "preprocessed" / dataset_name
    temp_dir = PROJECT_ROOT / "temp" / f"pilot_{slug}"
    output_dir = PROJECT_ROOT / "results" / "output"
    weight_dir = PROJECT_ROOT / "results" / "weights"

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"[reset] removed {cache_dir}")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        print(f"[reset] removed {temp_dir}")

    output_patterns = [
        f"tgn_baseline_history_{slug}_tgn.csv",
        f"tgn_baseline_predictions_{slug}_tgn.csv",
        f"tgn_perslay_constraint_history_{slug}_perslay.csv",
        f"tgn_perslay_constraint_predictions_{slug}_perslay.csv",
        f"tgn_rkhs_constraint_history_{slug}_rkhs.csv",
        f"tgn_rkhs_constraint_predictions_{slug}_rkhs.csv",
    ]
    for name in output_patterns:
        path = output_dir / name
        if path.exists():
            path.unlink()
            print(f"[reset] removed {path}")

    weight_patterns = [
        f"tgn_baseline_{slug}_tgn.pt",
        f"tgn_perslay_constraint_{slug}_perslay.pt",
        f"tgn_rkhs_constraint_{slug}_rkhs.pt",
    ]
    for name in weight_patterns:
        path = weight_dir / name
        if path.exists():
            path.unlink()
            print(f"[reset] removed {path}")


def resolve_beta_gamma_for_dataset(dataset_name: str, args: argparse.Namespace) -> tuple[float, float]:
    if args.use_calibrated and args.calibration_csv.exists():
        table = pd.read_csv(args.calibration_csv)
        rows = table[table["dataset"] == dataset_name]
        if not rows.empty:
            beta = float(rows.iloc[0]["beta_per_second"])
            gamma = float(rows.iloc[0]["gamma_per_second"])
            print(f"[calibrated] {dataset_name} beta={beta} gamma={gamma}")
            return beta, gamma
        raise RuntimeError(
            f"missing calibrated beta/gamma for dataset '{dataset_name}' in {args.calibration_csv}"
        )
    if args.use_calibrated and not args.calibration_csv.exists():
        raise RuntimeError(f"calibration CSV not found: {args.calibration_csv}")
    return float(args.beta_per_second), float(args.gamma_per_second)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one dataset benchmark and cache into collective CSVs.")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num-simulations", type=int, default=120)
    parser.add_argument("--beta-per-second", type=float, default=0.01)
    parser.add_argument("--gamma-per-second", type=float, default=0.000046296)
    parser.add_argument(
        "--calibration-csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "output" / "sir_calibration_by_dataset.csv",
    )
    parser.add_argument("--use-calibrated", dest="use_calibrated", action="store_true")
    parser.add_argument("--no-use-calibrated", dest="use_calibrated", action="store_false")
    parser.set_defaults(use_calibrated=True)
    parser.add_argument("--horizon-seconds", type=float, default=86400.0)
    parser.add_argument("--sir-workers", type=int, default=1)
    parser.add_argument("--sir-flush-every", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--reset-cache", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

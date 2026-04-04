from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.tgn_window_model import WindowTGNRegressor  # noqa: E402
from src.pilot_dataset import load_primary_school_pilot_dataset  # noqa: E402
from src.training_utils import TemporalSplitConfig, chronological_split_indices, rmse_on_indices  # noqa: E402

DEFAULT_DATASET_KEY = r"primary_school\primaryschool.csv\primaryschool.csv"


def main() -> None:
    args = parse_args()
    torch.manual_seed(14)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(14)
    dataset = load_primary_school_pilot_dataset(
        project_root=PROJECT_ROOT,
        dataset_key=args.dataset_key,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        min_events_per_window=args.min_events_per_window,
        feature_npz_path=args.feature_npz,
        preprocessed_cache_dir=args.preprocessed_cache_dir,
    )
    device = resolve_device(args.device)
    print(f"device={device}")
    model = WindowTGNRegressor(num_nodes=dataset.num_nodes, memory_dim=args.memory_dim, time_dim=args.time_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    mse = nn.MSELoss()
    train_ids, val_ids, test_ids = chronological_split_indices(
        len(dataset.y_t),
        TemporalSplitConfig(train_fraction=args.train_fraction, val_fraction=args.val_fraction),
    )
    train_index = torch.tensor(train_ids, dtype=torch.long, device=device)

    history: list[dict[str, float]] = []
    final_predictions = None
    best_predictions = None
    best_state: dict[str, torch.Tensor] | None = None
    best_metric = float("inf")
    epochs_without_improvement = 0
    for epoch in tqdm(range(args.epochs), desc="train_tgn_baseline", unit="epoch"):
        model.train()
        model.reset_state()
        optimizer.zero_grad()

        predictions = []
        for window in tqdm(dataset.windows, desc=f"epoch_{epoch}_windows", unit="window", leave=False):
            embedding = model.encode_window(window.src.to(device), window.dst.to(device), window.t_start.to(device), window.duration.to(device))
            predictions.append(model.predict_from_embedding(embedding))
            model.detach_memory()
        assert len(predictions) == len(dataset.windows), "prediction count mismatch with windows"

        prediction_tensor = torch.stack(predictions)
        target_tensor = dataset.y_t.to(device)
        loss = mse(prediction_tensor.index_select(0, train_index), target_tensor.index_select(0, train_index))
        loss.backward()
        optimizer.step()

        train_rmse = rmse_on_indices(prediction_tensor.detach(), target_tensor, train_ids)
        history_row = {"epoch": epoch, "loss": float(loss.item()), "train_rmse": float(train_rmse)}
        if len(val_ids) > 0:
            val_rmse = rmse_on_indices(prediction_tensor.detach(), target_tensor, val_ids)
            history_row["val_rmse"] = float(val_rmse)
        history.append(history_row)
        final_predictions = prediction_tensor.detach().cpu().numpy()

        monitor_metric = float(history_row["val_rmse"]) if "val_rmse" in history_row else float(history_row["train_rmse"])
        if monitor_metric + args.early_stopping_min_delta < best_metric:
            best_metric = monitor_metric
            epochs_without_improvement = 0
            best_predictions = final_predictions.copy()
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                print(f"early_stop_epoch={epoch}")
                break

    output_dir = PROJECT_ROOT / "results" / "output"
    weight_dir = PROJECT_ROOT / "results" / "weights"
    output_dir.mkdir(parents=True, exist_ok=True)
    weight_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{args.output_tag}" if args.output_tag else ""
    history_path = output_dir / f"tgn_baseline_history{suffix}.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    if best_state is not None:
        model.load_state_dict(best_state)
    final_to_save = best_predictions if best_predictions is not None else final_predictions
    if final_to_save is not None:
        pd.DataFrame(
            {
                "t": list(range(len(final_to_save))),
                "y_true": dataset.y_t.cpu().numpy(),
                "y_pred": final_to_save,
                "split": build_split_column(len(final_to_save), train_ids, val_ids, test_ids),
                "model": "tgn_baseline",
            }
        ).to_csv(output_dir / f"tgn_baseline_predictions{suffix}.csv", index=False)
    weight_path = weight_dir / f"tgn_baseline{suffix}.pt"
    torch.save(model.state_dict(), weight_path)

    print(f"Saved history: {history_path}")
    print(f"Saved weights: {weight_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TGN baseline on primary-school pilot data.")
    parser.add_argument("--dataset-key", default=DEFAULT_DATASET_KEY)
    parser.add_argument("--window-seconds", type=float, default=2700.0)
    parser.add_argument("--stride-seconds", type=float, default=1350.0)
    parser.add_argument("--min-events-per-window", type=int, default=100)
    parser.add_argument("--memory-dim", type=int, default=64)
    parser.add_argument("--time-dim", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--feature-npz", type=Path, default=PROJECT_ROOT / "temp" / "pilot_primary_school" / "features.npz")
    parser.add_argument("--preprocessed-cache-dir", type=Path, default=None)
    parser.add_argument("--output-tag", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise ValueError("cuda requested but torch.cuda.is_available() is False")
    return torch.device("cuda")


def build_split_column(num_points: int, train_ids: list[int], val_ids: list[int], test_ids: list[int]) -> list[str]:
    labels = ["test"] * num_points
    for idx in train_ids:
        labels[idx] = "train"
    for idx in val_ids:
        labels[idx] = "val"
    for idx in test_ids:
        labels[idx] = "test"
    return labels


if __name__ == "__main__":
    main()

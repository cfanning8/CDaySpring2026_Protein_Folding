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
from src.training_utils import TemporalSplitConfig, chronological_split_indices, pointwise_alignment_loss, rmse_on_indices  # noqa: E402

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
    train_ids, val_ids, test_ids = chronological_split_indices(
        len(dataset.y_t),
        TemporalSplitConfig(train_fraction=args.train_fraction, val_fraction=args.val_fraction),
    )
    train_index = torch.tensor(train_ids, dtype=torch.long, device=device)
    tgn = WindowTGNRegressor(num_nodes=dataset.num_nodes, memory_dim=args.memory_dim, time_dim=args.time_dim).to(device)
    rkhs_dim = int(dataset.rkhs_g_t.shape[1])
    if args.rkhs_proj_hidden_dim > 0:
        latent_to_rkhs = nn.Sequential(
            nn.Linear(args.memory_dim, args.rkhs_proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.rkhs_proj_hidden_dim, rkhs_dim),
        ).to(device)
    else:
        latent_to_rkhs = nn.Linear(args.memory_dim, rkhs_dim).to(device)
    fusion_norm = nn.LayerNorm(args.memory_dim + rkhs_dim).to(device)
    fusion_head = nn.Sequential(
        nn.Linear(args.memory_dim + rkhs_dim, args.fusion_hidden_dim),
        nn.ReLU(),
        nn.Linear(args.fusion_hidden_dim, 1),
        nn.Sigmoid(),
    ).to(device)
    params = list(tgn.parameters()) + list(latent_to_rkhs.parameters()) + list(fusion_norm.parameters()) + list(fusion_head.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    mse = nn.MSELoss()

    history: list[dict[str, float]] = []
    final_predictions = None
    best_predictions = None
    best_states: dict[str, dict[str, torch.Tensor]] | None = None
    best_metric = float("inf")
    epochs_without_improvement = 0
    for epoch in tqdm(range(args.epochs), desc="train_tgn_rkhs", unit="epoch"):
        tgn.train()
        latent_to_rkhs.train()
        fusion_norm.train()
        fusion_head.train()
        tgn.reset_state()
        optimizer.zero_grad()

        h_values = []
        z_g_values = []
        predictions = []
        for window in tqdm(dataset.windows, desc=f"epoch_{epoch}_windows", unit="window", leave=False):
            h_t = tgn.encode_window(window.src.to(device), window.dst.to(device), window.t_start.to(device), window.duration.to(device))
            z_g_t = latent_to_rkhs(h_t)
            fused = fusion_norm(torch.cat([h_t, z_g_t], dim=0))
            y_hat = fusion_head(fused).view(())
            h_values.append(h_t)
            z_g_values.append(z_g_t)
            predictions.append(y_hat)
            tgn.detach_memory()
        assert len(predictions) == len(dataset.windows), "prediction count mismatch with windows"

        h_tensor = torch.stack(h_values)
        rkhs_tensor = dataset.rkhs_g_t.to(device)
        latent_rkhs_tensor = torch.stack(z_g_values)
        y_tensor = torch.stack(predictions)
        y_target = dataset.y_t.to(device)

        task_loss = mse(y_tensor.index_select(0, train_index), y_target.index_select(0, train_index))
        g_constraint = pointwise_alignment_loss(latent_rkhs_tensor, rkhs_tensor, train_ids)
        total_loss = task_loss + args.lambda_g * g_constraint
        total_loss.backward()
        optimizer.step()

        train_rmse = rmse_on_indices(y_tensor.detach(), y_target, train_ids)
        history_row = {
            "epoch": epoch,
            "total_loss": float(total_loss.item()),
            "task_loss": float(task_loss.item()),
            "g_constraint": float(g_constraint.item()),
            "train_rmse": float(train_rmse),
        }
        if len(val_ids) > 0:
            val_rmse = rmse_on_indices(y_tensor.detach(), y_target, val_ids)
            history_row["val_rmse"] = float(val_rmse)
        history.append(history_row)
        final_predictions = y_tensor.detach().cpu().numpy()

        monitor_metric = float(history_row["val_rmse"]) if "val_rmse" in history_row else float(history_row["train_rmse"])
        if monitor_metric + args.early_stopping_min_delta < best_metric:
            best_metric = monitor_metric
            epochs_without_improvement = 0
            best_predictions = final_predictions.copy()
            best_states = {
                "tgn": copy.deepcopy(tgn.state_dict()),
                "latent_to_rkhs": copy.deepcopy(latent_to_rkhs.state_dict()),
                "fusion_norm": copy.deepcopy(fusion_norm.state_dict()),
                "fusion_head": copy.deepcopy(fusion_head.state_dict()),
            }
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
    history_path = output_dir / f"tgn_rkhs_constraint_history{suffix}.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    if best_states is not None:
        tgn.load_state_dict(best_states["tgn"])
        latent_to_rkhs.load_state_dict(best_states["latent_to_rkhs"])
        fusion_norm.load_state_dict(best_states["fusion_norm"])
        fusion_head.load_state_dict(best_states["fusion_head"])
    final_to_save = best_predictions if best_predictions is not None else final_predictions
    if final_to_save is not None:
        pd.DataFrame(
            {
                "t": list(range(len(final_to_save))),
                "y_true": dataset.y_t.cpu().numpy(),
                "y_pred": final_to_save,
                "split": build_split_column(len(final_to_save), train_ids, val_ids, test_ids),
                "model": "tgn_rkhs_constraint",
            }
        ).to_csv(output_dir / f"tgn_rkhs_constraint_predictions{suffix}.csv", index=False)
    torch.save(
        {
            "tgn": tgn.state_dict(),
            "latent_to_rkhs": latent_to_rkhs.state_dict(),
            "fusion_norm": fusion_norm.state_dict(),
            "fusion_head": fusion_head.state_dict(),
        },
        weight_dir / f"tgn_rkhs_constraint{suffix}.pt",
    )
    print(f"Saved history: {history_path}")
    print(f"Saved weights: {weight_dir / f'tgn_rkhs_constraint{suffix}.pt'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TGN + RKHS + topology constraint model.")
    parser.add_argument("--dataset-key", default=DEFAULT_DATASET_KEY)
    parser.add_argument("--window-seconds", type=float, default=2700.0)
    parser.add_argument("--stride-seconds", type=float, default=1350.0)
    parser.add_argument("--min-events-per-window", type=int, default=100)
    parser.add_argument("--memory-dim", type=int, default=64)
    parser.add_argument("--time-dim", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lambda-g", type=float, default=0.7)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--feature-npz", type=Path, default=PROJECT_ROOT / "temp" / "pilot_primary_school" / "features.npz")
    parser.add_argument("--preprocessed-cache-dir", type=Path, default=None)
    parser.add_argument("--output-tag", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--fusion-hidden-dim", type=int, default=64)
    parser.add_argument("--rkhs-proj-hidden-dim", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
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

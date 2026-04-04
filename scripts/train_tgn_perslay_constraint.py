from __future__ import annotations

import argparse
import copy
import json
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
from src.topology.perslay_module import PersLayModule  # noqa: E402
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
    raw_diagrams = load_raw_diagrams(args, len(dataset.y_t))
    train_ids, val_ids, test_ids = chronological_split_indices(
        len(dataset.y_t),
        TemporalSplitConfig(train_fraction=args.train_fraction, val_fraction=args.val_fraction),
    )
    train_index = torch.tensor(train_ids, dtype=torch.long, device=device)

    tgn = WindowTGNRegressor(num_nodes=dataset.num_nodes, memory_dim=args.memory_dim, time_dim=args.time_dim).to(device)
    perslay_h_dim = max(1, args.memory_dim // 2)
    perslay_h0 = PersLayModule(latent_dim=perslay_h_dim).to(device)
    perslay_h1 = PersLayModule(latent_dim=args.memory_dim - perslay_h_dim).to(device)
    fusion_head = nn.Sequential(
        nn.Linear(args.memory_dim + args.memory_dim, args.fusion_hidden_dim),
        nn.ReLU(),
        nn.Linear(args.fusion_hidden_dim, 1),
        nn.Sigmoid(),
    ).to(device)
    params = list(tgn.parameters()) + list(perslay_h0.parameters()) + list(perslay_h1.parameters()) + list(fusion_head.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    mse = nn.MSELoss()
    history: list[dict[str, float]] = []
    final_predictions = None
    best_predictions = None
    best_states: dict[str, dict[str, torch.Tensor]] | None = None
    best_metric = float("inf")
    epochs_without_improvement = 0
    for epoch in tqdm(range(args.epochs), desc="train_tgn_perslay", unit="epoch"):
        tgn.train()
        perslay_h0.train()
        perslay_h1.train()
        fusion_head.train()
        tgn.reset_state()
        optimizer.zero_grad()

        h_values = []
        z_d_values = []
        y_pred = []
        for idx, window in enumerate(tqdm(dataset.windows, desc=f"epoch_{epoch}_windows", unit="window", leave=False)):
            h_t = tgn.encode_window(window.src.to(device), window.dst.to(device), window.t_start.to(device), window.duration.to(device))
            diag_h0 = diagram_tensor(raw_diagrams[idx]["h0"], device)
            diag_h1 = diagram_tensor(raw_diagrams[idx]["h1"], device)
            z_h0 = perslay_h0(diag_h0)
            z_h1 = perslay_h1(diag_h1)
            z_d = torch.cat([z_h0, z_h1], dim=0)
            fused = torch.cat([h_t, z_d], dim=0)
            y_hat = fusion_head(fused).view(())
            h_values.append(h_t)
            z_d_values.append(z_d)
            y_pred.append(y_hat)
            tgn.detach_memory()
        assert len(y_pred) == len(dataset.windows), "prediction count mismatch with windows"

        h_tensor = torch.stack(h_values)
        z_d_tensor = torch.stack(z_d_values)
        y_tensor = torch.stack(y_pred)
        y_target = dataset.y_t.to(device)

        task_loss = mse(y_tensor.index_select(0, train_index), y_target.index_select(0, train_index))
        d_constraint = pointwise_alignment_loss(h_tensor, z_d_tensor, train_ids)
        total_loss = task_loss + args.lambda_d * d_constraint
        total_loss.backward()
        optimizer.step()

        train_rmse = rmse_on_indices(y_tensor.detach(), y_target, train_ids)
        history_row = {
            "epoch": epoch,
            "total_loss": float(total_loss.item()),
            "task_loss": float(task_loss.item()),
            "d_constraint": float(d_constraint.item()),
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
                "perslay_h0": copy.deepcopy(perslay_h0.state_dict()),
                "perslay_h1": copy.deepcopy(perslay_h1.state_dict()),
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
    history_path = output_dir / f"tgn_perslay_constraint_history{suffix}.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    if best_states is not None:
        tgn.load_state_dict(best_states["tgn"])
        perslay_h0.load_state_dict(best_states["perslay_h0"])
        perslay_h1.load_state_dict(best_states["perslay_h1"])
        fusion_head.load_state_dict(best_states["fusion_head"])
    final_to_save = best_predictions if best_predictions is not None else final_predictions
    if final_to_save is not None:
        pd.DataFrame(
            {
                "t": list(range(len(final_to_save))),
                "y_true": dataset.y_t.cpu().numpy(),
                "y_pred": final_to_save,
                "split": build_split_column(len(final_to_save), train_ids, val_ids, test_ids),
                "model": "tgn_perslay_constraint",
            }
        ).to_csv(output_dir / f"tgn_perslay_constraint_predictions{suffix}.csv", index=False)
    torch.save(
        {
            "tgn": tgn.state_dict(),
            "perslay_h0": perslay_h0.state_dict(),
            "perslay_h1": perslay_h1.state_dict(),
            "fusion_head": fusion_head.state_dict(),
        },
        weight_dir / f"tgn_perslay_constraint{suffix}.pt",
    )
    print(f"Saved history: {history_path}")
    print(f"Saved weights: {weight_dir / f'tgn_perslay_constraint{suffix}.pt'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TGN + PersLay + topology constraint model.")
    parser.add_argument("--dataset-key", default=DEFAULT_DATASET_KEY)
    parser.add_argument("--window-seconds", type=float, default=2700.0)
    parser.add_argument("--stride-seconds", type=float, default=1350.0)
    parser.add_argument("--min-events-per-window", type=int, default=100)
    parser.add_argument("--memory-dim", type=int, default=64)
    parser.add_argument("--time-dim", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lambda-d", type=float, default=0.3)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--feature-npz", type=Path, default=PROJECT_ROOT / "temp" / "pilot_primary_school" / "features.npz")
    parser.add_argument("--preprocessed-cache-dir", type=Path, default=None)
    parser.add_argument("--raw-diagram-cache-path", type=Path, default=None)
    parser.add_argument("--output-tag", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--fusion-hidden-dim", type=int, default=64)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise ValueError("cuda requested but torch.cuda.is_available() is False")
    return torch.device("cuda")


def load_raw_diagrams(args: argparse.Namespace, expected_rows: int) -> list[dict[str, list[list[float]]]]:
    if args.raw_diagram_cache_path is not None:
        path = args.raw_diagram_cache_path
    elif args.preprocessed_cache_dir is not None:
        path = args.preprocessed_cache_dir / "raw_persistence_diagrams.jsonl"
    else:
        raise ValueError("raw diagram cache requires --raw-diagram-cache-path or --preprocessed-cache-dir")
    if not path.exists():
        raise ValueError(f"missing raw diagram cache: {path}")
    rows: list[dict[str, list[list[float]]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            rows.append({"h0": record.get("h0", []), "h1": record.get("h1", [])})
    if len(rows) < expected_rows:
        raise ValueError(f"raw diagram rows {len(rows)} smaller than expected {expected_rows}")
    return rows[:expected_rows]


def diagram_tensor(points: list[list[float]], device: torch.device) -> torch.Tensor:
    if len(points) == 0:
        return torch.zeros((0, 2), dtype=torch.float32, device=device)
    return torch.tensor(points, dtype=torch.float32, device=device).view(-1, 2)


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

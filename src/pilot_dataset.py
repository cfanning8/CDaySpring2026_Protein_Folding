from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.dataloaders import load_all_datasets
from src.edge_preparation import extract_temporal_events_for_dataset


@dataclass(frozen=True)
class WindowEvents:
    src: torch.Tensor
    dst: torch.Tensor
    t_start: torch.Tensor
    duration: torch.Tensor


@dataclass(frozen=True)
class PilotDataset:
    windows: list[WindowEvents]
    y_t: torch.Tensor
    d_t: torch.Tensor
    g_t: torch.Tensor
    rkhs_g_t: torch.Tensor
    num_nodes: int


def load_primary_school_pilot_dataset(
    project_root: Path,
    dataset_key: str,
    window_seconds: float,
    stride_seconds: float,
    min_events_per_window: int,
    feature_npz_path: Path | None = None,
    preprocessed_cache_dir: Path | None = None,
) -> PilotDataset:
    if preprocessed_cache_dir is not None and (preprocessed_cache_dir / "windows.npz").exists():
        windows, num_nodes = _load_cached_windows(preprocessed_cache_dir)
    else:
        datasets = load_all_datasets(project_root / "data")
        if dataset_key not in datasets:
            raise ValueError(f"dataset_key not found: {dataset_key}")
        temporal = extract_temporal_events_for_dataset(dataset_key, datasets[dataset_key])
        if temporal is None:
            raise ValueError("dataset does not have temporal extraction")

        events = temporal.events.copy()
        windows_df = _build_windows(events, window_seconds, stride_seconds, min_events_per_window)
        if len(windows_df) < 2:
            raise ValueError("need at least two windows")

        nodes = sorted(set(events["source"]).union(set(events["target"])))
        node_index = {node: idx for idx, node in enumerate(nodes)}
        windows = []
        for window in windows_df:
            src_idx = torch.tensor([node_index[str(value)] for value in window["source"]], dtype=torch.long)
            dst_idx = torch.tensor([node_index[str(value)] for value in window["target"]], dtype=torch.long)
            t_start = torch.tensor(window["t_start"].to_numpy(dtype=np.int64), dtype=torch.long)
            duration = torch.tensor(window["duration_seconds"].to_numpy(dtype=np.float32), dtype=torch.float32)
            windows.append(WindowEvents(src=src_idx, dst=dst_idx, t_start=t_start, duration=duration))
        num_nodes = len(nodes)

    feature_path = feature_npz_path if feature_npz_path is not None else project_root / "temp" / "pilot_primary_school" / "features.npz"
    if not feature_path.exists():
        raise ValueError(f"missing pilot features file: {feature_path}")
    arrays = np.load(feature_path)
    return PilotDataset(
        windows=windows[: arrays["y_t"].shape[0]],
        y_t=torch.tensor(arrays["y_t"], dtype=torch.float32),
        d_t=torch.tensor(arrays["d_t"], dtype=torch.float32),
        g_t=torch.tensor(arrays["g_t"], dtype=torch.float32),
        rkhs_g_t=torch.tensor(arrays["rkhs_g_t"], dtype=torch.float32),
        num_nodes=num_nodes,
    )


def _build_windows(
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
        window = events.loc[mask, ["source", "target", "t_start", "duration_seconds"]].copy()
        if len(window) >= min_events:
            windows.append(window.sort_values("t_start"))
        cursor += stride_seconds
    return windows


def _load_cached_windows(cache_dir: Path) -> tuple[list[WindowEvents], int]:
    cache_file = cache_dir / "windows.npz"
    if not cache_file.exists():
        raise ValueError(f"missing cached windows file: {cache_file}")
    arrays = np.load(cache_file)
    src = arrays["src"].astype(np.int64)
    dst = arrays["dst"].astype(np.int64)
    t_start = arrays["t_start"].astype(np.int64)
    duration = arrays["duration"].astype(np.float32)
    window_id = arrays["window_id"].astype(np.int64)
    num_windows = int(np.asarray(arrays["num_windows"]).reshape(-1)[0])
    num_nodes = int(np.asarray(arrays["num_nodes"]).reshape(-1)[0])
    windows: list[WindowEvents] = []
    for wid in range(num_windows):
        mask = window_id == wid
        windows.append(
            WindowEvents(
                src=torch.tensor(src[mask], dtype=torch.long),
                dst=torch.tensor(dst[mask], dtype=torch.long),
                t_start=torch.tensor(t_start[mask], dtype=torch.long),
                duration=torch.tensor(duration[mask], dtype=torch.float32),
            )
        )
    return windows, num_nodes

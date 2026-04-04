from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_cached_windows(cache_dir: Path) -> list[pd.DataFrame]:
    npz_path = cache_dir / "windows.npz"
    if not npz_path.exists():
        raise ValueError(f"cached windows file missing: {npz_path}")
    arrays = np.load(npz_path)
    src = arrays["src"].astype(np.int64)
    dst = arrays["dst"].astype(np.int64)
    t_start = arrays["t_start"].astype(np.float64)
    duration = arrays["duration"].astype(np.float64)
    window_id = arrays["window_id"].astype(np.int64)
    num_windows = int(np.asarray(arrays["num_windows"]).reshape(-1)[0])
    windows: list[pd.DataFrame] = []
    for wid in range(num_windows):
        mask = window_id == wid
        if not np.any(mask):
            continue
        window = pd.DataFrame(
            {
                "source": src[mask].astype(str),
                "target": dst[mask].astype(str),
                "t_start": t_start[mask],
                "duration_seconds": duration[mask],
            }
        ).sort_values("t_start")
        windows.append(window)
    return windows


def aggregate_window_edges(window_events: pd.DataFrame) -> pd.DataFrame:
    ordered_source = np.where(
        window_events["source"].astype(str).to_numpy() <= window_events["target"].astype(str).to_numpy(),
        window_events["source"].astype(str).to_numpy(),
        window_events["target"].astype(str).to_numpy(),
    )
    ordered_target = np.where(
        window_events["source"].astype(str).to_numpy() <= window_events["target"].astype(str).to_numpy(),
        window_events["target"].astype(str).to_numpy(),
        window_events["source"].astype(str).to_numpy(),
    )
    edges = pd.DataFrame(
        {
            "source": ordered_source,
            "target": ordered_target,
            "duration_seconds": pd.to_numeric(window_events["duration_seconds"], errors="coerce").fillna(0.0),
        }
    )
    edges = edges[(edges["source"] != edges["target"]) & (edges["duration_seconds"] > 0)]
    if edges.empty:
        raise ValueError("window produced no positive-duration edges")
    return (
        edges.groupby(["source", "target"], as_index=False)
        .agg(duration_seconds=("duration_seconds", "sum"))
        .sort_values(["source", "target"])
    )

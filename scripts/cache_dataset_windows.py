from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataloaders import load_all_datasets  # noqa: E402
from src.edge_preparation import extract_temporal_events_for_dataset  # noqa: E402


DATASET_CONFIGS = {
    "Thiers13": {
        "key_exact": r"high_school\HighSchool2013_proximity_net.csv\High-School_data_2013.csv",
        "window_seconds": 45 * 60,
        "stride_seconds": 45 * 60,
        "min_events": 80,
        "mode": "session",
    },
    "LyonSchool": {
        "key_exact": r"primary_school\primaryschool.csv\primaryschool.csv",
        "window_seconds": 45 * 60,
        "stride_seconds": 45 * 60,
        "min_events": 80,
        "mode": "session",
    },
    "InVS15": {
        "key_exact": r"workplace\workplace_InVS15_tij.dat\tij_InVS15.dat",
        "window_seconds": 60 * 60,
        "stride_seconds": 60 * 60,
        "min_events": 60,
        "mode": "session",
    },
    "HT2009": {
        "key_exact": r"hypertext\ht2009_contact_list.dat\ht09_contact_list.dat",
        "window_seconds": 30 * 60,
        "stride_seconds": 30 * 60,
        "min_events": 60,
        "mode": "session",
    },
    "Infectious": {
        "key_prefix": r"infectious\sciencegallery_infectious_contacts\listcontacts_",
        "window_seconds": 24 * 60 * 60,
        "stride_seconds": 24 * 60 * 60,
        "min_events": 20,
        "mode": "day_independent",
    },
}


def main() -> None:
    args = parse_args()
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"dataset must be one of: {', '.join(DATASET_CONFIGS.keys())}")
    cfg = DATASET_CONFIGS[args.dataset]

    out_dir = PROJECT_ROOT / "data" / "preprocessed" / args.dataset
    npz_path = out_dir / "windows.npz"
    if npz_path.exists() and not args.force:
        print(f"[cache] windows already exist for {args.dataset}: {npz_path}")
        return

    datasets = load_all_datasets(PROJECT_ROOT / "data")
    events = collect_events(args.dataset, cfg, datasets)
    if events.empty:
        raise ValueError(f"no temporal events extracted for dataset {args.dataset}")

    windows = build_windows(
        events=events,
        window_seconds=float(cfg["window_seconds"]),
        stride_seconds=float(cfg["stride_seconds"]),
        min_events=int(cfg["min_events"]),
        mode=str(cfg.get("mode", "session")),
    )
    if len(windows) == 0:
        raise ValueError(f"dataset {args.dataset} produced no valid windows")
    if len(windows) < 2:
        print(f"[warn] {args.dataset} produced only {len(windows)} window; cache will still be written")

    write_cache(
        out_dir=out_dir,
        windows=windows,
        dataset_name=args.dataset,
        window_seconds=float(cfg["window_seconds"]),
        stride_seconds=float(cfg["stride_seconds"]),
    )
    print(f"[ok] cached windows for {args.dataset} -> {npz_path}")


def collect_events(dataset_name: str, cfg: dict[str, object], datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    keys: list[str] = []
    if "key_exact" in cfg:
        exact = str(cfg["key_exact"])
        if exact not in datasets:
            raise ValueError(f"dataset key missing for {dataset_name}: {exact}")
        keys = [exact]
    else:
        prefix = str(cfg["key_prefix"])
        keys = sorted([key for key in datasets.keys() if key.startswith(prefix)])
        if not keys:
            raise ValueError(f"no keys found for prefix {prefix}")

    chunks = []
    for file_idx, key in enumerate(tqdm(keys, desc=f"{dataset_name}:extract", unit="file")):
        temporal = extract_temporal_events_for_dataset(key, datasets[key])
        if temporal is None or temporal.events.empty:
            continue
        chunk = temporal.events[["source", "target", "t_start", "duration_seconds"]].copy()
        if dataset_name == "Infectious":
            chunk["t_start"] = pd.to_numeric(chunk["t_start"], errors="coerce").fillna(0.0) + float(file_idx * 86400.0)
        chunks.append(chunk)
    if not chunks:
        return pd.DataFrame(columns=["source", "target", "t_start", "duration_seconds"])
    merged = pd.concat(chunks, ignore_index=True)
    merged["source"] = merged["source"].astype(str)
    merged["target"] = merged["target"].astype(str)
    merged["t_start"] = pd.to_numeric(merged["t_start"], errors="coerce").fillna(0.0)
    merged["duration_seconds"] = pd.to_numeric(merged["duration_seconds"], errors="coerce").fillna(0.0)
    merged = merged[(merged["source"] != merged["target"]) & (merged["duration_seconds"] > 0)]
    merged = merged.sort_values("t_start").reset_index(drop=True)
    return merged


def build_windows(
    events: pd.DataFrame,
    window_seconds: float,
    stride_seconds: float,
    min_events: int,
    mode: str,
) -> list[pd.DataFrame]:
    if mode == "day_independent":
        return build_day_windows(events, min_events)

    t_min = float(events["t_start"].min())
    t_max = float(events["t_start"].max())
    if t_max <= t_min:
        return []
    approx = int(max(0, np.floor((t_max - t_min - window_seconds) / stride_seconds) + 1))
    windows: list[pd.DataFrame] = []
    cursor = t_min
    for _ in tqdm(range(max(approx, 1)), desc="build_windows", unit="window"):
        if cursor + window_seconds > t_max + 1e-9:
            break
        mask = (events["t_start"] >= cursor) & (events["t_start"] < cursor + window_seconds)
        window = events.loc[mask, ["source", "target", "t_start", "duration_seconds"]].copy()
        if len(window) >= min_events:
            windows.append(window.sort_values("t_start"))
        cursor += stride_seconds
    return windows


def build_day_windows(events: pd.DataFrame, min_events: int) -> list[pd.DataFrame]:
    t_min = float(events["t_start"].min())
    t_max = float(events["t_start"].max())
    day_start = int(np.floor(t_min / 86400.0))
    day_end = int(np.floor(t_max / 86400.0))
    windows: list[pd.DataFrame] = []
    for day in tqdm(range(day_start, day_end + 1), desc="build_day_windows", unit="day"):
        lo = float(day * 86400.0)
        hi = float((day + 1) * 86400.0)
        mask = (events["t_start"] >= lo) & (events["t_start"] < hi)
        window = events.loc[mask, ["source", "target", "t_start", "duration_seconds"]].copy()
        if len(window) >= min_events:
            windows.append(window.sort_values("t_start"))
    return windows


def write_cache(
    out_dir: Path,
    windows: list[pd.DataFrame],
    dataset_name: str,
    window_seconds: float,
    stride_seconds: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    node_ids = sorted(set(pd.concat(windows, ignore_index=True)["source"].astype(str)).union(set(pd.concat(windows, ignore_index=True)["target"].astype(str))))
    node_to_int = {node: idx for idx, node in enumerate(node_ids)}
    pd.DataFrame({"node": node_ids, "node_id": list(range(len(node_ids)))}).to_csv(out_dir / "node_index.csv", index=False)

    src_parts = []
    dst_parts = []
    t_parts = []
    dur_parts = []
    wid_parts = []
    start_rows = []
    for wid, window in enumerate(tqdm(windows, desc=f"{dataset_name}:serialize", unit="window")):
        src = window["source"].map(node_to_int).to_numpy(dtype=np.int32)
        dst = window["target"].map(node_to_int).to_numpy(dtype=np.int32)
        t_start = window["t_start"].to_numpy(dtype=np.float64)
        duration = window["duration_seconds"].to_numpy(dtype=np.float32)
        src_parts.append(src)
        dst_parts.append(dst)
        t_parts.append(t_start)
        dur_parts.append(duration)
        wid_parts.append(np.full(len(window), wid, dtype=np.int32))
        start_rows.append(
            {
                "window_id": wid,
                "window_start": float(t_start.min()),
                "window_end": float(t_start.max()),
                "events": int(len(window)),
            }
        )

    src_all = np.concatenate(src_parts).astype(np.int32)
    dst_all = np.concatenate(dst_parts).astype(np.int32)
    t_all = np.concatenate(t_parts).astype(np.float64)
    dur_all = np.concatenate(dur_parts).astype(np.float32)
    wid_all = np.concatenate(wid_parts).astype(np.int32)
    np.savez_compressed(
        out_dir / "windows.npz",
        src=src_all,
        dst=dst_all,
        t_start=t_all,
        duration=dur_all,
        window_id=wid_all,
        num_windows=np.array([len(windows)], dtype=np.int32),
        num_nodes=np.array([len(node_ids)], dtype=np.int32),
        window_seconds=np.array([window_seconds], dtype=np.float64),
        stride_seconds=np.array([stride_seconds], dtype=np.float64),
    )
    pd.DataFrame(start_rows).to_csv(out_dir / "window_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache temporal windows for one dataset into data/preprocessed.")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataloaders import load_all_datasets  # noqa: E402
from src.edge_preparation import extract_temporal_events_for_dataset  # noqa: E402


WINDOW_SECONDS = {
    "school": 3600,
    "workplace": 3600,
    "ward": 8 * 3600,
    "conference": 1800,
    "village": 3600,
    "default": 3600,
}


def main() -> None:
    datasets = load_all_datasets(PROJECT_ROOT / "data")
    audit_rows: list[dict[str, object]] = []
    issues: list[str] = []

    for dataset_key, frame in datasets.items():
        temporal = extract_temporal_events_for_dataset(dataset_key, frame)
        setting = infer_setting(dataset_key)
        window_seconds = WINDOW_SECONDS.get(setting, WINDOW_SECONDS["default"])

        if temporal is None:
            audit_rows.append(
                {
                    "dataset_key": dataset_key,
                    "setting": setting,
                    "temporal_ready": False,
                    "events": 0,
                    "nodes": 0,
                    "time_span_seconds": 0.0,
                    "window_seconds": window_seconds,
                    "windows_available": 0,
                    "notes": "no temporal extraction rule",
                }
            )
            continue

        events = temporal.events
        node_count = len(set(events["source"]).union(set(events["target"])))
        time_span = float(events["t_start"].max() - events["t_start"].min())
        windows = int(time_span // window_seconds) + 1 if time_span >= 0 else 0

        notes = []
        if node_count < 20:
            notes.append("small node count")
            issues.append(f"{dataset_key}: only {node_count} nodes for temporal epidemic modeling")
        if windows < 8:
            notes.append("few windows")
            issues.append(f"{dataset_key}: only {windows} windows for rolling prediction")
        if time_span <= window_seconds:
            notes.append("short time span")
            issues.append(f"{dataset_key}: time span {time_span:.1f}s is too short for robust temporal learning")

        audit_rows.append(
            {
                "dataset_key": dataset_key,
                "setting": setting,
                "temporal_ready": True,
                "events": len(events),
                "nodes": node_count,
                "time_span_seconds": time_span,
                "window_seconds": window_seconds,
                "windows_available": windows,
                "notes": "; ".join(notes),
            }
        )

    audit = pd.DataFrame(audit_rows).sort_values(["temporal_ready", "dataset_key"], ascending=[False, True])
    temp_dir = PROJECT_ROOT / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    audit_path = temp_dir / "pipeline_readiness_audit.csv"
    issues_path = temp_dir / "pipeline_critical_issues.txt"
    audit.to_csv(audit_path, index=False)
    issues_path.write_text("\n".join(issues) + ("\n" if issues else ""), encoding="utf-8")

    print(f"Audit rows: {len(audit)}")
    print(f"Audit file: {audit_path}")
    print(f"Critical issues: {len(issues)}")
    if issues:
        for issue in issues[:40]:
            print(f"- {issue}")


def infer_setting(dataset_key: str) -> str:
    lower_key = dataset_key.lower()
    if "school" in lower_key:
        return "school"
    if "workplace" in lower_key:
        return "workplace"
    if "ward" in lower_key or "hospital" in lower_key:
        return "ward"
    if "sfhh" in lower_key or "hypertext" in lower_key:
        return "conference"
    if "kenya" in lower_key or "malawi" in lower_key:
        return "village"
    return "default"


if __name__ == "__main__":
    main()

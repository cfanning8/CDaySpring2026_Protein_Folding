from __future__ import annotations

from pathlib import Path
import re
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataloaders import load_all_datasets
from src.edge_preparation import default_canonical_output_dir, prepare_edges_for_dataset


def main() -> None:
    data_dir = PROJECT_ROOT / "data"
    output_dir = default_canonical_output_dir(PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_all_datasets(data_dir)
    summary_rows: list[dict[str, object]] = []

    for key, frame in datasets.items():
        result = prepare_edges_for_dataset(key, frame)
        if result is None:
            continue

        output_name = _safe_name(key) + ".csv"
        output_path = output_dir / output_name
        result.canonical_edges.to_csv(output_path, index=False)

        summary_rows.append(
            {
                "dataset_key": key,
                "rule": result.rule,
                "edge_rows": len(result.canonical_edges),
                "total_duration_seconds": float(result.canonical_edges["duration_seconds"].sum()),
                "output_file": str(output_path.relative_to(PROJECT_ROOT)),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("dataset_key")
    summary_path = PROJECT_ROOT / "temp" / "canonical_edges_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Wrote {len(summary_rows)} canonical edge files to {output_dir}")
    print(f"Wrote summary to {summary_path}")


def _safe_name(dataset_key: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", dataset_key)


if __name__ == "__main__":
    main()

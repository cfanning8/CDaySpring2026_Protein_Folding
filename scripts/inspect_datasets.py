from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataloaders import load_all_datasets


def main() -> None:
    data_dir = PROJECT_ROOT / "data"
    datasets = load_all_datasets(data_dir)

    print(f"Loaded {len(datasets)} dataset tables")
    for key, frame in datasets.items():
        print(f"{key}: rows={len(frame)} cols={len(frame.columns)}")


if __name__ == "__main__":
    main()

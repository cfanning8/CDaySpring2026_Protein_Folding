from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataloaders import load_all_datasets


def main() -> None:
    data_dir = PROJECT_ROOT / "data"

    datasets = load_all_datasets(data_dir)
    failures = []

    for key, frame in datasets.items():
        if frame.empty:
            failures.append(f"{key}: empty table")
            continue
        if len(frame.columns) == 0:
            failures.append(f"{key}: zero columns")

    if failures:
        print("Loader smoke test failures:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print(f"Smoke test passed for {len(datasets)} dataset tables.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.protein.dataset_policy import FROZEN_SELECTION_SPEC_ID, rcsb_policy_query_group  # noqa: E402
from src.protein.rcsb_client import iter_rcsb_entry_identifiers  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch PDB IDs from RCSB Search matching the frozen policy.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--max-ids", type=int, default=None)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    query = rcsb_policy_query_group()
    count = 0
    with args.out.open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle)
        writer.writerow(["selection_spec_id", "pdb_id"])
        for identifier in iter_rcsb_entry_identifiers(
            query,
            page_size=int(args.page_size),
            max_ids=args.max_ids,
        ):
            writer.writerow([FROZEN_SELECTION_SPEC_ID, identifier.upper()])
            count += 1

    print(f"wrote={args.out.resolve()} rows={count}")


if __name__ == "__main__":
    main()

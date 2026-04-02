from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.protein.cath_clf import iter_cath_clf_rows  # noqa: E402
from src.protein.dataset_policy import FROZEN_SELECTION_SPEC_ID  # noqa: E402
from src.protein.eligible_domains import domain_passes_policy  # noqa: E402


def _load_rcsb_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open("r", encoding="ascii", errors="strict") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "pdb_id" not in reader.fieldnames:
            raise ValueError("rcsb_ids csv must include a pdb_id column")
        for row in reader:
            pdb = row.get("pdb_id", "").strip().upper()
            if pdb:
                ids.add(pdb)
    if not ids:
        raise ValueError("rcsb id set is empty")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream CATH CLF rows and write domains that pass the frozen eligibility policy.",
    )
    parser.add_argument("--rcsb-ids", type=Path, required=True)
    parser.add_argument("--cath-domain-list", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rcsb_ids = _load_rcsb_ids(args.rcsb_ids)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "selection_spec_id",
        "domain_id",
        "pdb_id",
        "chain_id",
        "superfamily",
        "s35_cluster",
        "domain_length",
        "cath_resolution",
    ]

    written = 0
    with args.out.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in iter_cath_clf_rows(args.cath_domain_list):
            if not domain_passes_policy(row, rcsb_ids):
                continue
            writer.writerow(
                {
                    "selection_spec_id": FROZEN_SELECTION_SPEC_ID,
                    "domain_id": row["domain_id"],
                    "pdb_id": row["pdb_id"],
                    "chain_id": row["chain_id"],
                    "superfamily": row["superfamily"],
                    "s35_cluster": row["s35_cluster"],
                    "domain_length": row["domain_length"],
                    "cath_resolution": row["cath_resolution"],
                }
            )
            written += 1

    print(f"wrote={args.out.resolve()} rows={written}")


if __name__ == "__main__":
    main()

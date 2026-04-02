from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.protein.dataset_policy import FROZEN_SELECTION_SPEC_ID, rcsb_policy_query_group  # noqa: E402
from src.protein.pdb_manifest_schema import PDB_MANIFEST_COLUMNS  # noqa: E402
from src.protein.rcsb_client import iter_rcsb_entry_identifiers  # noqa: E402
from src.protein.rcsb_data_api import build_pdb_manifest_row, normalize_manifest_row  # noqa: E402


def _read_pdb_ids(path: Path) -> list[str]:
    out: list[str] = []
    with path.open("r", encoding="ascii", errors="strict") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "pdb_id" not in reader.fieldnames:
            raise ValueError("input csv must contain pdb_id column")
        for row in reader:
            pid = (row.get("pdb_id") or "").strip().upper()
            if pid:
                out.append(pid)
    if not out:
        raise ValueError("no pdb_id values in input")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pass 1: metadata-only PDB manifest from RCSB Data API (no coordinate download).",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--entry-ids-csv",
        type=Path,
        default=None,
        help="CSV with pdb_id column (from fetch_rcsb_entry_ids.py or external).",
    )
    parser.add_argument(
        "--from-search",
        action="store_true",
        help="Ignore --entry-ids-csv and iterate search results from frozen rcsb_policy_query_group.",
    )
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--max-entries", type=int, default=None)
    parser.add_argument(
        "--only-without-exclusion",
        action="store_true",
        help="Write only rows with empty entry_exclusion_code (analysis-corpus preview).",
    )
    parser.add_argument("--sleep-s", type=float, default=0.15)
    args = parser.parse_args()

    if args.from_search:
        id_source = "search"
    elif args.entry_ids_csv is not None:
        id_source = "csv"
    else:
        raise SystemExit("provide --from-search or --entry-ids-csv PATH")
    if args.from_search and args.entry_ids_csv is not None:
        raise SystemExit("do not pass both --from-search and --entry-ids-csv")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "cdays-protein-pipeline/1.0 (metadata manifest; +https://data.rcsb.org)"})

    written = 0
    try:
        with args.out.open("w", newline="", encoding="ascii") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(PDB_MANIFEST_COLUMNS), extrasaction="ignore")
            writer.writeheader()

            if id_source == "csv":
                pdb_iter = _read_pdb_ids(args.entry_ids_csv)
                for idx, pid in enumerate(pdb_iter):
                    if args.max_entries is not None and idx >= args.max_entries:
                        break
                    row = normalize_manifest_row(
                        build_pdb_manifest_row(
                            selection_spec_id=FROZEN_SELECTION_SPEC_ID,
                            pdb_id=pid,
                            session=session,
                        )
                    )
                    if args.only_without_exclusion and row.get("entry_exclusion_code"):
                        continue
                    writer.writerow(row)
                    written += 1
                    time.sleep(float(args.sleep_s))
            else:
                query = rcsb_policy_query_group()
                for idx, identifier in enumerate(
                    iter_rcsb_entry_identifiers(
                        query,
                        page_size=int(args.page_size),
                        max_ids=args.max_entries,
                        session=session,
                    )
                ):
                    pid = str(identifier).upper()
                    row = normalize_manifest_row(
                        build_pdb_manifest_row(
                            selection_spec_id=FROZEN_SELECTION_SPEC_ID,
                            pdb_id=pid,
                            session=session,
                        )
                    )
                    if args.only_without_exclusion and row.get("entry_exclusion_code"):
                        continue
                    writer.writerow(row)
                    written += 1
                    time.sleep(float(args.sleep_s))
    finally:
        session.close()

    print(f"wrote={args.out.resolve()} rows={written}")


if __name__ == "__main__":
    main()

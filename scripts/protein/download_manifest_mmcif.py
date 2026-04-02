from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download mmCIF files for unique PDB IDs in a manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data" / "protein" / "mmcif")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    pdbs: set[str] = set()
    with args.manifest.open("r", encoding="ascii", errors="strict") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "pdb_id" not in reader.fieldnames:
            raise ValueError("manifest must include pdb_id")
        for row in reader:
            pdb = row.get("pdb_id", "").strip().upper()
            if pdb:
                pdbs.add(pdb)

    if not pdbs:
        raise SystemExit("no pdb_id values found")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    try:
        for pdb in sorted(pdbs):
            out_path = args.out_dir / f"{pdb}.cif"
            if args.skip_existing and out_path.is_file():
                print(f"skip existing: {out_path.name}")
                continue
            url = f"https://files.rcsb.org/download/{pdb}.cif"
            response = session.get(url, timeout=180)
            response.raise_for_status()
            out_path.write_bytes(response.content)
            print(f"saved: {out_path.name}")
            time.sleep(0.2)
    finally:
        session.close()


if __name__ == "__main__":
    main()

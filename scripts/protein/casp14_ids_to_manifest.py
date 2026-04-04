#!/usr/bin/env python3
"""Build casp14_manifest.csv (pdb_id column) from data/protein/manifests/casp14_pdb_ids.txt."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="CASP14 PDB id list → CSV for download_manifest_mmcif.py")
    parser.add_argument(
        "--ids",
        type=Path,
        default=PROJECT_ROOT / "data" / "protein" / "manifests" / "casp14_pdb_ids.txt",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "data" / "protein" / "manifests" / "casp14_manifest.csv",
    )
    args = parser.parse_args()

    lines = args.ids.read_text(encoding="utf-8", errors="replace").splitlines()
    pdbs: list[str] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.upper()[:4]
        if len(p) == 4 and p.isalnum():
            pdbs.append(p)

    seen: set[str] = set()
    unique = [x for x in pdbs if not (x in seen or seen.add(x))]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="ascii") as f:
        w = csv.DictWriter(f, fieldnames=["pdb_id"])
        w.writeheader()
        for p in unique:
            w.writerow({"pdb_id": p})

    print(f"wrote={args.out.resolve()} n={len(unique)}")


if __name__ == "__main__":
    main()

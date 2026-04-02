from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.protein.mmcif_io import load_ca_coords_from_mmcif  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse one mmCIF and print CA count.")
    parser.add_argument("--mmcif-path", type=Path, required=True)
    parser.add_argument("--chain-id", type=str, default=None)
    args = parser.parse_args()

    chain = args.chain_id
    if chain is not None and len(chain) != 1:
        raise SystemExit("--chain-id must be a single character")

    coords = load_ca_coords_from_mmcif(args.mmcif_path, chain_id=chain)
    print(f"path={args.mmcif_path.resolve()}")
    print(f"n_ca={coords.shape[0]}")


if __name__ == "__main__":
    main()

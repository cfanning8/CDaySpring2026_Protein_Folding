from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="List mmCIF files under data/protein/mmcif.")
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=None,
        help="Directory to scan (default: <repo>/data/protein/mmcif)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    corpus_dir = args.corpus_dir if args.corpus_dir is not None else project_root / "data" / "protein" / "mmcif"
    corpus_dir = corpus_dir.resolve()

    if not corpus_dir.is_dir():
        raise SystemExit(f"corpus directory does not exist: {corpus_dir}")

    paths: list[Path] = []
    for pattern in ("*.cif", "*.mmcif"):
        paths.extend(sorted(corpus_dir.glob(pattern)))

    unique = sorted({p.resolve() for p in paths})
    print(f"corpus_dir={corpus_dir}")
    print(f"count={len(unique)}")
    for path in unique:
        print(path)


if __name__ == "__main__":
    main()

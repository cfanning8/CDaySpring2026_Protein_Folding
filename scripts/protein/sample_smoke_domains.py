from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.protein.dataset_policy import SMOKE_SAMPLE_RANDOM_SEED  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratified random smoke sample over CATH superfamilies.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-superfamilies", type=int, default=10)
    parser.add_argument("--per-superfamily", type=int, default=3)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    if df.empty:
        raise SystemExit("manifest is empty")

    required = {"superfamily", "pdb_id", "chain_id", "domain_id"}
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"manifest missing columns: {sorted(missing)}")

    groups = {name: sub.reset_index(drop=True) for name, sub in df.groupby("superfamily", sort=False)}
    superfamilies = list(groups.keys())

    rng = np.random.RandomState(SMOKE_SAMPLE_RANDOM_SEED)
    rng.shuffle(superfamilies)

    picked: list[str] = []
    for name in superfamilies:
        if len(groups[name]) > 0:
            picked.append(name)
        if len(picked) >= int(args.target_superfamilies):
            break

    if len(picked) < int(args.target_superfamilies):
        raise SystemExit("not enough superfamilies with at least one domain")

    parts = []
    for name in picked:
        sub = groups[name]
        k = min(int(args.per_superfamily), len(sub))
        idx = rng.choice(len(sub), size=k, replace=False)
        parts.append(sub.iloc[sorted(idx.tolist())])

    out = pd.concat(parts, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"wrote={args.out.resolve()} rows={len(out)} superfamilies={len(picked)}")


if __name__ == "__main__":
    main()

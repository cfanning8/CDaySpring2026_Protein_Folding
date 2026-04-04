#!/usr/bin/env python3
"""
Static checks for protein-track alignment with README Design freeze.
Exit 0 if ok; nonzero if a regression is detected.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    bad: list[str] = []
    pro = ROOT / "src" / "protein"
    for p in pro.rglob("*.py"):
        rel = p.relative_to(ROOT)
        if "legacy" in rel.parts:
            continue
        t = p.read_text(encoding="utf-8", errors="replace")
        if "Landscape" in t:
            bad.append(f"forbidden stage name 'Landscape' in {rel}")

    sm = ast.parse((pro / "smoke_metrics.py").read_text(encoding="utf-8"))
    found_stages: tuple[str, ...] | None = None
    for node in sm.body:
        if isinstance(node, ast.Assign):
            for tg in node.targets:
                if isinstance(tg, ast.Name) and tg.id == "SMOKE_STAGE_ORDER":
                    if isinstance(node.value, ast.Tuple):
                        found_stages = tuple(
                            str(elt.value) for elt in node.value.elts if isinstance(elt, ast.Constant)
                        )
    if found_stages != ("Baseline", "Wasserstein", "RKHS"):
        bad.append(f"SMOKE_STAGE_ORDER must be (Baseline, Wasserstein, RKHS), got {found_stages!r}")

    stub = ROOT / "scripts" / "protein" / "run_openfold3_predict_smoke.py"
    if stub.is_file() and "legacy" not in stub.read_text(encoding="utf-8").lower():
        bad.append("run_openfold3_predict_smoke.py should reference legacy path")

    smoke_tables = ROOT / "results" / "protein" / "tables" / "smoke"
    if smoke_tables.is_dir():
        for csv_path in sorted(smoke_tables.glob("*.csv")):
            txt = csv_path.read_text(encoding="utf-8", errors="replace")
            rel = csv_path.relative_to(ROOT)
            if ",Landscape," in txt or "\nLandscape," in txt:
                bad.append(f"forbidden stage name 'Landscape' in {rel}")
            if "train_m2_landscape" in txt:
                bad.append(f"obsolete regime id train_m2_landscape_ph in {rel}")
            if "train_m4_jet" in txt:
                bad.append(f"obsolete jet training row in {rel}")

    if bad:
        print("ALIGNMENT FAILURES:")
        for b in bad:
            print(" ", b)
        return 1
    print("verify_protein_alignment: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

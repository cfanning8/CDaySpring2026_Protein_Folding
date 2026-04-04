#!/usr/bin/env python3
"""Run fast in-repo checks for the protein + topology track (no ColabFold GPU)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(argv: list[str]) -> int:
    print(">>>", " ".join(argv))
    return subprocess.call(argv, cwd=str(ROOT))


def main() -> int:
    steps = [
        [sys.executable, str(ROOT / "scripts" / "protein" / "verify_protein_alignment.py")],
        [sys.executable, str(ROOT / "scripts" / "smoke_test_paper1_tools.py")],
        [sys.executable, str(ROOT / "scripts" / "protein" / "run_training_contract_demo.py")],
    ]
    if "--with-smoke" in sys.argv:
        steps.append(
            [
                sys.executable,
                "-u",
                str(ROOT / "scripts" / "protein" / "run_smoke_pipeline.py"),
                "--skip-structure-figures",
                "--no-colabfold-smoke",
            ]
        )
    for cmd in steps:
        rc = run(cmd)
        if rc != 0:
            return rc
    print("protein_ci_checks: all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

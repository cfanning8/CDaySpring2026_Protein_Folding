"""Deprecated entry point — use `scripts/protein/legacy/run_openfold3_predict_smoke.py`."""

from __future__ import annotations

import sys


def main() -> None:
    raise SystemExit(
        "This stack is embargoed for the ColabFold publication path.\n"
        "Run instead:\n"
        "  python scripts/protein/legacy/run_openfold3_predict_smoke.py [args...]\n"
        "See scripts/protein/legacy/README.md and README Design freeze DEF-O01."
    )


if __name__ == "__main__":
    main()

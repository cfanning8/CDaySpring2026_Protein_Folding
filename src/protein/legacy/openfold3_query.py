"""
Archived JSON helpers for a third-party AlphaFold3-class CLI stack.

**Not part of the ColabFold publication / fine-tuning path** (README Design freeze DEF-O01).
Kept for historical forks only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def monomer_query_document(query_key: str, *, chain_id: str, sequence: str) -> dict[str, Any]:
    return {
        "queries": {
            query_key: {
                "chains": [
                    {
                        "molecule_type": "protein",
                        "chain_ids": [chain_id],
                        "sequence": sequence,
                    }
                ]
            }
        }
    }


def write_query_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def merge_monomer_queries(queries: list[tuple[str, str, str]]) -> dict[str, Any]:
    """
    Build one JSON with multiple named queries.
    Each tuple is (query_key, chain_id, sequence).
    """
    inner: dict[str, Any] = {}
    for key, chain_id, sequence in queries:
        inner[key] = {
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": [chain_id],
                    "sequence": sequence,
                }
            ]
        }
    return {"queries": inner}

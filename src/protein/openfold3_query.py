"""
Backward-compatible shim. Implementation lives in `src.protein.legacy.openfold3_query`.

The protein **publication path is ColabFold only**; this module is **not** imported by
topology fine-tuning code (see README Design freeze).
"""

from __future__ import annotations

from src.protein.legacy.openfold3_query import (  # noqa: F401
    merge_monomer_queries,
    monomer_query_document,
    write_query_json,
)

__all__ = ["merge_monomer_queries", "monomer_query_document", "write_query_json"]

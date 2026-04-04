r"""
Nine smoke experiments: diagnostic **Level** {0,1,2} × stage {Baseline, Wasserstein, RKHS}.

**Metrics (publication-style):** only **TM-score** and **pLDDT** per cell.

- **TM-score:** CASP-style comparison using **TM-align** when ``TMalign`` is on ``PATH`` (see
  ``src/protein/tm_score_eval.py``); otherwise Zhang–Skolnick TM-score on CA after Kabsch alignment
  (fallback, labeled in CSV kinds).

- **pLDDT:** mean / min / std over the module’s residues by stage (same ColabFold prediction for all
  stages until separate checkpoints exist).

**Level** is a diagnostic split (three contiguous thirds), not recursive depth ``D^(n)``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.protein.smoke_metrics import SMOKE_STAGE_ORDER
from src.protein.tm_score_eval import tm_score_ca_arrays


def _split_three_modules(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contiguous index ranges for level 0,1,2 (as equal as possible)."""
    idx = np.arange(n, dtype=np.int64)
    parts = np.array_split(idx, 3)
    if len(parts) != 3:
        raise ValueError("expected three index groups")
    a, b, c = parts[0], parts[1], parts[2]
    if a.size == 0 or b.size == 0 or c.size == 0:
        raise ValueError("chain too short for three-way split")
    return a, b, c


def compute_nine_experiment_metrics(
    *,
    native_ca: np.ndarray,
    pred_ca: np.ndarray,
    plddt_per_residue: np.ndarray | None,
    graph_mode: str = "",
    radius_max_a: float = 0.0,
    max_dimension: int = 1,
    grid_size: int = 50,
    rkhs_random_state: int = 42,
) -> list[dict[str, Any]]:
    """
    Nine slots: per module P_a (level a) × stage.

    Unused kwargs (graph_mode, radius_max_a, …) remain for backward compatibility with
    ``run_smoke_pipeline.py``.
    """
    del graph_mode, radius_max_a, max_dimension, grid_size, rkhs_random_state

    native_ca = np.asarray(native_ca, dtype=np.float64)
    pred_ca = np.asarray(pred_ca, dtype=np.float64)
    if native_ca.shape != pred_ca.shape or native_ca.ndim != 2:
        raise ValueError("native_ca and pred_ca must match (n,3)")
    n = int(native_ca.shape[0])
    if n < 3:
        raise ValueError("need at least 3 residues for nine-way split")

    mod0, mod1, mod2 = _split_three_modules(n)
    modules = (mod0, mod1, mod2)

    plddt = np.asarray(plddt_per_residue, dtype=np.float64) if plddt_per_residue is not None else None

    rows: list[dict[str, Any]] = []
    slot = 0
    for level in (0, 1, 2):
        ix = modules[level]
        tm_val, tm_src = tm_score_ca_arrays(pred_ca[ix], native_ca[ix])
        tm_rounded = round(float(tm_val), 4)
        tm_kind = f"tm_score_{tm_src}"

        for stage in SMOKE_STAGE_ORDER:
            slot += 1
            col1: float | str = tm_rounded
            col2: float | str = ""
            k2 = ""
            if plddt is not None and plddt.shape[0] >= n:
                if stage == "Baseline":
                    col2 = round(float(np.mean(plddt[ix])), 3)
                    k2 = "mean_plddt_in_module"
                elif stage == "Wasserstein":
                    col2 = round(float(np.min(plddt[ix])), 3)
                    k2 = "min_plddt_in_module"
                else:
                    col2 = round(float(np.std(plddt[ix])), 3)
                    k2 = "std_plddt_in_module"

            rows.append(
                {
                    "slot_id": slot,
                    "structural_level": level,
                    "stage": stage,
                    "metric_col1_value": col1,
                    "metric_col2_value": col2,
                    "metric_col1_kind": tm_kind,
                    "metric_col2_kind": k2,
                }
            )
    return rows


def build_empty_nine_experiment_rows(*, reason: str) -> list[dict[str, Any]]:
    """Nine rows with empty metrics (e.g. missing prediction PDB)."""
    rows: list[dict[str, Any]] = []
    slot = 0
    for level in (0, 1, 2):
        for stage in SMOKE_STAGE_ORDER:
            slot += 1
            rows.append(
                {
                    "slot_id": slot,
                    "structural_level": level,
                    "stage": stage,
                    "metric_col1_value": "",
                    "metric_col2_value": "",
                    "metric_col1_kind": "empty",
                    "metric_col2_kind": "empty",
                    "error": reason if slot == 1 else "",
                }
            )
    return rows


def nine_rows_to_legacy_csv_dicts(
    rows: list[dict[str, Any]],
    *,
    header_note: str | None = None,
) -> list[dict[str, Any]]:
    """Rows for smoke_nine_run_metrics.csv."""
    default_note = (
        "TM-score (TM-align if PATH, else Kabsch-Zhang fallback); pLDDT mean/min/std by stage. "
        "Contiguous thirds; not CATH-internal VPD. See README."
    )
    out: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        note = ""
        if i == 0:
            note = header_note if header_note is not None else default_note
            err = r.get("error", "")
            if err:
                note = f"{note} Error: {err}"
        out.append(
            {
                "slot_id": r["slot_id"],
                "structural_level": r["structural_level"],
                "stage": r["stage"],
                "metric_col1_value": r["metric_col1_value"],
                "metric_col2_value": r["metric_col2_value"],
                "metric_col1_kind": r["metric_col1_kind"],
                "metric_col2_kind": r["metric_col2_kind"],
                "notes": note,
            }
        )
    return out

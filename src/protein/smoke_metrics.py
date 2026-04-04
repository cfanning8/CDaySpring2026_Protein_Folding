"""
Lightweight metrics for protein smoke / ColabFold vs native comparison.

Smoke **nine-cell** table: **TM-score** (TM-align when on ``PATH``, else Kabsch–Zhang fallback) and
**pLDDT** only — see ``src/protein/tm_score_eval.py`` and ``smoke_nine_experiments.py``.

The nine-slot smoke table is smoke **Level** in {0,1,2} (H0 / H1 / combined diagnostics) × **training
stage** {Baseline, Wasserstein, RKHS}. That **Level** is **not** the publication **depth** axis ``n``
for recursive ``D^(n)`` (see README). Not the same as pipeline layers mmCIF → cache → learning.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.protein.mmcif_io import load_ca_coords_from_mmcif
from src.protein.topology_cache import load_topology_npz


def rmsd_kabsch_ca(pred: np.ndarray, native: np.ndarray) -> float:
    """RMSD (Å) after optimal rotation (Kabsch). pred, native: (n, 3), same length."""
    p = np.asarray(pred, dtype=np.float64)
    q = np.asarray(native, dtype=np.float64)
    if p.shape != q.shape or p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("pred and native must have shape (n, 3)")
    n = p.shape[0]
    if n == 0:
        raise ValueError("empty coordinates")

    pc = p - p.mean(axis=0)
    qc = q - q.mean(axis=0)
    h = pc.T @ qc
    u, _, vt = np.linalg.svd(h, full_matrices=True)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    p_aligned = pc @ r
    return float(np.sqrt(np.mean(np.sum((p_aligned - qc) ** 2, axis=1))))


def load_ca_coords_b_factors_from_pdb(pdb_path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """CA coordinates and optional B-factors (used as pLDDT in many AF2 PDB exports)."""
    from biotite.structure.io.pdb import PDBFile

    pdb_path = Path(pdb_path)
    if not pdb_path.is_file():
        raise FileNotFoundError(str(pdb_path))
    pdb_file = PDBFile.read(str(pdb_path))
    structure = pdb_file.get_structure(model=1, extra_fields=["b_factor"])
    ca = structure[structure.atom_name == "CA"]
    if len(ca) == 0:
        raise ValueError("no CA atoms in PDB")
    coords = np.asarray(ca.coord, dtype=np.float64)
    bfac: np.ndarray | None = None
    if "b_factor" in ca.get_annotation_categories():
        bfac = np.asarray(ca.b_factor, dtype=np.float64)
    return coords, bfac


def mean_plddt_from_pdb(pdb_path: Path) -> float | None:
    """Mean B-factor over CA atoms (ColabFold/AlphaFold PDBs often store pLDDT there)."""
    _, bf = load_ca_coords_b_factors_from_pdb(pdb_path)
    if bf is None or bf.size == 0:
        return None
    m = float(np.mean(bf))
    if m > 1.5:  # already 0–100 scale
        return m
    if m <= 1.0:  # sometimes stored as fraction
        return float(m * 100.0)
    return m


def find_colabfold_ranked_pdb(pred_dir: Path, pdb_id: str) -> Path | None:
    """Best-effort locate a ColabFold output PDB under predictions/."""
    pred_dir = Path(pred_dir)
    if not pred_dir.is_dir():
        return None
    pid = pdb_id.upper().strip()
    ranked = sorted(
        pred_dir.rglob("*.pdb"),
        key=lambda p: ("ranked_001" in p.name.lower(), "ranked_1" in p.name.lower(), p.name.lower()),
        reverse=True,
    )
    for p in ranked:
        if pid in p.stem.upper():
            return p
    for p in pred_dir.rglob("*.pdb"):
        if pid in p.stem.upper():
            return p
    return None


def persistence_lifetime_sums(npz_path: Path) -> tuple[float, float]:
    """Sum of (death - birth) over finite intervals, split by dimension."""
    loaded = load_topology_npz(Path(npz_path))
    pers = np.asarray(loaded["persistence"], dtype=np.float64)
    s0 = s1 = 0.0
    for row in pers:
        dim, b, d = int(row[0]), float(row[1]), float(row[2])
        if not np.isfinite(b) or not np.isfinite(d):
            continue
        lif = d - b
        if lif <= 0:
            continue
        if dim == 0:
            s0 += lif
        elif dim == 1:
            s1 += lif
    return s0, s1


def compute_per_structure_metrics(
    *,
    mmcif_path: Path,
    chain_id: str,
    npz_path: Path,
    pred_dir: Path,
    pdb_id: str,
) -> dict[str, Any]:
    """One row dict: native topology + optional prediction comparison."""
    s0, s1 = persistence_lifetime_sums(npz_path)
    native_ca = load_ca_coords_from_mmcif(mmcif_path, chain_id=chain_id)
    out: dict[str, Any] = {
        "pdb_id": pdb_id,
        "n_residues_native": int(native_ca.shape[0]),
        "ph_h0_total_persistence_sum": round(s0, 6),
        "ph_h1_total_persistence_sum": round(s1, 6),
        "rmsd_ca_kabsch_angstrom": "",
        "mean_plddt_colabfold": "",
        "prediction_pdb_path": "",
        "metric_notes": "",
    }
    pred_pdb = find_colabfold_ranked_pdb(pred_dir, pdb_id)
    if pred_pdb is None:
        out["metric_notes"] = "no_prediction_pdb_found"
        return out
    out["prediction_pdb_path"] = str(pred_pdb).replace("\\", "/")
    try:
        pred_ca, _ = load_ca_coords_b_factors_from_pdb(pred_pdb)
    except Exception as exc:  # noqa: BLE001
        out["metric_notes"] = f"pred_pdb_parse_error:{exc!r}"
        return out
    n = min(int(native_ca.shape[0]), int(pred_ca.shape[0]))
    if n < 1:
        out["metric_notes"] = "empty_ca"
        return out
    if native_ca.shape[0] != pred_ca.shape[0]:
        out["metric_notes"] = f"length_mismatch_native_{native_ca.shape[0]}_pred_{pred_ca.shape[0]}_using_min_{n}"
        native_ca = native_ca[:n]
        pred_ca = pred_ca[:n]
    try:
        out["rmsd_ca_kabsch_angstrom"] = round(rmsd_kabsch_ca(pred_ca, native_ca), 4)
    except Exception as exc:  # noqa: BLE001
        out["metric_notes"] = f"rmsd_error:{exc!r}"
    mp = mean_plddt_from_pdb(pred_pdb)
    if mp is not None:
        out["mean_plddt_colabfold"] = round(mp, 3)
    return out


# Training recipe labels for the 3×3 smoke grid (not epidemiology model names).
SMOKE_STAGE_ORDER = ("Baseline", "Wasserstein", "RKHS")


def load_native_pred_plddt_aligned(
    mmcif_path: Path,
    chain_id: str,
    pred_pdb: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Trim-aligned native CA, pred CA, and per-residue pLDDT (B-factors on CA), or None if missing."""
    native_ca = load_ca_coords_from_mmcif(mmcif_path, chain_id=chain_id)
    pred_ca, bf = load_ca_coords_b_factors_from_pdb(Path(pred_pdb))
    n = min(int(native_ca.shape[0]), int(pred_ca.shape[0]))
    if n < 1:
        raise ValueError("empty CA after alignment")
    native_ca = native_ca[:n]
    pred_ca = pred_ca[:n]
    plddt: np.ndarray | None = None
    if bf is not None and bf.size >= n:
        plddt = np.asarray(bf[:n], dtype=np.float64)
    return native_ca, pred_ca, plddt


def nine_run_metrics_to_display_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Rows for PNG: Level | Stage | TM-score | pLDDT."""

    def _fmt_tm(v: object) -> str:
        if v == "" or v is None:
            return ""
        return f"{float(v):.4f}"

    def _fmt_plddt(v: object) -> str:
        if v == "" or v is None:
            return ""
        return f"{float(v):.3f}"

    out: list[dict[str, Any]] = []
    for r in sorted(rows, key=lambda x: (int(x["structural_level"]), SMOKE_STAGE_ORDER.index(str(x["stage"])))):
        out.append(
            {
                "structural_level": int(r["structural_level"]),
                "stage": str(r["stage"]),
                "tm_score": _fmt_tm(r.get("metric_col1_value", "")),
                "plddt": _fmt_plddt(r.get("metric_col2_value", "")),
            }
        )
    return pd.DataFrame(out)

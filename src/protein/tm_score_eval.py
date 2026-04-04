"""
CASP-style TM-score for structure comparison.

Primary path: **TM-align** (Yang & Zhang), the same family of tool used in CASP assessment
(https://zhanggroup.org/TM-align/). Install locally and ensure `TMalign` is on ``PATH``, or set
``TMALIGN_BIN`` to the executable.

Fallback: Zhang–Skolnick TM-score on CA traces after Kabsch alignment (no extra binaries), for smoke
when TM-align is unavailable — document as approximate vs official CASP/TM-align pipelines.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

_TM_SCORE_LINE = re.compile(r"TM-score\s*=\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def tm_score_kabsch_fallback(pred_ca: np.ndarray, native_ca: np.ndarray) -> float:
    """
    TM-score in (0,1] after Kabsch alignment (Zhang & Skolnick definition, single chain).
    Use when ``TMalign`` is not installed.
    """
    p = np.asarray(pred_ca, dtype=np.float64)
    q = np.asarray(native_ca, dtype=np.float64)
    if p.shape != q.shape or p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("pred_ca and native_ca must have shape (n, 3)")
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
    d = np.linalg.norm(p_aligned - qc, axis=1)
    L = float(n)
    if L > 15:
        d0 = 1.24 * (L - 15.0) ** (1.0 / 3.0) - 1.8
    else:
        d0 = 0.5
    if d0 <= 0:
        d0 = 0.5
    return float(np.mean(1.0 / (1.0 + (d / d0) ** 2)))


def find_tm_align_executable() -> str | None:
    env = os.environ.get("TMALIGN_BIN", "").strip()
    if env and Path(env).is_file():
        return env
    return shutil.which("TMalign")


def run_tm_align_two_pdbs(pred_pdb: Path, native_pdb: Path, *, timeout_s: float = 120.0) -> float | None:
    """
    Run TM-align on two PDB files; return first TM-score reported (normalized by first structure).
    Returns None if executable missing or parse failure.
    """
    exe = find_tm_align_executable()
    if not exe:
        return None
    pred_pdb = Path(pred_pdb).resolve()
    native_pdb = Path(native_pdb).resolve()
    try:
        proc = subprocess.run(
            [exe, str(pred_pdb), str(native_pdb)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    scores: list[float] = []
    for m in _TM_SCORE_LINE.finditer(text):
        try:
            scores.append(float(m.group(1)))
        except ValueError:
            continue
    return scores[0] if scores else None


def write_minimal_ca_pdb(path: Path, coords: np.ndarray, *, chain_id: str = "A") -> None:
    """Write a single-chain CA-only PDB for TM-align (minimal ATOM records)."""
    path = Path(path)
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must be (n, 3)")
    n = coords.shape[0]
    cid = chain_id[:1].upper()
    lines = ["HEADER    TM-EVAL", "MODEL        1"]
    for i in range(n):
        x, y, z = coords[i]
        # PDB v3: ATOM serial, CA, ALA, chain, resSeq, coords
        lines.append(
            f"ATOM  {i + 1:5d}  CA  ALA {cid}{i + 1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
    lines.extend(["ENDMDL", "END"])
    path.write_text("\n".join(lines) + "\n", encoding="ascii", errors="strict")


def tm_score_ca_arrays(
    pred_ca: np.ndarray,
    native_ca: np.ndarray,
    *,
    work_dir: Path | None = None,
) -> tuple[float, str]:
    """
    TM-score for aligned CA arrays.

    Returns (score, source) where source is ``"TMalign"`` or ``"kabsch_fallback"``.
    """
    pred_ca = np.asarray(pred_ca, dtype=np.float64)
    native_ca = np.asarray(native_ca, dtype=np.float64)
    if pred_ca.shape != native_ca.shape:
        raise ValueError("pred_ca and native_ca must match in shape")

    if work_dir is not None:
        wd = Path(work_dir)
        wd.mkdir(parents=True, exist_ok=True)
        p_pred = wd / "pred_ca.pdb"
        p_nat = wd / "native_ca.pdb"
        write_minimal_ca_pdb(p_pred, pred_ca)
        write_minimal_ca_pdb(p_nat, native_ca)
        s = run_tm_align_two_pdbs(p_pred, p_nat)
        if s is not None and np.isfinite(s):
            return float(s), "TMalign"
        return tm_score_kabsch_fallback(pred_ca, native_ca), "kabsch_fallback"

    with tempfile.TemporaryDirectory(prefix="tm_eval_") as td:
        wd = Path(td)
        p_pred = wd / "pred_ca.pdb"
        p_nat = wd / "native_ca.pdb"
        write_minimal_ca_pdb(p_pred, pred_ca)
        write_minimal_ca_pdb(p_nat, native_ca)
        s = run_tm_align_two_pdbs(p_pred, p_nat)
        if s is not None and np.isfinite(s):
            return float(s), "TMalign"

    return tm_score_kabsch_fallback(pred_ca, native_ca), "kabsch_fallback"

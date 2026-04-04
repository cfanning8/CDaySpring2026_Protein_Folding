"""Use shared epidemiology-style 3D persistence engine for protein NPZ outputs."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def render_npz_persistence_3d(npz_path: Path, output_path: Path) -> bool:
    """
    Render clique persistence from topology NPZ using `persistence_diagram_3d_engine`.
    Returns True if a figure was written.
    """
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "figures"))
    from persistence_diagram_3d_engine import render_persistence_diagram_3d  # noqa: PLC0415

    from src.protein.topology_cache import load_topology_npz  # noqa: PLC0415

    npz_path = Path(npz_path)
    loaded = load_topology_npz(npz_path)
    pers = np.asarray(loaded["persistence"], dtype=np.float64)
    if pers.size == 0:
        return False
    n = pers.shape[0]
    mult = np.ones((n, 1), dtype=np.float64)
    arr4 = np.hstack([pers, mult])

    b, d = arr4[:, 1], arr4[:, 2]
    fin = np.isfinite(b) & np.isfinite(d) & (d > b)
    arr4 = arr4[fin]
    if len(arr4) == 0:
        return False

    bmin, bmax = float(np.min(arr4[:, 1])), float(np.max(arr4[:, 1]))
    dmin, dmax = float(np.min(arr4[:, 2])), float(np.max(arr4[:, 2]))
    span = max(bmax - bmin, dmax - dmin, 1e-9)
    pad = 0.05 * span

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_persistence_diagram_3d(
        arr4,
        output_path,
        xlim=(bmin - pad, bmax + pad),
        ylim=(dmin - pad, dmax + pad),
        figsize=(8, 8),
    )
    return True

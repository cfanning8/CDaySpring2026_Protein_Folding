"""
Virtual persistence diagram visualization engine.

Renders the pointwise difference A - B of two persistence diagrams,
where multiplicities can be negative. Positive mult = filled markers;
negative mult = distinct style with negative number label.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from persistence_diagram_engine import (
    FIGURE_DPI,
    COLOR_H0,
    COLOR_H1,
    COLOR_IMPOSSIBLE,
    ALPHA_IMPOSSIBLE,
)

# Colors for negative multiplicity (lighter variants)
COLOR_H0_NEG = "#87CEEB"
COLOR_H1_NEG = "#FFB6C1"


def _pointwise_subtract(
    A: np.ndarray, B: np.ndarray, decimals: int = 6
) -> np.ndarray:
    """
    Compute A - B pointwise. Both are (N,4): [dim, birth, death, mult].
    Returns (M,4) with [dim, birth, death, signed_mult] for points with mult != 0.
    """
    from collections import defaultdict
    diff: dict[Tuple[int, float, float], int] = defaultdict(int)
    rep: dict[Tuple[int, float, float], Tuple[float, float]] = {}

    def _key(dim: int, b: float, d: float) -> Tuple[int, float, float]:
        return (dim, round(b, decimals), round(d, decimals))

    for row in A:
        dim, b, d, m = int(row[0]), float(row[1]), float(row[2]), int(row[3])
        if np.isfinite(b) and np.isfinite(d) and d > b:
            k = _key(dim, b, d)
            diff[k] += m
            rep[k] = (b, d)
    for row in B:
        dim, b, d, m = int(row[0]), float(row[1]), float(row[2]), int(row[3])
        if np.isfinite(b) and np.isfinite(d) and d > b:
            k = _key(dim, b, d)
            diff[k] -= m
            if k not in rep:
                rep[k] = (b, d)
    rows = []
    for k, m in diff.items():
        if m != 0 and k in rep:
            b, d = rep[k]
            rows.append([k[0], b, d, m])
    return np.array(rows) if rows else np.zeros((0, 4))


def render_virtual_persistence_diagram(
    diagram_a: np.ndarray,
    diagram_b: np.ndarray,
    output_path: Path,
    show_impossible_triangle: bool = True,
    figsize: Tuple[float, float] = (6, 6),
) -> None:
    """
    Render the virtual diagram A - B (pointwise subtraction).
    Points with positive multiplicity use H0/H1 colors; negative use lighter variants.

    Args:
        diagram_a: (N,4) array [dim, birth, death, multiplicity].
        diagram_b: (N,4) array [dim, birth, death, multiplicity].
        output_path: Where to save the figure (PNG).
        show_impossible_triangle: If True, shade the region where death < birth.
        figsize: Figure size (width, height).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    diff = _pointwise_subtract(
        np.asarray(diagram_a, dtype=np.float64),
        np.asarray(diagram_b, dtype=np.float64),
    )

    if len(diff) == 0:
        # Empty virtual diagram
        fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_DPI)
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Birth", fontsize=12)
        ax.set_ylabel("Death", fontsize=12)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3, linestyle=":", linewidth=2.0)
        if show_impossible_triangle:
            x = np.linspace(0, 1, 200)
            ax.fill_between(x, 0, x, alpha=ALPHA_IMPOSSIBLE, color=COLOR_IMPOSSIBLE, zorder=5)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=2, zorder=6)
        plt.tight_layout()
        plt.savefig(
            output_path, bbox_inches="tight", dpi=FIGURE_DPI,
            facecolor="none", edgecolor="none", transparent=True,
        )
        plt.close(fig)
        return

    dims = diff[:, 0].astype(int)
    births = diff[:, 1]
    deaths = diff[:, 2]
    mults = diff[:, 3].astype(int)

    fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_DPI)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    max_val = max(np.max(births), np.max(deaths))
    min_val = min(np.min(births), np.min(deaths))
    margin = max((max_val - min_val) * 0.1, 0.05)
    x_max = max_val + margin
    x_min = max(0, min_val - margin)

    pos_mask = mults > 0
    neg_mask = mults < 0
    dim_markers = {0: "o", 1: "s"}

    for dim in sorted(np.unique(dims)):
        for mask, use_pos in [(pos_mask, True), (neg_mask, False)]:
            m = (dims == dim) & mask
            if not np.any(m):
                continue
            b, d, mi = births[m], deaths[m], mults[m]
            if use_pos:
                color = COLOR_H0 if dim == 0 else COLOR_H1
            else:
                color = COLOR_H0_NEG if dim == 0 else COLOR_H1_NEG
            marker = dim_markers.get(dim, "^")
            s = 150 if dim == 0 else 120
            ax.scatter(
                b, d, c=color, marker=marker, s=s, alpha=0.8,
                edgecolors="black", linewidths=1.5, zorder=7,
            )
            for xi, yi, mval in zip(b, d, mi):
                ax.text(
                    xi, yi, str(int(mval)), ha="center", va="center",
                    fontsize=10, fontweight="bold", color="black", zorder=8,
                )

    ax.set_xlabel("Birth", fontsize=12)
    ax.set_ylabel("Death", fontsize=12)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([x_min, x_max])
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=2.0)
    if show_impossible_triangle and x_max > 0:
        x = np.linspace(0, x_max, 200)
        ax.fill_between(x, 0, x, alpha=ALPHA_IMPOSSIBLE, color=COLOR_IMPOSSIBLE, zorder=5)
    ax.plot([x_min, x_max], [x_min, x_max], "k--", alpha=0.5, linewidth=2, zorder=6)

    plt.tight_layout()
    plt.savefig(
        output_path, bbox_inches="tight", dpi=FIGURE_DPI,
        facecolor="none", edgecolor="none", transparent=True,
    )
    plt.close(fig)

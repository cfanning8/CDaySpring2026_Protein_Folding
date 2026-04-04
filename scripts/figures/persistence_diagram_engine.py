"""
Persistence diagram visualization engine.

Renders persistence diagrams (birth vs death) with:
- Shaded triangle in the impossible region (death < birth)
- Diagonal line (birth = death)
- H0 (components) in blue, H1 (loops) in red
- No title, transparent background, clean design.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

FIGURE_DPI = 300

# Dimension colors (H0 components, H1 loops)
COLOR_H0 = "#1976d2"
COLOR_H1 = "#dc143c"
COLOR_IMPOSSIBLE = "#ffffff"
ALPHA_IMPOSSIBLE = 0.5


def render_persistence_diagram(
    persistence: Union[np.ndarray, List[Tuple[float, float]], List[Tuple[int, float, float]]],
    output_path: Path,
    format_: str = "auto",
    show_impossible_triangle: bool = True,
    figsize: Tuple[float, float] = (6, 6),
) -> None:
    """
    Render a persistence diagram to a file.

    Args:
        persistence: Persistence data. Formats:
            - np.ndarray shape (N,3): [dim, birth, death] per row
            - List of (birth, death) pairs (single dimension)
            - List of (dim, birth, death) tuples
        output_path: Where to save the figure (PNG).
        format_: 'auto' infers format; 'dim_birth_death' expects (N,3) or list of (dim,b,d).
        show_impossible_triangle: If True, shade the region where death < birth.
        figsize: Figure size (width, height).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(persistence, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim == 2 and arr.shape[1] == 2:
        dims = np.zeros(arr.shape[0], dtype=int)
        births = arr[:, 0]
        deaths = arr[:, 1]
    elif arr.ndim == 2 and arr.shape[1] >= 3:
        dims = arr[:, 0].astype(int)
        births = arr[:, 1]
        deaths = arr[:, 2]
    else:
        raise ValueError(f"Unsupported persistence shape: {arr.shape}")

    valid = np.isfinite(births) & np.isfinite(deaths) & (deaths > births)
    dims = dims[valid]
    births = births[valid]
    deaths = deaths[valid]

    fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_DPI)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    max_val = max(np.max(births), np.max(deaths)) if len(births) > 0 else 1.0
    min_val = min(np.min(births), np.min(deaths)) if len(births) > 0 else 0.0
    margin = max((max_val - min_val) * 0.1, 0.05)
    x_max = max_val + margin
    x_min = max(0, min_val - margin)


    mask0 = dims == 0
    mask1 = dims == 1
    mask_other = ~mask0 & ~mask1

    if np.any(mask0):
        ax.scatter(
            births[mask0],
            deaths[mask0],
            c=COLOR_H0,
            marker="o",
            s=150,
            alpha=0.8,
            edgecolors="black",
            linewidths=1.5,
            zorder=7,
            label="H0",
        )
    if np.any(mask1):
        ax.scatter(
            births[mask1],
            deaths[mask1],
            c=COLOR_H1,
            marker="s",
            s=120,
            alpha=0.8,
            edgecolors="black",
            linewidths=1.5,
            zorder=7,
            label="H1",
        )
    if np.any(mask_other):
        ax.scatter(
            births[mask_other],
            deaths[mask_other],
            c="#808080",
            marker="^",
            s=100,
            alpha=0.8,
            edgecolors="black",
            linewidths=1.5,
            zorder=7,
            label="H2",
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
    if len(births) > 0:
        ax.plot([x_min, x_max], [x_min, x_max], "k--", alpha=0.5, linewidth=2, zorder=6)
    if np.any(mask0) or np.any(mask1) or np.any(mask_other):
        ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(
        output_path,
        bbox_inches="tight",
        dpi=FIGURE_DPI,
        facecolor="none",
        edgecolor="none",
        transparent=True,
    )
    plt.close(fig)


def render_persistence_diagram_with_multiplicity(
    persistence: Union[np.ndarray, List[Tuple[int, float, float, int]]],
    output_path: Path,
    show_impossible_triangle: bool = True,
    figsize: Tuple[float, float] = (6, 6),
) -> None:
    """
    Render a persistence diagram with multiplicity labels (black numbers inside markers).

    Args:
        persistence: np.ndarray shape (N,4): [dim, birth, death, multiplicity] per row.
        output_path: Where to save the figure (PNG).
        show_impossible_triangle: If True, shade the region where death < birth.
        figsize: Figure size (width, height).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(persistence, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError("Expect (N,4) array: [dim, birth, death, multiplicity]")

    dims = arr[:, 0].astype(int)
    births = arr[:, 1]
    deaths = arr[:, 2]
    mults = arr[:, 3].astype(int)

    valid = np.isfinite(births) & np.isfinite(deaths) & (deaths > births) & (mults != 0)
    dims = dims[valid]
    births = births[valid]
    deaths = deaths[valid]
    mults = mults[valid]

    fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_DPI)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    max_val = max(np.max(births), np.max(deaths)) if len(births) > 0 else 1.0
    min_val = min(np.min(births), np.min(deaths)) if len(births) > 0 else 0.0
    margin = max((max_val - min_val) * 0.1, 0.05)
    x_max = max_val + margin
    x_min = max(0, min_val - margin)

    dim_colors = {0: COLOR_H0, 1: COLOR_H1}
    dim_markers = {0: "o", 1: "s"}

    for dim in sorted(np.unique(dims)):
        mask = dims == dim
        b = births[mask]
        d = deaths[mask]
        m = mults[mask]
        color = dim_colors.get(dim, "#808080")
        marker = dim_markers.get(dim, "^")
        s = 150 if dim == 0 else 120
        ax.scatter(
            b, d, c=color, marker=marker, s=s, alpha=0.8,
            edgecolors="black", linewidths=1.5, zorder=7,
        )
        for xi, yi, mi in zip(b, d, m):
            ax.text(
                xi, yi, str(int(mi)), ha="center", va="center",
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
    if len(births) > 0:
        ax.plot([x_min, x_max], [x_min, x_max], "k--", alpha=0.5, linewidth=2, zorder=6)

    plt.tight_layout()
    plt.savefig(
        output_path, bbox_inches="tight", dpi=FIGURE_DPI,
        facecolor="none", edgecolor="none", transparent=True,
    )
    plt.close(fig)

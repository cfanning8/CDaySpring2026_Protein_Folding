"""
3D persistence diagram visualization engine.

Renders persistence diagrams in 3D:
- Square base 0 to 1 on birth (x) and death (y) axes
- Z-axis = multiplicity (vertical bars)
- Diagonal line on the floor
- Points at bar locations; vertical lines show multiplicity
- Color by homology dimension (H0 blue, H1 red)
"""

from __future__ import annotations

from pathlib import Path
from typing import Union, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d import Axes3D

from persistence_diagram_engine import FIGURE_DPI, COLOR_H0, COLOR_H1

COLOR_OTHER = "#808080"
COLOR_DIAGONAL = "#000000"
COLOR_FLOOR_GRID = "#999999"


def _parse_diagram(
    persistence: np.ndarray, require_mult: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return dims, births, deaths, mults from (N,4) array."""
    arr = np.asarray(persistence, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 4 and require_mult:
        # Add mult=1 if missing
        ext = np.ones((arr.shape[0], 4 - arr.shape[1]), dtype=np.float64)
        arr = np.hstack([arr, ext])
    dims = arr[:, 0].astype(int)
    births = arr[:, 1]
    deaths = arr[:, 2]
    mults = arr[:, 3].astype(int)
    valid = np.isfinite(births) & np.isfinite(deaths) & (deaths > births) & (mults != 0)
    return dims[valid], births[valid], deaths[valid], mults[valid]


def render_persistence_diagram_3d(
    persistence: Union[np.ndarray, List[Tuple[int, float, float, int]]],
    output_path: Path,
    xlim: Tuple[float, float] = (0.0, 1.0),
    ylim: Tuple[float, float] = (0.0, 1.0),
    figsize: Tuple[float, float] = (10, 10),
    elev: float = 25,
    azim: float = -60,
    bar_scale: float = 1.0,
) -> None:
    """
    Render a 3D persistence diagram with multiplicity as vertical bars.

    Args:
        persistence: (N,4) array [dim, birth, death, multiplicity].
        output_path: Where to save the figure (PNG).
        xlim, ylim: Birth and death axis limits.
        figsize: Figure size.
        elev, azim: 3D view elevation and azimuth.
        bar_scale: Scale factor for bar heights (multiplicity).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dims, births, deaths, mults = _parse_diagram(np.asarray(persistence))

    fig = plt.figure(figsize=figsize, dpi=FIGURE_DPI)
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("Birth", fontsize=12, labelpad=10)
    ax.set_ylabel("Death", fontsize=12, labelpad=10)

    z_min, z_max = 0, 1
    if len(mults) > 0:
        z_max = max(1, np.max(mults) * bar_scale)
    ax.set_zlim(0, z_max)
    ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_zlabel("Multiplicity", fontsize=12, labelpad=10)

    ax.grid(False)

    # Render order: lower zorder = drawn first (back), higher = drawn last (front)
    Z_GRID, Z_PLANE, Z_DIAG, Z_BARS = 1, 2, 3, 4

    # 1. Floor grid
    for val in np.linspace(xlim[0], xlim[1], 6):
        ax.plot([val, val], [ylim[0], ylim[1]], [0, 0], color=COLOR_FLOOR_GRID, linewidth=2, alpha=0.6, zorder=Z_GRID)
    for val in np.linspace(ylim[0], ylim[1], 6):
        ax.plot([xlim[0], xlim[1]], [val, val], [0, 0], color=COLOR_FLOOR_GRID, linewidth=2, alpha=0.6, zorder=Z_GRID)

    # 2. White plane at z=0
    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 2),
        np.linspace(ylim[0], ylim[1], 2),
    )
    zz = np.zeros_like(xx)
    surf = ax.plot_surface(xx, yy, zz, color="white", alpha=0.5, shade=False)
    surf.set_zorder(Z_PLANE)

    # 3. Diagonal on the floor (in front of grid) - dashed
    diag = np.linspace(xlim[0], min(xlim[1], ylim[1]), 100)
    ax.plot(diag, diag, np.zeros_like(diag), color=COLOR_DIAGONAL, linewidth=5, alpha=1.0, linestyle="--", zorder=Z_DIAG)

    # 4. Persistence bars (in front of everything)
    dim_colors = {0: COLOR_H0, 1: COLOR_H1}

    for dim in sorted(np.unique(dims)):
        mask = dims == dim
        b, d, m = births[mask], deaths[mask], mults[mask]
        color = dim_colors.get(dim, COLOR_OTHER)
        h = m.astype(float) * bar_scale
        for xi, yi, hi in zip(b, d, h):
            if hi > 0:
                ax.plot([xi, xi], [yi, yi], [0, hi], color=color, linewidth=4, alpha=0.9, zorder=Z_BARS)
                ax.scatter([xi], [yi], [hi], c=color, s=80, alpha=0.9, edgecolors="black", linewidths=1, zorder=Z_BARS)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("none")
    ax.yaxis.pane.set_edgecolor("none")
    ax.zaxis.pane.set_edgecolor("none")
    ax.grid(False)
    ax.view_init(elev=elev, azim=azim)

    plt.tight_layout(pad=2)
    plt.savefig(
        output_path, bbox_inches="tight", pad_inches=0.3, dpi=FIGURE_DPI,
        facecolor="none", edgecolor="none", transparent=True,
    )
    plt.close(fig)


def render_virtual_persistence_diagram_3d(
    diagram_a: np.ndarray,
    diagram_b: np.ndarray,
    output_path: Path,
    xlim: Tuple[float, float] = (0.0, 1.0),
    ylim: Tuple[float, float] = (0.0, 1.0),
    figsize: Tuple[float, float] = (10, 10),
    elev: float = 25,
    azim: float = -60,
    bar_scale: float = 1.0,
) -> None:
    """
    Render 3D virtual diagram (A - B) with signed multiplicities.
    Positive bars go up; negative bars go down from z=0.
    """
    from virtual_persistence_diagram_engine import _pointwise_subtract

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    diff = _pointwise_subtract(
        np.asarray(diagram_a, dtype=np.float64),
        np.asarray(diagram_b, dtype=np.float64),
    )

    if len(diff) == 0:
        fig = plt.figure(figsize=figsize, dpi=FIGURE_DPI)
        fig.patch.set_alpha(0.0)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(-0.5, 0.5)
        ax.set_xlabel("Birth", fontsize=12)
        ax.set_ylabel("Death", fontsize=12)
        ax.set_zlabel("Multiplicity", fontsize=12, labelpad=10)
        for val in np.linspace(0, 1, 6):
            ax.plot([val, val], [0, 1], [0, 0], color=COLOR_FLOOR_GRID, linewidth=2, alpha=0.6)
            ax.plot([0, 1], [val, val], [0, 0], color=COLOR_FLOOR_GRID, linewidth=2, alpha=0.6)
        xx, yy = np.meshgrid(np.linspace(0, 1, 2), np.linspace(0, 1, 2))
        zz = np.zeros_like(xx)
        ax.plot_surface(xx, yy, zz, color="white", alpha=0.5, shade=False)
        diag = np.linspace(0, 1, 100)
        ax.plot(diag, diag, np.zeros_like(diag), color=COLOR_DIAGONAL, linewidth=5, alpha=0.5, linestyle="--")
        ax.grid(False)
        ax.view_init(elev=elev, azim=azim)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        plt.tight_layout(pad=2)
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0.3, dpi=FIGURE_DPI,
                    facecolor="none", transparent=True)
        plt.close(fig)
        return

    dims = diff[:, 0].astype(int)
    births = diff[:, 1]
    deaths = diff[:, 2]
    mults = diff[:, 3].astype(int)

    fig = plt.figure(figsize=figsize, dpi=FIGURE_DPI)
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    z_abs_max = max(1, np.max(np.abs(mults)) * bar_scale)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(-z_abs_max, z_abs_max)
    ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xlabel("Birth", fontsize=12, labelpad=10)
    ax.set_ylabel("Death", fontsize=12, labelpad=10)
    ax.set_zlabel("Multiplicity", fontsize=12, labelpad=10)

    ax.grid(False)

    # Render order: lower zorder = drawn first (back), higher = drawn last (front)
    Z_NEG, Z_GRID, Z_PLANE, Z_DIAG, Z_POS = 0, 1, 2, 3, 4

    # Filter: only points above diagonal (death > birth)
    above = deaths > births
    dims, births, deaths, mults = dims[above], births[above], deaths[above], mults[above]

    dim_colors = {0: COLOR_H0, 1: COLOR_H1}

    # 1. Negative multiplicity (render order 0 - back)
    neg_mask = mults < 0
    for dim in sorted(np.unique(dims)):
        mask = (dims == dim) & neg_mask
        if not np.any(mask):
            continue
        b, d, m = births[mask], deaths[mask], mults[mask]
        color = dim_colors.get(dim, COLOR_OTHER)
        h = m.astype(float) * bar_scale
        for xi, yi, hi in zip(b, d, h):
            ln, = ax.plot([xi, xi], [yi, yi], [0, hi], color=color, linewidth=4, alpha=0.9, zorder=Z_NEG)
            sc = ax.scatter([xi], [yi], [hi], c=color, s=80, alpha=0.9, edgecolors="black", linewidths=1, zorder=Z_NEG)

    # 2. Floor grid
    for val in np.linspace(xlim[0], xlim[1], 6):
        ax.plot([val, val], [ylim[0], ylim[1]], [0, 0], color=COLOR_FLOOR_GRID, linewidth=2, alpha=0.6, zorder=Z_GRID)
    for val in np.linspace(ylim[0], ylim[1], 6):
        ax.plot([xlim[0], xlim[1]], [val, val], [0, 0], color=COLOR_FLOOR_GRID, linewidth=2, alpha=0.6, zorder=Z_GRID)

    # 2b. White plane
    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 2),
        np.linspace(ylim[0], ylim[1], 2),
    )
    zz = np.zeros_like(xx)
    surf = ax.plot_surface(xx, yy, zz, color="white", alpha=0.5, shade=False)
    surf.set_zorder(Z_PLANE)

    # 3. Diagonal
    diag = np.linspace(xlim[0], min(xlim[1], ylim[1]), 100)
    ax.plot(diag, diag, np.zeros_like(diag), color=COLOR_DIAGONAL, linewidth=5, alpha=0.5, linestyle="--", zorder=Z_DIAG)

    # 4. Positive multiplicity (front)
    pos_mask = mults > 0
    for dim in sorted(np.unique(dims)):
        mask = (dims == dim) & pos_mask
        if not np.any(mask):
            continue
        b, d, m = births[mask], deaths[mask], mults[mask]
        color = dim_colors.get(dim, COLOR_OTHER)
        h = m.astype(float) * bar_scale
        for xi, yi, hi in zip(b, d, h):
            ax.plot([xi, xi], [yi, yi], [0, hi], color=color, linewidth=4, alpha=0.9, zorder=Z_POS)
            ax.scatter([xi], [yi], [hi], c=color, s=80, alpha=0.9, edgecolors="black", linewidths=1, zorder=Z_POS)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("none")
    ax.yaxis.pane.set_edgecolor("none")
    ax.zaxis.pane.set_edgecolor("none")
    ax.grid(False)
    ax.view_init(elev=elev, azim=azim)

    plt.tight_layout(pad=2)
    plt.savefig(
        output_path, bbox_inches="tight", pad_inches=0.3, dpi=FIGURE_DPI,
        facecolor="none", edgecolor="none", transparent=True,
    )
    plt.close(fig)

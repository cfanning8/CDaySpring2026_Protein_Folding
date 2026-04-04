"""
PNG table for protein smoke: 3×3 grid.

Columns: Level | Stage | TM-score | pLDDT

**Level** = smoke diagnostic axis (H0 / H1 / combined modules), not recursive depth ``n``.
TM-score uses **TM-align** if ``TMalign`` is on ``PATH`` (CASP-style); else Kabsch–Zhang fallback
(see ``src/protein/tm_score_eval.py``).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.protein.smoke_metrics import SMOKE_STAGE_ORDER

FIGURE_DPI = 300

_MISSING = "—"


def render_nine_run_metrics_table_png(table_df: pd.DataFrame, output_path: Path) -> None:
    """
    table_df columns: structural_level, stage, tm_score, plddt
    (9 rows: levels 0,1,2 × Baseline, Wasserstein, RKHS).
    """
    required = {"structural_level", "stage", "tm_score", "plddt"}
    if not required.issubset(set(table_df.columns)):
        raise ValueError(f"need columns {required}, got {list(table_df.columns)}")
    if len(table_df) != 9:
        raise ValueError(f"expected 9 rows, got {len(table_df)}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    col_widths = [0.72, 1.25, 1.35, 1.25]
    row_h = 0.52
    header_h = 0.62
    n_rows = len(table_df)
    fig_w = sum(col_widths)
    fig_h = header_h + n_rows * row_h + 0.22
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=FIGURE_DPI, facecolor="none")
    ax.set_xlim(0.0, fig_w)
    ax.set_ylim(0.0, fig_h)
    ax.axis("off")

    headers = ["Level", "Stage", "TM-score", "pLDDT"]
    x0 = 0.0
    y_top = fig_h - 0.10
    for w, label in zip(col_widths, headers):
        rect = plt.Rectangle(
            (x0, y_top - header_h),
            w,
            header_h,
            facecolor=(0.85, 0.85, 0.85, 0.98),
            edgecolor="#222222",
            linewidth=1.0,
        )
        ax.add_patch(rect)
        ax.text(
            x0 + w * 0.5,
            y_top - header_h * 0.5,
            label,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
        x0 += w

    def _cell(s: str) -> str:
        t = str(s).strip()
        return _MISSING if t == "" else t

    current_y = y_top - header_h
    for level in (0, 1, 2):
        group = table_df[table_df["structural_level"].astype(int) == level]
        if len(group) != 3:
            raise ValueError(f"expected 3 rows for level {level}, got {len(group)}")
        level_h = row_h * 3.0
        drect = plt.Rectangle(
            (0.0, current_y - level_h),
            col_widths[0],
            level_h,
            facecolor=(0.92, 0.92, 0.92, 0.95),
            edgecolor="#222222",
            linewidth=1.0,
        )
        ax.add_patch(drect)
        ax.text(
            col_widths[0] * 0.5,
            current_y - level_h * 0.5,
            str(level),
            ha="center",
            va="center",
            fontsize=10,
        )

        row_y = current_y
        for st in SMOKE_STAGE_ORDER:
            match = group[group["stage"] == st]
            if match.empty:
                raise ValueError(f"missing stage {st!r} for level {level}")
            row = match.iloc[0]

            entries = [
                str(row["stage"]),
                _cell(str(row["tm_score"])),
                _cell(str(row["plddt"])),
            ]
            x = col_widths[0]
            for idx, txt in enumerate(entries):
                w = col_widths[idx + 1]
                rect = plt.Rectangle(
                    (x, row_y - row_h),
                    w,
                    row_h,
                    facecolor=(1.0, 1.0, 1.0, 0.92),
                    edgecolor="#222222",
                    linewidth=1.0,
                )
                ax.add_patch(rect)
                ax.text(x + w * 0.5, row_y - row_h * 0.5, txt, ha="center", va="center", fontsize=9)
                x += w
            row_y -= row_h
        current_y -= level_h

    fig.savefig(output_path, transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

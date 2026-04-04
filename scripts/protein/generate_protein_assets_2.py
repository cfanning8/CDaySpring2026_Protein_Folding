"""
Generate result assets under results/protein/figures/assets_2/:
  EDA (contact maps, Ramachandran, Rg, corpus stats via biotite),
  pipeline visual mosaic (figures, not prose),
  native 3D PD + virtual 3D PD only (same engine style as project figures),
  cartoon = SSE-colored 3D backbone (helix/sheet/coil),
  compound = 3D native + 3D pred + pLDDT + 3D virtual PD panel.

Run from repo root: python -u scripts/protein/generate_protein_assets_2.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "figures"))

from src.protein.dataset_policy import CONTACT_GRAPH_RADIUS_MAX_A  # noqa: E402
from src.protein.smoke_metrics import (  # noqa: E402
    find_colabfold_ranked_pdb,
    load_ca_coords_b_factors_from_pdb,
    mean_plddt_from_pdb,
)
from src.protein.topology_cache import build_edges_and_persistence, load_topology_npz  # noqa: E402

FIG_DPI = 300

ASSETS = PROJECT_ROOT / "results" / "protein" / "figures" / "assets_2"
LAYER_A = PROJECT_ROOT / "results" / "protein" / "tables" / "smoke" / "layer_a_structure.csv"
LAYER_B = PROJECT_ROOT / "results" / "protein" / "tables" / "smoke" / "layer_b_topology.csv"
METRICS = PROJECT_ROOT / "results" / "protein" / "tables" / "smoke" / "smoke_per_structure_metrics.csv"
COLABFOLD_PRED = PROJECT_ROOT / "results" / "protein" / "output" / "colabfold_smoke" / "predictions"

try:
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
except ImportError:
    sns = None

# SSE colors (P-SEA): helix / sheet / coil
_SSE_COLOR = {"a": "#e74c3c", "b": "#2980b9", "c": "#7f8c8d"}


def _load_chain_atom_array(mmcif_path: Path, chain_id: str):
    import biotite.structure as struc  # noqa: PLC0415
    import biotite.structure.io.pdbx as pdbx  # noqa: PLC0415

    cif_file = pdbx.CIFFile.read(str(mmcif_path))
    arr = pdbx.get_structure(cif_file, model=1)
    arr = arr[arr.chain_id.astype(str) == chain_id]
    arr = arr[struc.filter_amino_acids(arr)]
    if arr.array_length() == 0:
        raise ValueError("empty chain after filters")
    return arr


def _ca_coords_and_sse(mmcif_path: Path, chain_id: str) -> tuple[np.ndarray, np.ndarray]:
    import biotite.structure as struc  # noqa: PLC0415
    from biotite.structure.sse import annotate_sse  # noqa: PLC0415

    arr = _load_chain_atom_array(mmcif_path, chain_id)
    sse_full = annotate_sse(arr)
    ca = arr[arr.atom_name == "CA"]
    coords = np.asarray(ca.coord, dtype=np.float64)
    n = min(len(coords), len(sse_full))
    return coords[:n], np.asarray(sse_full[:n])


def _intervals_to_diagram4(pers: np.ndarray) -> np.ndarray:
    from collections import defaultdict

    pers = np.asarray(pers, dtype=np.float64)
    if pers.size == 0:
        return np.zeros((0, 4), dtype=np.float64)
    acc: dict[tuple[int, float, float], int] = defaultdict(int)
    rep: dict[tuple[int, float, float], tuple[float, float, int]] = {}
    for row in pers:
        dim, b, d = int(row[0]), float(row[1]), float(row[2])
        if not (np.isfinite(b) and np.isfinite(d) and d > b):
            continue
        key = (dim, round(b, 5), round(d, 5))
        acc[key] += 1
        rep[key] = (b, d, dim)
    out: list[list[float]] = []
    for k, m in acc.items():
        b, d, dim = rep[k]
        out.append([float(dim), b, d, float(m)])
    return np.asarray(out, dtype=np.float64) if out else np.zeros((0, 4))


def _ph_from_ca(coords: np.ndarray, *, graph_mode: str = "ca_legacy", max_dim: int = 1) -> np.ndarray:
    _, _, _, ptab, _ = build_edges_and_persistence(
        np.asarray(coords, dtype=np.float64),
        float(CONTACT_GRAPH_RADIUS_MAX_A),
        graph_mode=graph_mode,
        max_dimension=max_dim,
    )
    return np.asarray(ptab, dtype=np.float64)


def _trim_pair(native: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(native.shape[0], pred.shape[0])
    return native[:n], pred[:n]


def _ensure_dirs() -> dict[str, Path]:
    sub = ("eda", "pipeline", "topology", "virtual_pd", "compound", "cartoon")
    out = {}
    for s in sub:
        p = ASSETS / s
        p.mkdir(parents=True, exist_ok=True)
        out[s] = p
    return out


def _radius_of_gyration(ca: np.ndarray) -> float:
    c = np.asarray(ca, dtype=np.float64)
    com = c.mean(axis=0)
    d = np.linalg.norm(c - com, axis=1)
    return float(np.sqrt(np.mean(d**2)))


def fig_eda(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    df_m: pd.DataFrame | None,
    mmcif_ex: Path,
    chain_ex: str,
    pdb_ex: str,
    d: dict[str, Path],
) -> None:
    """Corpus-wide stats + exemplar contact map + Ramachandran (biotite)."""
    # 01 — length distribution (KDE if seaborn)
    fig = plt.figure(figsize=(8, 4.5), dpi=FIG_DPI)
    ax = fig.add_subplot(111)
    lens = df_a["n_residues"].astype(float).to_numpy()
    if sns is not None:
        sns.histplot(lens, bins=min(20, max(6, len(df_a) // 2)), kde=True, color="#34495e", ax=ax)
    else:
        ax.hist(lens, bins=min(20, max(6, len(df_a) // 2)), color="#34495e", edgecolor="white", density=True)
        ax.set_ylabel("Density")
    ax.set_xlabel("Residues per chain (CA count)")
    ax.set_ylabel("Count" if sns is None else None)
    ax.set_title("Corpus: chain length distribution")
    fig.tight_layout()
    fig.savefig(d["eda"] / "01_chain_length_distribution.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    # 02 — residues vs edges
    fig, ax = plt.subplots(figsize=(7, 5), dpi=FIG_DPI)
    m = df_a.merge(df_b, on="pdb_id", how="inner")
    ax.scatter(m["n_residues"], m["n_edges"], c="#c0392b", s=42, alpha=0.85, edgecolors="k", linewidths=0.25)
    ax.set_xlabel("Residues")
    ax.set_ylabel("Contact-graph edges")
    ax.set_title("Graph complexity vs chain length")
    fig.tight_layout()
    fig.savefig(d["eda"] / "02_residues_vs_edges.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    # 03 — Rg distribution (folding papers often report compactness)
    rgs: list[float] = []
    for _, row in df_a.iterrows():
        p = Path(str(row["mmcif_path"]))
        ch = str(row["chain_id"])
        if not p.is_file():
            continue
        try:
            ca, _ = _ca_coords_and_sse(p, ch)
            rgs.append(_radius_of_gyration(ca))
        except (OSError, ValueError):
            continue
    if rgs:
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=FIG_DPI)
        if sns is not None:
            sns.histplot(np.asarray(rgs), bins=min(16, len(rgs)), kde=True, color="#16a085", ax=ax)
        else:
            ax.hist(rgs, bins=min(16, len(rgs)), color="#16a085", edgecolor="white")
        ax.set_xlabel(r"Radius of gyration $R_g$ (Å)")
        ax.set_ylabel("Structures")
        ax.set_title(r"Corpus: $R_g$ from native CA (compactness)")
        fig.tight_layout()
        fig.savefig(d["eda"] / "03_radius_of_gyration.png", dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)

    # 04 — contact map (Cα distance) for exemplar
    try:
        ca_ex, _ = _ca_coords_and_sse(mmcif_ex, chain_ex)
        dist = np.linalg.norm(ca_ex[:, None, :] - ca_ex[None, :, :], axis=-1)
        fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=FIG_DPI)
        im = ax.imshow(dist, cmap="afmhot_r", vmin=0.0, vmax=min(20.0, float(np.percentile(dist, 99))))
        plt.colorbar(im, ax=ax, fraction=0.046, label="Distance (Å)")
        ax.set_xlabel("Residue j")
        ax.set_ylabel("Residue i")
        ax.set_title(f"Cα contact map — {pdb_ex}")
        fig.tight_layout()
        fig.savefig(d["eda"] / f"04_contact_map_{pdb_ex}.png", dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
    except (OSError, ValueError):
        pass

    # 05 — Ramachandran (φ, ψ) for exemplar
    try:
        import biotite.structure as struc  # noqa: PLC0415

        arr = _load_chain_atom_array(mmcif_ex, chain_ex)
        phi, psi, _ = struc.dihedral_backbone(arr)
        phi_d = np.rad2deg(phi)
        psi_d = np.rad2deg(psi)
        mask = np.isfinite(phi_d) & np.isfinite(psi_d)
        fig, ax = plt.subplots(figsize=(6.5, 6), dpi=FIG_DPI)
        ax.hexbin(
            phi_d[mask],
            psi_d[mask],
            gridsize=55,
            cmap="Blues",
            mincnt=1,
            linewidths=0,
        )
        ax.axhline(0, color="#555", lw=0.6)
        ax.axvline(0, color="#555", lw=0.6)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$\phi$ (deg)")
        ax.set_ylabel(r"$\psi$ (deg)")
        ax.set_title(f"Ramachandran — {pdb_ex} (backbone dihedrals)")
        fig.tight_layout()
        fig.savefig(d["eda"] / f"05_ramachandran_{pdb_ex}.png", dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
    except (OSError, ValueError):
        pass

    if (
        df_m is not None
        and not df_m.empty
        and "mean_plddt_colabfold" in df_m.columns
        and "rmsd_ca_kabsch_angstrom" in df_m.columns
    ):
        fig, ax = plt.subplots(figsize=(7, 5), dpi=FIG_DPI)
        x = df_m["rmsd_ca_kabsch_angstrom"].astype(float)
        y = df_m["mean_plddt_colabfold"].astype(float)
        ax.scatter(x, y, c="#27ae60", s=48, alpha=0.88, edgecolors="k", linewidths=0.2)
        ax.set_xlabel("CA RMSD native vs ColabFold (Å)")
        ax.set_ylabel("Mean pLDDT")
        ax.set_title("Model confidence vs CA deviation")
        fig.tight_layout()
        fig.savefig(d["eda"] / "06_rmsd_vs_plddt.png", dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)


def fig_pipeline_visual_mosaic(pdb_id: str, d: dict[str, Path]) -> None:
    """Four-image storyboard: cartoon → contact → native PH → virtual PH (minimal labels)."""
    paths = [
        d["cartoon"] / f"sse_backbone_3d_{pdb_id}.png",
        d["eda"] / f"04_contact_map_{pdb_id}.png",
        d["topology"] / f"native_ph_3d_{pdb_id}.png",
        d["virtual_pd"] / f"virtual_pd_3d_{pdb_id}_native_minus_pred.png",
    ]
    imgs = []
    for p in paths:
        if not p.is_file():
            return
        imgs.append(mpimg.imread(p))

    fig = plt.figure(figsize=(16, 4.2), dpi=FIG_DPI)
    gs = GridSpec(1, 7, figure=fig, width_ratios=[1, 0.12, 1, 0.12, 1, 0.12, 1], wspace=0.05)
    titles = ("Structure (SSE)", "Contacts", "Native PH (3D)", "Virtual PH (3D)")
    for i, (im, title) in enumerate(zip(imgs, titles)):
        ax = fig.add_subplot(gs[0, i * 2])
        ax.imshow(im)
        ax.axis("off")
        ax.set_title(title, fontsize=10, pad=6)
    for j in range(3):
        axa = fig.add_subplot(gs[0, j * 2 + 1])
        axa.axis("off")
        axa.text(0.5, 0.5, "→", ha="center", va="center", fontsize=22, color="#2c3e50", transform=axa.transAxes)
    fig.suptitle(f"Pipeline — {pdb_id}", fontsize=11, fontweight="bold", y=1.02)
    fig.savefig(d["pipeline"] / f"01_visual_mosaic_{pdb_id}.png", dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_backbone_3d_sse(ax, coords: np.ndarray, sse: np.ndarray, *, title: str, elev: float = 20, azim: float = -65) -> None:
    coords = np.asarray(coords, dtype=np.float64)
    sse = np.asarray(sse)
    n = coords.shape[0]
    if n < 2:
        return
    for i in range(n - 1):
        s0, s1 = str(sse[i]), str(sse[i + 1])
        col0 = _SSE_COLOR.get(s0, "#bdc3c7")
        col1 = _SSE_COLOR.get(s1, "#bdc3c7")
        # blend segment color
        if s0 == s1:
            col = col0
        else:
            col = col0
        ax.plot(
            coords[i : i + 2, 0],
            coords[i : i + 2, 1],
            coords[i : i + 2, 2],
            color=col,
            lw=3.2,
            solid_capstyle="round",
        )
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c="#2c3e50", s=8, alpha=0.9, zorder=3)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_zlabel("z (Å)")
    ax.view_init(elev=elev, azim=azim)


def plot_backbone_3d_plddt(ax, coords: np.ndarray, plddt: np.ndarray | None, *, title: str, elev: float = 20, azim: float = -65) -> None:
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    if plddt is None or plddt.size < n:
        sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=np.arange(n), cmap="viridis", s=10)
        plt.colorbar(sc, ax=ax, shrink=0.5, label="index")
    else:
        pl = np.asarray(plddt[:n], dtype=np.float64)
        if np.nanmax(pl) <= 1.5:
            pl = pl * 100.0
        sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=pl, cmap="RdYlGn", s=12, vmin=50, vmax=100)
        plt.colorbar(sc, ax=ax, shrink=0.5, label="pLDDT")
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color="#333", lw=0.35, alpha=0.45)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_zlabel("z (Å)")
    ax.view_init(elev=elev, azim=azim)


def fig_cartoon_sse_3d(mmcif_path: Path, chain_id: str, pdb_id: str, d: dict[str, Path]) -> None:
    coords, sse = _ca_coords_and_sse(mmcif_path, chain_id)
    fig = plt.figure(figsize=(8, 7), dpi=FIG_DPI)
    ax = fig.add_subplot(111, projection="3d")
    plot_backbone_3d_sse(ax, coords, sse, title=f"{pdb_id} — backbone (SSE: α / β / coil)")
    h = [
        mpatches.Patch(color=_SSE_COLOR["a"], label="α-helix"),
        mpatches.Patch(color=_SSE_COLOR["b"], label="β-sheet"),
        mpatches.Patch(color=_SSE_COLOR["c"], label="coil"),
    ]
    ax.legend(handles=h, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(d["cartoon"] / f"sse_backbone_3d_{pdb_id}.png", dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_virtual_and_native_for_pdb(
    pdb_id: str,
    mmcif_path: Path,
    chain_id: str,
    npz_path: Path,
    pred_pdb: Path | None,
    d: dict[str, Path],
) -> dict[str, Any]:
    from persistence_diagram_3d_engine import (  # noqa: PLC0415
        render_persistence_diagram_3d,
        render_virtual_persistence_diagram_3d,
    )

    loaded = load_topology_npz(npz_path)
    nat_pers = np.asarray(loaded["persistence"], dtype=np.float64)
    nat4 = _intervals_to_diagram4(nat_pers)

    out: dict[str, Any] = {"pdb_id": pdb_id, "has_pred": pred_pdb is not None}
    n = nat_pers.shape[0]
    mult = np.ones((n, 1))
    arr4 = np.hstack([nat_pers, mult])
    fin = np.isfinite(arr4[:, 1]) & np.isfinite(arr4[:, 2]) & (arr4[:, 2] > arr4[:, 1])
    arr4 = arr4[fin]
    if len(arr4):
        bmin, bmax = float(np.min(arr4[:, 1])), float(np.max(arr4[:, 1]))
        dmin, dmax = float(np.min(arr4[:, 2])), float(np.max(arr4[:, 2]))
        span = max(bmax - bmin, dmax - dmin, 1e-9)
        pad = 0.06 * span
        render_persistence_diagram_3d(
            arr4,
            d["topology"] / f"native_ph_3d_{pdb_id}.png",
            xlim=(bmin - pad, bmax + pad),
            ylim=(dmin - pad, dmax + pad),
            figsize=(9, 9),
            bar_scale=1.0,
            elev=22,
            azim=-58,
        )

    if pred_pdb is None or not pred_pdb.is_file():
        return out

    nat_ca = load_ca_coords_from_mmcif_safe(mmcif_path, chain_id)
    pred_ca, _ = load_ca_coords_b_factors_from_pdb(pred_pdb)
    nat_ca, pred_ca = _trim_pair(nat_ca, pred_ca)
    pred_pers = _ph_from_ca(pred_ca)
    pred4 = _intervals_to_diagram4(pred_pers)

    comb = np.vstack([nat4, pred4]) if len(nat4) and len(pred4) else (nat4 if len(nat4) else pred4)
    if len(comb):
        bmin, bmax = float(np.min(comb[:, 1])), float(np.max(comb[:, 1]))
        dmin, dmax = float(np.min(comb[:, 2])), float(np.max(comb[:, 2]))
        span = max(bmax - bmin, dmax - dmin, 1e-9)
        pad = 0.08 * span
        xlim = (bmin - pad, bmax + pad)
        ylim = (dmin - pad, dmax + pad)
    else:
        xlim, ylim = (0, 1), (0, 1)

    # Virtual persistence: 3D only (signed multiplicity bars), two viewing angles
    render_virtual_persistence_diagram_3d(
        nat4,
        pred4,
        d["virtual_pd"] / f"virtual_pd_3d_{pdb_id}_native_minus_pred.png",
        xlim=xlim,
        ylim=ylim,
        figsize=(9, 9),
        bar_scale=0.85,
        elev=22,
        azim=-58,
    )
    render_virtual_persistence_diagram_3d(
        nat4,
        pred4,
        d["virtual_pd"] / f"virtual_pd_3d_{pdb_id}_native_minus_pred_alt.png",
        xlim=xlim,
        ylim=ylim,
        figsize=(9, 9),
        bar_scale=0.85,
        elev=36,
        azim=42,
    )
    out["n_native_intervals"] = int(len(nat_pers))
    out["n_pred_intervals"] = int(len(pred_pers))
    return out


def load_ca_coords_from_mmcif_safe(mmcif_path: Path, chain_id: str) -> np.ndarray:
    from src.protein.mmcif_io import load_ca_coords_from_mmcif  # noqa: PLC0415

    return load_ca_coords_from_mmcif(mmcif_path, chain_id=chain_id)


def fig_compound_overview(
    pdb_id: str,
    mmcif_path: Path,
    chain_id: str,
    pred_pdb: Path | None,
    d: dict[str, Path],
) -> None:
    """2×2: 3D native SSE | 3D pred+pLDDT | pLDDT curve | virtual PD (3D render PNG)."""
    nat, sse = _ca_coords_and_sse(mmcif_path, chain_id)
    vpd_path = d["virtual_pd"] / f"virtual_pd_3d_{pdb_id}_native_minus_pred.png"

    fig = plt.figure(figsize=(14, 12), dpi=FIG_DPI)
    gs = GridSpec(2, 2, figure=fig, hspace=0.22, wspace=0.18)

    ax0 = fig.add_subplot(gs[0, 0], projection="3d")
    plot_backbone_3d_sse(ax0, nat, sse, title="Native Cα — SSE coloring", elev=18, azim=-60)

    ax1 = fig.add_subplot(gs[0, 1], projection="3d")
    pred_bf: np.ndarray | None = None
    pred_trim: np.ndarray | None = None
    if pred_pdb is not None and pred_pdb.is_file():
        pred_raw, bf_raw = load_ca_coords_b_factors_from_pdb(pred_pdb)
        pred_trim, _nat_trim = _trim_pair(pred_raw, nat)
        pred_bf = np.asarray(bf_raw[: pred_trim.shape[0]], dtype=np.float64) if bf_raw is not None else None
        plot_backbone_3d_plddt(ax1, pred_trim, pred_bf, title="ColabFold Cα — pLDDT", elev=18, azim=-60)
    else:
        ax1.text2D(0.5, 0.5, "No prediction", transform=ax1.transAxes, ha="center")

    ax2 = fig.add_subplot(gs[1, 0])
    if pred_trim is not None and pred_bf is not None:
        n = pred_trim.shape[0]
        pl = pred_bf[:n].astype(np.float64)
        if np.nanmax(pl) <= 1.5:
            pl = pl * 100.0
        ax2.fill_between(np.arange(n), pl, alpha=0.22, color="#c0392b")
        ax2.plot(np.arange(n), pl, color="#c0392b", lw=1.0)
        ax2.set_ylabel("pLDDT")
        ax2.set_xlabel("Residue (aligned)")
        m = mean_plddt_from_pdb(pred_pdb) if pred_pdb else None
        ax2.set_title("Per-residue confidence" + (f" (mean {m:.1f})" if m else ""))
    elif pred_pdb is not None and pred_pdb.is_file():
        ax2.text(0.5, 0.5, "No pLDDT in PDB", ha="center", va="center", transform=ax2.transAxes)
    else:
        ax2.text(0.5, 0.5, "No pLDDT", ha="center", va="center", transform=ax2.transAxes)

    ax3 = fig.add_subplot(gs[1, 1])
    if vpd_path.is_file():
        ax3.imshow(mpimg.imread(vpd_path))
        ax3.axis("off")
        ax3.set_title("Virtual persistence (native − pred), 3D", fontsize=10)
    else:
        ax3.text(0.5, 0.5, "Virtual PD not available", ha="center", va="center", transform=ax3.transAxes)

    fig.suptitle(f"{pdb_id} — structure, confidence, topology", fontsize=12, fontweight="bold", y=0.98)
    fig.savefig(d["compound"] / f"overview_{pdb_id}_compound.png", dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_manifest(rows: list[dict[str, Any]], d: dict[str, Path]) -> None:
    pd.DataFrame(rows).to_csv(ASSETS / "assets_manifest.csv", index=False)
    txt = ASSETS / "ASSETS_INDEX.txt"
    lines = [
        "assets_2 — EDA (biotite), pipeline mosaic, 3D PH / virtual PH, SSE cartoons, compound panels.",
        "See assets_manifest.csv.",
        "",
    ]
    for r in rows:
        lines.append(f"{r.get('file','')}\t{r.get('category','')}\t{r.get('description','')}")
    txt.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    d = _ensure_dirs()
    manifest: list[dict[str, Any]] = []

    if not LAYER_A.is_file():
        raise SystemExit(f"Missing {LAYER_A}; run smoke pipeline first.")

    df_a = pd.read_csv(LAYER_A)
    df_b = pd.read_csv(LAYER_B) if LAYER_B.is_file() else pd.DataFrame()
    df_m = pd.read_csv(METRICS) if METRICS.is_file() else None

    row0 = df_a.iloc[0]
    pdb_ex = str(row0["pdb_id"])
    mmcif_ex = Path(str(row0["mmcif_path"]))
    chain_ex = str(row0["chain_id"])

    fig_eda(df_a, df_b, df_m, mmcif_ex, chain_ex, pdb_ex, d)

    manifest.extend(
        [
            {"file": "eda/01_chain_length_distribution.png", "category": "eda", "description": "Chain length hist (+KDE if seaborn)"},
            {"file": "eda/02_residues_vs_edges.png", "category": "eda", "description": "Residues vs contact edges"},
        ]
    )
    if (ASSETS / "eda" / "03_radius_of_gyration.png").is_file():
        manifest.append({"file": "eda/03_radius_of_gyration.png", "category": "eda", "description": r"Rg distribution (native CA)"})
    if (ASSETS / "eda" / f"04_contact_map_{pdb_ex}.png").is_file():
        manifest.append({"file": f"eda/04_contact_map_{pdb_ex}.png", "category": "eda", "description": "Cα distance map (exemplar)"})
    if (ASSETS / "eda" / f"05_ramachandran_{pdb_ex}.png").is_file():
        manifest.append({"file": f"eda/05_ramachandran_{pdb_ex}.png", "category": "eda", "description": "Ramachandran hexbin (exemplar)"})
    if (ASSETS / "eda" / "06_rmsd_vs_plddt.png").is_file():
        manifest.append({"file": "eda/06_rmsd_vs_plddt.png", "category": "eda", "description": "RMSD vs mean pLDDT"})

    candidates = [str(df_a.iloc[0]["pdb_id"])]
    if df_m is not None and not df_m.empty and "rmsd_ca_kabsch_angstrom" in df_m.columns:
        j = df_m["rmsd_ca_kabsch_angstrom"].astype(float).idxmin()
        best = str(df_m.loc[j, "pdb_id"])
        if best not in candidates:
            candidates.append(best)

    first_for_pipeline: str | None = None
    for pdb_id in candidates[:2]:
        row = df_a[df_a["pdb_id"].astype(str).str.upper() == pdb_id.upper()]
        if row.empty:
            continue
        row = row.iloc[0]
        mmcif = Path(str(row["mmcif_path"]))
        chain = str(row["chain_id"])
        brow = df_b[df_b["pdb_id"].astype(str).str.upper() == pdb_id.upper()]
        if brow.empty:
            continue
        npz_path = Path(str(brow.iloc[0]["npz_path"]))
        pred = find_colabfold_ranked_pdb(COLABFOLD_PRED, pdb_id)

        fig_cartoon_sse_3d(mmcif, chain, pdb_id, d)
        manifest.append({"file": f"cartoon/sse_backbone_3d_{pdb_id}.png", "category": "cartoon", "description": "3D Cα colored by P-SEA SSE (α/β/coil)"})

        meta = render_virtual_and_native_for_pdb(pdb_id, mmcif, chain, npz_path, pred, d)
        manifest.append({"file": f"topology/native_ph_3d_{pdb_id}.png", "category": "topology", "description": "Native clique PH (3D bars)"})
        if meta.get("has_pred"):
            manifest.append(
                {
                    "file": f"virtual_pd/virtual_pd_3d_{pdb_id}_native_minus_pred.png",
                    "category": "virtual_pd",
                    "description": "Virtual PD 3D (signed mult.) primary view",
                }
            )
            manifest.append(
                {
                    "file": f"virtual_pd/virtual_pd_3d_{pdb_id}_native_minus_pred_alt.png",
                    "category": "virtual_pd",
                    "description": "Virtual PD 3D alternate view",
                }
            )

        fig_compound_overview(pdb_id, mmcif, chain, pred, d)
        manifest.append({"file": f"compound/overview_{pdb_id}_compound.png", "category": "compound", "description": "3D SSE + 3D pLDDT + curve + VPD 3D"})

        if first_for_pipeline is None:
            first_for_pipeline = pdb_id

    if first_for_pipeline is not None:
        fig_pipeline_visual_mosaic(first_for_pipeline, d)
        manifest.append(
            {
                "file": f"pipeline/01_visual_mosaic_{first_for_pipeline}.png",
                "category": "pipeline",
                "description": "Visual storyboard: SSE → contacts → native PH → virtual PH",
            }
        )

    write_manifest(manifest, d)
    print(f"wrote assets under {ASSETS.resolve()}")


if __name__ == "__main__":
    main()

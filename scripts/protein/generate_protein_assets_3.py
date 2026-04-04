"""
Publication-oriented assets under results/protein/figures/assets_3/:

  structures_py3dmol/ — ribbon/cartoon (spectrum, pLDDT/b-factor), dual-panel board
  quantitative/       — seaborn/matplotlib: RMSD, pLDDT, joint plots, PH vs RMSD
  contact_maps/       — native vs predicted Cα distance matrices + |Δ|
  topology/           — native 3D persistence (matplotlib engine)
  virtual_pd/         — virtual 3D persistence (signed mult.)

PyMOL / ChimeraX are the usual choice for final journal figures; this script uses the
Python stack from requirements-protein (py3Dmol + Playwright PNG, biotite, seaborn).

Run from repo root: python -u scripts/protein/generate_protein_assets_3.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "figures"))

FIG_DPI = 300
ASSETS = PROJECT_ROOT / "results" / "protein" / "figures" / "assets_3"
LAYER_A = PROJECT_ROOT / "results" / "protein" / "tables" / "smoke" / "layer_a_structure.csv"
LAYER_B = PROJECT_ROOT / "results" / "protein" / "tables" / "smoke" / "layer_b_topology.csv"
METRICS = PROJECT_ROOT / "results" / "protein" / "tables" / "smoke" / "smoke_per_structure_metrics.csv"
COLABFOLD_PRED = PROJECT_ROOT / "results" / "protein" / "output" / "colabfold_smoke" / "predictions"

try:
    import seaborn as sns

    sns.set_theme(style="ticks", context="paper", font_scale=1.05)
except ImportError:
    sns = None


def _load_assets2_module():
    p = PROJECT_ROOT / "scripts" / "protein" / "generate_protein_assets_2.py"
    spec = importlib.util.spec_from_file_location("protein_assets_2", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _ensure_dirs() -> dict[str, Path]:
    keys = ("structures_py3dmol", "quantitative", "contact_maps", "topology", "virtual_pd")
    out: dict[str, Path] = {}
    for k in keys:
        p = ASSETS / k
        p.mkdir(parents=True, exist_ok=True)
        out[k] = p
    return out


def _screenshot_py3dmol_view(
    view: Any,
    png_path: Path,
    *,
    browser: Any,
    width: int,
    height: int,
    wait_ms: int,
) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fd, path_str = tempfile.mkstemp(suffix=".html")
    os.close(fd)
    tmp = Path(path_str)
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            view.write_html(fh)
        page = browser.new_page(viewport={"width": width, "height": height})
        try:
            page.goto(tmp.as_uri(), wait_until="networkidle", timeout=180_000)
            page.wait_for_selector("canvas", timeout=180_000)
            page.wait_for_timeout(int(wait_ms))
            page.locator("canvas").first.screenshot(path=str(png_path))
        finally:
            page.close()
    finally:
        tmp.unlink(missing_ok=True)


def _view_native_cartoon_spectrum(mmcif_text: str, chain: str, width: int, height: int) -> Any:
    import py3Dmol  # noqa: PLC0415

    v = py3Dmol.view(width=width, height=height)
    v.addModel(mmcif_text, "mmcif")
    v.setStyle({"chain": chain}, {"cartoon": {"color": "spectrum", "style": "oval"}})
    v.zoomTo()
    return v


def _view_pred_cartoon_plddt(pdb_text: str, width: int, height: int) -> Any:
    """AlphaFold-style: B-factors as pLDDT on cartoon (3Dmol 'bfactor' colorscheme)."""
    import py3Dmol  # noqa: PLC0415

    v = py3Dmol.view(width=width, height=height)
    v.addModel(pdb_text, "pdb")
    v.setStyle(
        {},
        {
            "cartoon": {
                "style": "oval",
                "colorscheme": "bfactor",
                "color": "bfactor",
            }
        },
    )
    v.zoomTo()
    return v


def _trim_ca_dist(ca: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
    return d


def fig_contact_maps(
    nat_ca: np.ndarray,
    pred_ca: np.ndarray,
    pdb_id: str,
    out_dir: Path,
) -> None:
    n = min(nat_ca.shape[0], pred_ca.shape[0])
    a, b = nat_ca[:n], pred_ca[:n]
    dn = _trim_ca_dist(a)
    dp = _trim_ca_dist(b)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), dpi=FIG_DPI)
    vmax = float(np.percentile(np.maximum(dn, dp), 99))
    for ax, mat, title in zip(
        axes,
        (dn, dp, np.abs(dn - dp)),
        ("Native Cα distances", "Predicted Cα distances", "|Δ| native − pred"),
    ):
        im = ax.imshow(mat, cmap="magma_r", vmin=0.0, vmax=max(vmax, 1e-6))
        ax.set_title(f"{pdb_id}: {title}", fontsize=9)
        ax.set_xlabel("j")
        ax.set_ylabel("i")
        plt.colorbar(im, ax=ax, fraction=0.046, label="Å")
    fig.tight_layout()
    fig.savefig(out_dir / f"contacts_{pdb_id}_native_pred_diff.png", dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_quantitative(df: pd.DataFrame, out_dir: Path) -> None:
    df = df.copy()
    if df.empty:
        return
    rmsd = df["rmsd_ca_kabsch_angstrom"].astype(float) if "rmsd_ca_kabsch_angstrom" in df.columns else None
    plddt = df["mean_plddt_colabfold"].astype(float) if "mean_plddt_colabfold" in df.columns else None

    if rmsd is not None and sns is not None:
        fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=FIG_DPI)
        sns.violinplot(y=rmsd, ax=ax, color="#5dade2", inner="box")
        ax.set_ylabel(r"CA RMSD native vs prediction (Å)")
        ax.set_title("Distribution over structures (smoke corpus)")
        fig.tight_layout()
        fig.savefig(out_dir / "01_rmsd_violin.png", dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    elif rmsd is not None:
        fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=FIG_DPI)
        ax.boxplot(rmsd, vert=True)
        ax.set_ylabel(r"CA RMSD (Å)")
        fig.tight_layout()
        fig.savefig(out_dir / "01_rmsd_violin.png", dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    if plddt is not None and sns is not None:
        fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=FIG_DPI)
        sns.violinplot(y=plddt, ax=ax, color="#f5b041", inner="box")
        ax.set_ylabel("Mean pLDDT (prediction)")
        ax.set_title("Confidence distribution")
        fig.tight_layout()
        fig.savefig(out_dir / "02_plddt_violin.png", dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    elif plddt is not None:
        fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=FIG_DPI)
        ax.boxplot(plddt, vert=True)
        ax.set_ylabel("Mean pLDDT")
        fig.tight_layout()
        fig.savefig(out_dir / "02_plddt_violin.png", dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    if rmsd is not None and plddt is not None:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=FIG_DPI)
        if sns is not None:
            sns.scatterplot(x=rmsd, y=plddt, ax=ax, s=55, alpha=0.85)
            sns.kdeplot(x=rmsd, y=plddt, ax=ax, levels=5, alpha=0.35, color="gray")
        else:
            ax.scatter(rmsd, plddt, alpha=0.85)
        ax.set_xlabel(r"CA RMSD (Å)")
        ax.set_ylabel("Mean pLDDT")
        ax.set_title("Confidence vs geometric deviation")
        fig.tight_layout()
        fig.savefig(out_dir / "03_rmsd_vs_plddt_scatter.png", dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    h0 = df["ph_h0_total_persistence_sum"].astype(float) if "ph_h0_total_persistence_sum" in df.columns else None
    if rmsd is not None and h0 is not None:
        fig, ax = plt.subplots(figsize=(6, 4.8), dpi=FIG_DPI)
        ax.scatter(rmsd, h0, c="#8e44ad", s=45, alpha=0.85, edgecolors="k", linewidths=0.2)
        ax.set_xlabel(r"CA RMSD (Å)")
        ax.set_ylabel("Σ persistence (H0)")
        ax.set_title("Total H0 persistence vs RMSD")
        fig.tight_layout()
        fig.savefig(out_dir / "04_ph_h0_vs_rmsd.png", dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def render_py3dmol_board(
    mmcif_path: Path,
    pred_pdb: Path | None,
    chain: str,
    pdb_id: str,
    out_dir: Path,
    *,
    width: int,
    height: int,
    wait_ms: int,
) -> list[str]:
    """Returns list of relative filenames written (empty if Playwright/py3Dmol unavailable)."""
    try:
        import py3Dmol  # noqa: F401, PLC0415
        from playwright.sync_api import sync_playwright  # noqa: PLC0415
    except ImportError:
        return []

    mmcif_text = mmcif_path.read_text(encoding="utf-8", errors="replace")
    written: list[str] = []

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                v_nat = _view_native_cartoon_spectrum(mmcif_text, chain, width, height)
                p_nat = out_dir / f"{pdb_id}_native_cartoon_spectrum.png"
                _screenshot_py3dmol_view(v_nat, p_nat, browser=browser, width=width, height=height, wait_ms=wait_ms)
                written.append(str(p_nat.relative_to(ASSETS)))

                if pred_pdb is not None and pred_pdb.is_file():
                    pdb_text = pred_pdb.read_text(encoding="utf-8", errors="replace")
                    v_pred = _view_pred_cartoon_plddt(pdb_text, width, height)
                    p_pred = out_dir / f"{pdb_id}_pred_cartoon_plddt.png"
                    _screenshot_py3dmol_view(v_pred, p_pred, browser=browser, width=width, height=height, wait_ms=wait_ms)
                    written.append(str(p_pred.relative_to(ASSETS)))

                    fig = plt.figure(figsize=(14, 5.2), dpi=FIG_DPI)
                    gs = GridSpec(1, 2, figure=fig, wspace=0.08)
                    ax0 = fig.add_subplot(gs[0, 0])
                    ax1 = fig.add_subplot(gs[0, 1])
                    ax0.imshow(mpimg.imread(p_nat))
                    ax0.axis("off")
                    ax0.set_title("Native (mmCIF) — cartoon spectrum", fontsize=10)
                    ax1.imshow(mpimg.imread(p_pred))
                    ax1.axis("off")
                    ax1.set_title("ColabFold — cartoon pLDDT (B-factor)", fontsize=10)
                    fig.suptitle(f"{pdb_id} structure comparison (py3Dmol)", fontsize=11, fontweight="bold")
                    board = out_dir / f"{pdb_id}_board_native_pred.png"
                    fig.savefig(board, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
                    plt.close(fig)
                    written.append(str(board.relative_to(ASSETS)))
            finally:
                browser.close()
    except Exception as exc:
        note = ASSETS / "structures_py3dmol" / "PLAYWRIGHT_SKIP.txt"
        note.parent.mkdir(parents=True, exist_ok=True)
        note.write_text(
            "py3Dmol PNG export was skipped (Playwright/Chromium missing or error).\n"
            f"Reason: {exc!s}\n"
            "Install: pip install playwright && python -m playwright install chromium\n",
            encoding="utf-8",
        )

    return written


def write_readme() -> None:
    text = """assets_3 — publication-style figures (Python stack)

STRUCTURES (structures_py3dmol/)
  py3Dmol + Playwright: ribbon/cartoon with spectrum (native) and b-factor/pLDDT
  coloring (prediction). For journal submission, many groups refine the same
  structures in PyMOL or UCSF ChimeraX (lighting, transparent surfaces, insets).

QUANTITATIVE (quantitative/)
  matplotlib + seaborn: RMSD / pLDDT distributions, scatter, PH summary vs RMSD.

CONTACT MAPS (contact_maps/)
  Cα distance matrices: native, predicted, and absolute difference.

TOPOLOGY / VIRTUAL PD (topology/, virtual_pd/)
  Same 3D persistence engines as elsewhere in this repo (matplotlib).

Requirements: pip install -r requirements-protein.txt and:
  python -m playwright install chromium
"""
    (ASSETS / "README_assets_3.txt").write_text(text, encoding="utf-8")


def write_manifest(rows: list[dict[str, str]]) -> None:
    pd.DataFrame(rows).to_csv(ASSETS / "assets_manifest.csv", index=False)


def main() -> None:
    d = _ensure_dirs()
    manifest: list[dict[str, str]] = []
    if not LAYER_A.is_file():
        raise SystemExit(f"Missing {LAYER_A}; run smoke pipeline first.")

    df_a = pd.read_csv(LAYER_A)
    df_b = pd.read_csv(LAYER_B) if LAYER_B.is_file() else pd.DataFrame()
    df_m = pd.read_csv(METRICS) if METRICS.is_file() else pd.DataFrame()

    ga2 = _load_assets2_module()
    from src.protein.mmcif_io import load_ca_coords_from_mmcif  # noqa: PLC0415
    from src.protein.smoke_metrics import find_colabfold_ranked_pdb  # noqa: PLC0415

    if not df_m.empty:
        fig_quantitative(df_m, d["quantitative"])
        for fn, desc in [
            ("quantitative/01_rmsd_violin.png", "Violin/box: CA RMSD"),
            ("quantitative/02_plddt_violin.png", "Violin/box: mean pLDDT"),
            ("quantitative/03_rmsd_vs_plddt_scatter.png", "Scatter (+ optional KDE): RMSD vs pLDDT"),
            ("quantitative/04_ph_h0_vs_rmsd.png", "H0 total persistence vs RMSD"),
        ]:
            if (ASSETS / fn).is_file():
                manifest.append({"file": fn, "category": "quantitative", "description": desc})

    candidates = [str(df_a.iloc[0]["pdb_id"])]
    if not df_m.empty and "rmsd_ca_kabsch_angstrom" in df_m.columns:
        j = df_m["rmsd_ca_kabsch_angstrom"].astype(float).idxmin()
        best = str(df_m.loc[j, "pdb_id"])
        if best not in candidates:
            candidates.append(best)

    win_w, win_h = 1200, 1200
    wait_ms = 4500

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

        meta = ga2.render_virtual_and_native_for_pdb(pdb_id, mmcif, chain, npz_path, pred, d)
        manifest.append({"file": f"topology/native_ph_3d_{pdb_id}.png", "category": "topology", "description": "Native clique PH (3D)"})
        if meta.get("has_pred"):
            manifest.append({"file": f"virtual_pd/virtual_pd_3d_{pdb_id}_native_minus_pred.png", "category": "virtual_pd", "description": "Virtual PD 3D"})
            manifest.append({"file": f"virtual_pd/virtual_pd_3d_{pdb_id}_native_minus_pred_alt.png", "category": "virtual_pd", "description": "Virtual PD 3D alt view"})

        if mmcif.is_file() and pred is not None and pred.is_file():
            nat_ca = load_ca_coords_from_mmcif(mmcif, chain_id=chain)
            pred_ca, _ = ga2.load_ca_coords_b_factors_from_pdb(pred)
            nat_ca, pred_ca = ga2._trim_pair(nat_ca, pred_ca)
            fig_contact_maps(nat_ca, pred_ca, pdb_id, d["contact_maps"])
            manifest.append(
                {
                    "file": f"contact_maps/contacts_{pdb_id}_native_pred_diff.png",
                    "category": "contact_maps",
                    "description": "Native / pred / |Δ| Cα distance maps",
                }
            )

        py3_paths = render_py3dmol_board(
            mmcif,
            pred,
            chain,
            pdb_id,
            d["structures_py3dmol"],
            width=win_w,
            height=win_h,
            wait_ms=wait_ms,
        )
        for rel in py3_paths:
            manifest.append({"file": rel.replace("\\", "/"), "category": "structures_py3dmol", "description": "py3Dmol + Playwright"})

    write_readme()
    manifest.append({"file": "README_assets_3.txt", "category": "meta", "description": "Stack notes (PyMOL/ChimeraX vs Python)"})
    write_manifest(manifest)
    print(f"wrote assets under {ASSETS.resolve()}")


if __name__ == "__main__":
    main()

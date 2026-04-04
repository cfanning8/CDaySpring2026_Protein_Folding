"""
assets_4 — structure-centric figures ONLY (py3Dmol + optional PyMOL).

Every output is rendered through 3Dmol.js (Playwright PNG) and/or PyMOL ray tracing.
Composites are montages of those renders plus captions (metrics text), not abstract
matplotlib plots.

Wipes results/protein/figures/assets_4/ on each run.

Requirements: requirements-protein.txt, python -m playwright install chromium
Optional: PyMOL on PATH for pymol/ folder.

Run: python -u scripts/protein/generate_protein_assets_4.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.protein.smoke_metrics import find_colabfold_ranked_pdb  # noqa: E402

ASSETS = PROJECT_ROOT / "results" / "protein" / "figures" / "assets_4"
LAYER_A = PROJECT_ROOT / "results" / "protein" / "tables" / "smoke" / "layer_a_structure.csv"
METRICS = PROJECT_ROOT / "results" / "protein" / "tables" / "smoke" / "smoke_per_structure_metrics.csv"
COLABFOLD_PRED = PROJECT_ROOT / "results" / "protein" / "output" / "colabfold_smoke" / "predictions"

FIG_DPI = 200
CELL = 520
WAIT_MS = 5000


def _screenshot(
    view: Any,
    png_path: Path,
    *,
    browser: Any,
    width: int,
    height: int,
    wait_ms: int = WAIT_MS,
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


def _chain_sel(chain: str) -> dict[str, str]:
    return {"chain": chain}


def view_native_spectrum(mmcif_text: str, chain: str, w: int, h: int) -> Any:
    import py3Dmol  # noqa: PLC0415

    v = py3Dmol.view(width=w, height=h)
    v.addModel(mmcif_text, "mmcif")
    v.setStyle(_chain_sel(chain), {"cartoon": {"color": "spectrum", "style": "oval"}})
    v.zoomTo()
    return v


def view_pred_plddt(pdb_text: str, w: int, h: int) -> Any:
    import py3Dmol  # noqa: PLC0415

    v = py3Dmol.view(width=w, height=h)
    v.addModel(pdb_text, "pdb")
    v.setStyle(
        {},
        {"cartoon": {"style": "oval", "colorscheme": "bfactor", "color": "bfactor"}},
    )
    v.zoomTo()
    return v


def view_native_gray_plus_pred_plddt(mmcif_text: str, pdb_text: str, w: int, h: int) -> Any:
    """Single viewer: deposit (grey) + prediction (pLDDT) — structural relationship."""
    import py3Dmol  # noqa: PLC0415

    v = py3Dmol.view(width=w, height=h)
    v.addModel(mmcif_text, "mmcif")
    v.setStyle({"model": 0}, {"cartoon": {"color": "#c0c0c0", "style": "oval", "opacity": 0.75}})
    v.addModel(pdb_text, "pdb")
    v.setStyle(
        {"model": 1},
        {"cartoon": {"style": "oval", "colorscheme": "bfactor", "color": "bfactor"}},
    )
    v.zoomTo()
    return v


def view_pred_line_trace(pdb_text: str, w: int, h: int) -> Any:
    """Cα trace as line — complements cartoon (same pred, topology of backbone)."""
    import py3Dmol  # noqa: PLC0415

    v = py3Dmol.view(width=w, height=h)
    v.addModel(pdb_text, "pdb")
    v.setStyle({}, {"line": {"radius": 0.15, "colorscheme": "bfactor"}, "cartoon": {"hidden": True}})
    v.zoomTo()
    return v


def view_surface(mmcif_text: str, chain: str, w: int, h: int, color: str = "#6ec0ff") -> Any:
    import py3Dmol  # noqa: PLC0415

    v = py3Dmol.view(width=w, height=h)
    v.addModel(mmcif_text, "mmcif")
    sel = _chain_sel(chain)
    v.setStyle(sel, {"stick": {"hidden": True}, "sphere": {"hidden": True}, "line": {"hidden": True}})
    v.addSurface(py3Dmol.VDW, {"opacity": 0.92, "color": color}, sel)
    v.zoomTo()
    return v


def view_native_cartoon_plus_surface_split(mmcif_text: str, chain: str, w: int, h: int) -> Any:
    """viewergrid 1×2: cartoon | surface — same structure, two representations."""
    import py3Dmol  # noqa: PLC0415

    v = py3Dmol.view(width=w * 2, height=h, viewergrid=(1, 2))
    sel = _chain_sel(chain)
    v.addModel(mmcif_text, "mmcif", viewer=(0, 0))
    v.setStyle(sel, {"cartoon": {"color": "spectrum", "style": "oval"}}, viewer=(0, 0))
    v.zoomTo(viewer=(0, 0))
    v.addModel(mmcif_text, "mmcif", viewer=(0, 1))
    v.setStyle(
        sel,
        {"stick": {"hidden": True}, "sphere": {"hidden": True}, "line": {"hidden": True}},
        viewer=(0, 1),
    )
    v.addSurface(py3Dmol.VDW, {"opacity": 0.9, "color": "#88ccee"}, sel, viewer=(0, 1))
    v.zoomTo(viewer=(0, 1))
    return v


def view_corpus_extremes(
    small: tuple[str, str, str],
    large: tuple[str, str, str],
    w: int,
    h: int,
) -> Any:
    """viewergrid 1×2: smallest vs largest chain (by residue count) — scaling story."""
    import py3Dmol  # noqa: PLC0415

    sm_text, sm_ch, _sm_lbl = small
    lg_text, lg_ch, _lg_lbl = large
    v = py3Dmol.view(width=w * 2, height=h, viewergrid=(1, 2))
    v.addModel(sm_text, "mmcif", viewer=(0, 0))
    v.setStyle(_chain_sel(sm_ch), {"cartoon": {"color": "spectrum", "style": "oval"}}, viewer=(0, 0))
    v.zoomTo(viewer=(0, 0))
    v.addModel(lg_text, "mmcif", viewer=(0, 1))
    v.setStyle(_chain_sel(lg_ch), {"cartoon": {"color": "spectrum", "style": "oval"}}, viewer=(0, 1))
    v.zoomTo(viewer=(0, 1))
    return v


def montage_labeled(
    images: list,
    titles: list[str],
    suptitle: str,
    out_path: Path,
) -> None:
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.4), dpi=FIG_DPI)
    if n == 1:
        axes = [axes]
    for ax, im, t in zip(axes, images, titles):
        ax.imshow(im)
        ax.axis("off")
        ax.set_title(t, fontsize=10)
    fig.suptitle(suptitle, fontsize=11, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def grid2_montage(
    images: list[np.ndarray],
    row_titles: list[str],
    out_path: Path,
    suptitle: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=FIG_DPI)
    for ax, im, t in zip(axes.flat, images, row_titles):
        ax.imshow(im)
        ax.axis("off")
        ax.set_title(t, fontsize=9)
    fig.suptitle(suptitle, fontsize=11, fontweight="bold")
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def try_pymol_native_pred_png(
    mmcif_path: Path,
    pred_pdb: Path,
    chain: str,
    out_png: Path,
) -> bool:
    exe = shutil.which("pymol")
    if not exe:
        return False
    # PyMOL: load both; align; grey native; spectrum pred by b-factor
    script = f"""# auto
load {str(mmcif_path).replace(chr(92), "/")}, nat
load {str(pred_pdb).replace(chr(92), "/")}, pred
hide everything
show cartoon
color grey, nat
select br. pred and name CA
spectrum b, blue_red, pred
orient
set ray_trace_mode, 1
set antialias, 2
png {str(out_png).replace(chr(92), "/")}, width=1400, height=1400, dpi=200, ray=1
quit
"""
    fd, path_str = tempfile.mkstemp(suffix=".pml")
    os.close(fd)
    tmp = Path(path_str)
    try:
        tmp.write_text(script, encoding="utf-8")
        subprocess.run(
            [exe, "-cq", str(tmp)],
            check=False,
            timeout=300,
            capture_output=True,
            text=True,
        )
    finally:
        tmp.unlink(missing_ok=True)
    return out_png.is_file()


def main() -> None:
    if not LAYER_A.is_file():
        raise SystemExit(f"Missing {LAYER_A}")

    if ASSETS.is_dir():
        shutil.rmtree(ASSETS)
    d_py = ASSETS / "py3dmol"
    d_pm = ASSETS / "pymol"
    d_comp = ASSETS / "composites"
    for p in (d_py, d_pm, d_comp, ASSETS):
        p.mkdir(parents=True, exist_ok=True)

    df_a = pd.read_csv(LAYER_A)
    df_m = pd.read_csv(METRICS) if METRICS.is_file() else pd.DataFrame()

    try:
        from playwright.sync_api import sync_playwright  # noqa: PLC0415
    except ImportError as e:
        raise SystemExit("Playwright required: pip install playwright && python -m playwright install chromium") from e

    manifest: list[dict[str, str]] = []

    # Exemplars: first row + lowest RMSD
    candidates: list[str] = [str(df_a.iloc[0]["pdb_id"])]
    if not df_m.empty and "rmsd_ca_kabsch_angstrom" in df_m.columns:
        j = df_m["rmsd_ca_kabsch_angstrom"].astype(float).idxmin()
        b = str(df_m.loc[j, "pdb_id"])
        if b not in candidates:
            candidates.append(b)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            for pdb_id in candidates[:3]:
                row = df_a[df_a["pdb_id"].astype(str).str.upper() == pdb_id.upper()]
                if row.empty:
                    continue
                row = row.iloc[0]
                mmcif = Path(str(row["mmcif_path"]))
                chain = str(row["chain_id"])
                if not mmcif.is_file():
                    continue
                mmcif_text = mmcif.read_text(encoding="utf-8", errors="replace")
                pred = find_colabfold_ranked_pdb(COLABFOLD_PRED, pdb_id)
                if pred is None or not pred.is_file():
                    continue
                pdb_text = pred.read_text(encoding="utf-8", errors="replace")

                mrow = df_m[df_m["pdb_id"].astype(str).str.upper() == pdb_id.upper()]
                rmsd = float(mrow.iloc[0]["rmsd_ca_kabsch_angstrom"]) if not mrow.empty else float("nan")
                plddt = float(mrow.iloc[0]["mean_plddt_colabfold"]) if not mrow.empty and "mean_plddt_colabfold" in mrow.columns else float("nan")
                cap = f"{pdb_id}  |  RMSD {rmsd:.2f} Å  |  mean pLDDT {plddt:.1f}"

                # 1) Native spectrum
                v1 = view_native_spectrum(mmcif_text, chain, CELL, CELL)
                p1 = d_py / f"{pdb_id}_01_native_spectrum.png"
                _screenshot(v1, p1, browser=browser, width=CELL, height=CELL)
                manifest.append({"file": str(p1.relative_to(ASSETS)), "kind": "py3dmol", "note": "native cartoon spectrum"})

                # 2) Pred pLDDT
                v2 = view_pred_plddt(pdb_text, CELL, CELL)
                p2 = d_py / f"{pdb_id}_02_pred_plddt.png"
                _screenshot(v2, p2, browser=browser, width=CELL, height=CELL)
                manifest.append({"file": str(p2.relative_to(ASSETS)), "kind": "py3dmol", "note": "ColabFold pLDDT cartoon"})

                # 3) Overlay two models
                v3 = view_native_gray_plus_pred_plddt(mmcif_text, pdb_text, CELL, CELL)
                p3 = d_py / f"{pdb_id}_03_overlay_native_grey_pred_plddt.png"
                _screenshot(v3, p3, browser=browser, width=CELL, height=CELL)
                manifest.append({"file": str(p3.relative_to(ASSETS)), "kind": "py3dmol", "note": "superposition native+pred"})

                # 4) Cartoon | surface split
                try:
                    v4 = view_native_cartoon_plus_surface_split(mmcif_text, chain, CELL, CELL)
                    p4 = d_py / f"{pdb_id}_04_cartoon_surface_split.png"
                    _screenshot(v4, p4, browser=browser, width=CELL * 2, height=CELL)
                    manifest.append({"file": str(p4.relative_to(ASSETS)), "kind": "py3dmol", "note": "cartoon vs surface same chain"})
                except Exception as ex:
                    (d_py / f"{pdb_id}_04_cartoon_surface_split_ERROR.txt").write_text(str(ex), encoding="utf-8")

                v5 = view_pred_line_trace(pdb_text, CELL, CELL)
                p5 = d_py / f"{pdb_id}_05_pred_line_plddt.png"
                _screenshot(v5, p5, browser=browser, width=CELL, height=CELL)
                manifest.append({"file": str(p5.relative_to(ASSETS)), "kind": "py3dmol", "note": "pred CA line + bfactor"})

                # 5) Triptych montage (pixels = py3Dmol only)
                im1 = mpimg.imread(p1)
                im2 = mpimg.imread(p2)
                im3 = mpimg.imread(p3)
                montage_labeled(
                    [im1, im2, im3],
                    ["Native (spectrum)", "Prediction (pLDDT)", "Overlay (grey + pLDDT)"],
                    cap,
                    d_comp / f"{pdb_id}_triptych_relationships.png",
                )
                manifest.append({"file": str((d_comp / f"{pdb_id}_triptych_relationships.png").relative_to(ASSETS)), "kind": "composite", "note": "montage of py3Dmol panels + caption"})

                # 6) Rotation 2×2 (reuse browser from outer loop - view_rotation_series opens own browser - inefficient)
                # Inline 4 shots with same browser:
                angles = [0, 35, 70, 105]
                rot_paths: list[Path] = []
                for i, ang in enumerate(angles):
                    import py3Dmol  # noqa: PLC0415

                    v = py3Dmol.view(width=CELL, height=CELL)
                    v.addModel(mmcif_text, "mmcif")
                    v.setStyle(_chain_sel(chain), {"cartoon": {"color": "spectrum", "style": "oval"}})
                    v.zoomTo()
                    if abs(ang) > 1e-6:
                        v.rotate(ang, "y")
                    rp = d_py / f"{pdb_id}_rot_{i}.png"
                    _screenshot(v, rp, browser=browser, width=CELL, height=CELL)
                    rot_paths.append(rp)
                imgs = [mpimg.imread(x) for x in rot_paths]
                grid2_montage(
                    imgs,
                    [f"view {i+1} (y-rot {a:.0f}°)" for i, a in enumerate(angles)],
                    d_comp / f"{pdb_id}_rotation_tour_2x2.png",
                    f"{pdb_id} — orientation tour (same structure)",
                )
                manifest.append({"file": str((d_comp / f"{pdb_id}_rotation_tour_2x2.png").relative_to(ASSETS)), "kind": "composite", "note": "2×2 py3Dmol rotations"})

        finally:
            browser.close()

    # Rotation series used separate browser in view_rotation_series - we inlined above

    # Corpus smallest vs largest (single viewergrid screenshot)
    if len(df_a) >= 2:
        df_s = df_a.sort_values("n_residues", ascending=True)
        df_l = df_a.sort_values("n_residues", ascending=False)
        small_row, large_row = df_s.iloc[0], df_l.iloc[0]
        if str(small_row["pdb_id"]) != str(large_row["pdb_id"]):
            sm_path = Path(str(small_row["mmcif_path"]))
            lg_path = Path(str(large_row["mmcif_path"]))
            sm_ch, lg_ch = str(small_row["chain_id"]), str(large_row["chain_id"])
            if sm_path.is_file() and lg_path.is_file():
                sm_txt = sm_path.read_text(encoding="utf-8", errors="replace")
                lg_txt = lg_path.read_text(encoding="utf-8", errors="replace")
                sm_lbl = f"{small_row['pdb_id']} ({int(small_row['n_residues'])} res)"
                lg_lbl = f"{large_row['pdb_id']} ({int(large_row['n_residues'])} res)"
                try:
                    import py3Dmol  # noqa: PLC0415
                    from playwright.sync_api import sync_playwright  # noqa: PLC0415

                    v = view_corpus_extremes(
                        (sm_txt, sm_ch, sm_lbl),
                        (lg_txt, lg_ch, lg_lbl),
                        CELL,
                        CELL,
                    )
                    with sync_playwright() as p:
                        browser = p.chromium.launch()
                        try:
                            outp = d_py / "corpus_smallest_vs_largest_cartoon.png"
                            _screenshot(v, outp, browser=browser, width=CELL * 2, height=CELL)
                            manifest.append({"file": str(outp.relative_to(ASSETS)), "kind": "py3dmol", "note": "size extremes"})
                        finally:
                            browser.close()
                except Exception as ex:
                    (d_py / "corpus_extremes_ERROR.txt").write_text(str(ex), encoding="utf-8")

    # PyMOL optional — one composite per first exemplar
    if candidates:
        pid = candidates[0]
        row = df_a[df_a["pdb_id"].astype(str).str.upper() == pid.upper()]
        pred = find_colabfold_ranked_pdb(COLABFOLD_PRED, pid)
        if not row.empty and pred is not None and pred.is_file():
            mmcif = Path(str(row.iloc[0]["mmcif_path"]))
            chain = str(row.iloc[0]["chain_id"])
            pm_out = d_pm / f"{pid}_pymol_ray_native_pred.png"
            if try_pymol_native_pred_png(mmcif, pred, chain, pm_out):
                manifest.append({"file": str(pm_out.relative_to(ASSETS)), "kind": "pymol", "note": "ray trace align"})

    (ASSETS / "README_assets_4.txt").write_text(
        """assets_4 (rebuilt): every asset is py3Dmol and/or PyMOL.

py3dmol/     Raw Playwright screenshots from 3Dmol.js (native, pred, overlay, splits, corpus).
composites/  Montages of those PNGs + titles (no abstract charts).
pymol/       Optional ray-traced PNG if `pymol` is on PATH.

Previous matplotlib-only figures were removed by design.
""",
        encoding="utf-8",
    )
    manifest.append({"file": "README_assets_4.txt", "kind": "meta", "note": "scope"})
    pd.DataFrame(manifest).to_csv(ASSETS / "assets_manifest.csv", index=False)
    print(f"wrote {len(manifest)} entries under {ASSETS.resolve()}")


if __name__ == "__main__":
    main()

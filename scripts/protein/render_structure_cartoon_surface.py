from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import py3Dmol

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.protein.mmcif_io import infer_primary_chain_id, load_ca_coords_from_mmcif  # noqa: E402


def _selection_chain(chain: str) -> dict[str, str]:
    return {"chain": chain}


def _build_cartoon_view(
    mmcif_text: str,
    chain: str,
    *,
    width: int,
    height: int,
    cartoon_thickness: float | None,
) -> py3Dmol.view:
    v = py3Dmol.view(width=width, height=height)
    v.addModel(mmcif_text, "mmcif")
    cartoon: dict[str, object] = {"color": "spectrum"}
    if cartoon_thickness is not None:
        cartoon["thickness"] = float(cartoon_thickness)
    v.setStyle(_selection_chain(chain), {"cartoon": cartoon})
    v.zoomTo()
    return v


def _build_surface_view(
    mmcif_text: str,
    chain: str,
    *,
    width: int,
    height: int,
    surface_color: str,
    surface_opacity: float,
) -> py3Dmol.view:
    v = py3Dmol.view(width=width, height=height)
    v.addModel(mmcif_text, "mmcif")
    sel = _selection_chain(chain)
    v.setStyle(
        sel,
        {
            "stick": {"hidden": True},
            "sphere": {"hidden": True},
            "line": {"hidden": True},
        },
    )
    v.addSurface(
        py3Dmol.VDW,
        {"opacity": float(surface_opacity), "color": surface_color},
        sel,
    )
    v.zoomTo()
    return v


def _screenshot_view_to_png(
    view: py3Dmol.view,
    png_path: Path,
    *,
    browser,
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


def process_mmcif(
    mmcif_path: Path,
    *,
    pdb_label: str | None,
    chain_id: str | None,
    out_dir: Path,
    window_size: tuple[int, int],
    cartoon_thickness: float | None,
    surface_color: str,
    surface_opacity: float,
    render_wait_ms: int,
) -> None:
    label = pdb_label or mmcif_path.stem.upper()
    chain = chain_id
    if chain is not None and len(chain) != 1:
        raise ValueError("chain_id must be one character if set")
    if chain is None:
        chain = infer_primary_chain_id(mmcif_path)

    # Validates the chain exists and matches mmcif_io conventions.
    _ = load_ca_coords_from_mmcif(mmcif_path, chain_id=chain)

    mmcif_text = mmcif_path.read_text(encoding="utf-8", errors="replace")
    w, h = window_size

    cartoon_path = out_dir / f"{label}_cartoon.png"
    surface_path = out_dir / f"{label}_surface.png"

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit(
            "Playwright is required for PNG export with py3Dmol. "
            "Install dependencies from requirements-protein.txt and run: "
            "python -m playwright install chromium"
        ) from exc

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            v_cartoon = _build_cartoon_view(
                mmcif_text,
                chain,
                width=w,
                height=h,
                cartoon_thickness=cartoon_thickness,
            )
            _screenshot_view_to_png(
                v_cartoon,
                cartoon_path,
                browser=browser,
                width=w,
                height=h,
                wait_ms=render_wait_ms,
            )
            v_surf = _build_surface_view(
                mmcif_text,
                chain,
                width=w,
                height=h,
                surface_color=surface_color,
                surface_opacity=surface_opacity,
            )
            _screenshot_view_to_png(
                v_surf,
                surface_path,
                browser=browser,
                width=w,
                height=h,
                wait_ms=render_wait_ms,
            )
        finally:
            browser.close()

    print(f"wrote={cartoon_path.resolve()}")
    print(f"wrote={surface_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render structure cartoon (py3Dmol secondary-structure cartoon) "
            "and VDW surface PNGs via headless Chromium (Playwright)."
        ),
    )
    parser.add_argument("--mmcif-path", type=Path, default=None)
    parser.add_argument(
        "--mmcif-dir",
        type=Path,
        default=None,
        help="Process every *.cif in this directory (mutually exclusive with --mmcif-path).",
    )
    parser.add_argument("--pdb-label", type=str, default=None, help="Output base name when using a single --mmcif-path.")
    parser.add_argument("--chain-id", type=str, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "protein" / "figures" / "structure_views",
    )
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1600)
    parser.add_argument(
        "--cartoon-thickness",
        type=float,
        default=None,
        help="Optional py3Dmol cartoon strand thickness (default: library default).",
    )
    parser.add_argument("--surface-color", type=str, default="#6ec0ff")
    parser.add_argument("--surface-opacity", type=float, default=1.0)
    parser.add_argument(
        "--render-wait-ms",
        type=int,
        default=5000,
        help="Extra time after canvas appears for WebGL/surface generation (ms).",
    )
    args = parser.parse_args()

    if (args.mmcif_path is None) == (args.mmcif_dir is None):
        raise SystemExit("set exactly one of --mmcif-path or --mmcif-dir")

    if args.mmcif_dir is not None:
        paths = sorted(args.mmcif_dir.glob("*.cif"))
        if not paths:
            raise SystemExit(f"no .cif files in {args.mmcif_dir.resolve()}")
    else:
        assert args.mmcif_path is not None
        paths = [args.mmcif_path]

    win = (int(args.width), int(args.height))
    for path in paths:
        process_mmcif(
            path,
            pdb_label=args.pdb_label if len(paths) == 1 else None,
            chain_id=args.chain_id,
            out_dir=args.out_dir,
            window_size=win,
            cartoon_thickness=args.cartoon_thickness,
            surface_color=str(args.surface_color),
            surface_opacity=float(args.surface_opacity),
            render_wait_ms=int(args.render_wait_ms),
        )


if __name__ == "__main__":
    main()

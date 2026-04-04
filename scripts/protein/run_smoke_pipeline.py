from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "protein"))

from smoke_persistence_3d import render_npz_persistence_3d  # noqa: E402
from smoke_table_render import render_nine_run_metrics_table_png  # noqa: E402
from src.protein.dataset_policy import (  # noqa: E402
    CONTACT_GRAPH_RADIUS_MAX_A,
    TOPOLOGY_GRAPH_POLICY_ID,
)
from src.protein.mmcif_io import infer_primary_chain_id, load_ca_coords_from_mmcif  # noqa: E402
from src.protein.residue_points import load_cb_primary_residue_coords_from_mmcif  # noqa: E402
from src.protein.smoke_metrics import (  # noqa: E402
    compute_per_structure_metrics,
    load_native_pred_plddt_aligned,
    nine_run_metrics_to_display_table,
)
from src.protein.smoke_nine_experiments import (  # noqa: E402
    build_empty_nine_experiment_rows,
    compute_nine_experiment_metrics,
    nine_rows_to_legacy_csv_dicts,
)
from src.protein.topology_cache import (  # noqa: E402
    build_edges_and_persistence,
    default_topology_npz_path,
    load_topology_npz,
    mmcif_content_fingerprint,
    save_topology_npz,
)

FIGURE_DPI = 300


def _corpus_mmcifs(corpus_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in ("*.cif", "*.mmcif"):
        paths.extend(corpus_dir.glob(pattern))
    return sorted({p.resolve() for p in paths})


def _colabfold_batch_path_probe() -> str | None:
    root = os.environ.get("LOCALCOLABFOLD_ROOT", "").strip()
    if root:
        p = Path(root).expanduser() / ".pixi" / "envs" / "default" / "bin" / "colabfold_batch"
        if p.is_file():
            return str(p)
    return shutil.which("colabfold_batch")


def _probe_colabfold() -> dict[str, object]:
    return {
        "platform": sys.platform,
        "colabfold_batch_on_path": _colabfold_batch_path_probe(),
        "LOCALCOLABFOLD_ROOT": os.environ.get("LOCALCOLABFOLD_ROOT", "").strip() or None,
        "layerC_inference": (
            "ColabFold runs AlphaFold2-class prediction via `colabfold_batch` (see "
            "https://github.com/sokrypton/ColabFold ). On Windows use WSL2 + LocalColabFold "
            "( https://github.com/YoshitakaMo/localcolabfold ); small jobs use the public MSA server."
        ),
    }


def _ensure_topology_npz(
    mmcif_path: Path,
    chain: str,
    *,
    force: bool,
    max_dimension: int,
    graph_mode: str,
) -> tuple[Path, float]:
    representative = "cb_primary" if graph_mode == "cb_topology" else "ca_legacy"
    out_path = default_topology_npz_path(
        mmcif_path,
        chain,
        policy_id=TOPOLOGY_GRAPH_POLICY_ID,
        representative=representative,
        radius_max_a=float(CONTACT_GRAPH_RADIUS_MAX_A),
        project_root=PROJECT_ROOT,
    )
    t0 = time.perf_counter()
    if out_path.is_file() and not force:
        return out_path, time.perf_counter() - t0

    from src.protein.mmcif_io import load_ca_coords_from_mmcif  # noqa: PLC0415

    if graph_mode == "cb_topology":
        coords = load_cb_primary_residue_coords_from_mmcif(mmcif_path, chain_id=chain)
    else:
        coords = load_ca_coords_from_mmcif(mmcif_path, chain_id=chain)

    sha = mmcif_content_fingerprint(mmcif_path)
    es, et, ef, ptab, _ = build_edges_and_persistence(
        coords,
        float(CONTACT_GRAPH_RADIUS_MAX_A),
        graph_mode=graph_mode,
        max_dimension=int(max_dimension),
    )
    save_topology_npz(
        out_path,
        topology_graph_policy_id=TOPOLOGY_GRAPH_POLICY_ID,
        mmcif_path=mmcif_path,
        mmcif_sha256=sha,
        chain_id=chain,
        representative=representative,
        radius_max_a=float(CONTACT_GRAPH_RADIUS_MAX_A),
        graph_mode=graph_mode,
        max_dimension=int(max_dimension),
        coords=coords,
        edges_source=es,
        edges_target=et,
        edges_filtration=ef,
        persistence_table=ptab,
    )
    return out_path, time.perf_counter() - t0


def _plot_layer_ab_bars_ax(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    ax,
) -> None:
    labels = df_a["pdb_id"].tolist()
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, df_a["n_residues"], w, label="residues", color="#4c72b0")
    ax.bar(x + w / 2, df_b["n_edges"], w, label="edges", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("count")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.3)


def _plot_persistence_count_bars_ax(df_b: pd.DataFrame, df_a: pd.DataFrame, ax) -> None:
    labels = df_a["pdb_id"].tolist()
    x = np.arange(len(labels))
    h0 = df_b["n_persistence_h0"].to_numpy()
    h1 = df_b["n_persistence_h1"].to_numpy()
    ax.bar(x, h0, label="H0", color="#1976d2")
    ax.bar(x, h1, bottom=h0, label="H1", color="#dc143c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("intervals")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.3)


def _figures_from_tables(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = df_a["pdb_id"].tolist()
    w = max(6.0, 0.45 * len(labels))

    fig, ax = plt.subplots(figsize=(w, 4.0), dpi=FIGURE_DPI, facecolor="none")
    ax.patch.set_alpha(0.0)
    _plot_layer_ab_bars_ax(df_a, df_b, ax)
    fig.tight_layout()
    fig.savefig(out_dir / "smoke_layer_a_b_counts.png", dpi=FIGURE_DPI, transparent=True, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(w, 4.0), dpi=FIGURE_DPI, facecolor="none")
    ax.patch.set_alpha(0.0)
    _plot_persistence_count_bars_ax(df_b, df_a, ax)
    fig.tight_layout()
    fig.savefig(out_dir / "smoke_persistence_by_dim.png", dpi=FIGURE_DPI, transparent=True, bbox_inches="tight")
    plt.close(fig)


def _coords_pca2d(coords: np.ndarray) -> np.ndarray:
    """First two principal components of centered coordinates (numpy only)."""
    x = np.asarray(coords, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("coords must be (n, d) with d>=2")
    xc = x - x.mean(axis=0)
    _, _, vh = np.linalg.svd(xc, full_matrices=False)
    return xc @ vh[:2].T


def _figures_persistence_3d_assets(df_b: pd.DataFrame, out_dir: Path, *, max_panels: int = 3) -> None:
    """One 3D PD PNG per PDB under figures/smoke/assets/ (shared engine with epi figures)."""
    if df_b.empty:
        return
    asset_dir = out_dir / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    for _, row in df_b.head(int(max_panels)).iterrows():
        pdb_id = str(row["pdb_id"])
        render_npz_persistence_3d(Path(row["npz_path"]), asset_dir / f"{pdb_id}_pd_3d.png")


def _figures_residue_graph_projections(df_b: pd.DataFrame, out_dir: Path, *, max_panels: int = 3) -> None:
    """Residue graph in 2D (PCA of Cβ/Cα coords); edges as faint segments."""
    if df_b.empty:
        return
    asset_dir = out_dir / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    sub = df_b.head(int(max_panels))
    n = len(sub)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.2), dpi=FIGURE_DPI, facecolor="none", squeeze=False)
    for i, (_, row) in enumerate(sub.iterrows()):
        ax = axes[0][i]
        ax.patch.set_alpha(0.0)
        loaded = load_topology_npz(Path(row["npz_path"]))
        coords = np.asarray(loaded["coords"], dtype=np.float64)
        es = np.asarray(loaded["edges_source"], dtype=np.int64)
        et = np.asarray(loaded["edges_target"], dtype=np.int64)
        xy = _coords_pca2d(coords)
        ne = int(es.shape[0])
        if ne > 8000:
            rng = np.random.default_rng(14)
            pick = rng.choice(ne, size=8000, replace=False)
            es, et = es[pick], et[pick]
        for e in range(es.shape[0]):
            a, b = int(es[e]), int(et[e])
            ax.plot(
                [xy[a, 0], xy[b, 0]],
                [xy[a, 1], xy[b, 1]],
                color="#7f8c8d",
                alpha=0.12,
                lw=0.35,
            )
        ax.scatter(xy[:, 0], xy[:, 1], s=6, c="#2980b9", alpha=0.85, zorder=3)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2, linestyle=":")
    fig.tight_layout()
    fig.savefig(asset_dir / "smoke_residue_graph_pca.png", dpi=FIGURE_DPI, transparent=True, bbox_inches="tight")
    plt.close(fig)


def _ab_status_strings(
    *,
    n_corpus: int,
    n_layer_ab_ok: int,
    n_parse_errors: int,
    graph_mode: str,
) -> tuple[str, str]:
    rep = "CA-only (ca_legacy)" if graph_mode == "ca_legacy" else "Cβ-primary (cb_topology)"
    if n_layer_ab_ok == n_corpus and n_parse_errors == 0:
        return "ok", f"all {n_corpus} mmCIF(s) passed frozen representative policy ({rep})"
    if n_layer_ab_ok > 0:
        return (
            "partial",
            f"{n_layer_ab_ok}/{n_corpus} mmCIF(s) passed; "
            f"{n_parse_errors} parse/representative skip(s) — see smoke_parse_errors.csv ({rep})",
        )
    return "failed", "no structures passed Layer A+B; see smoke_parse_errors.csv"


def _write_smoke_artifacts_manifest(table_dir: Path, figure_dir: Path) -> None:
    rows = [
        ("table", "layer_a_structure.csv", "Per-PDB: chain, residue count, mmCIF hash"),
        ("table", "layer_b_topology.csv", "Per-PDB: edges, H0/H1 interval counts, NPZ path"),
        ("table", "smoke_pipeline_status.csv", "Long-form pipeline stages + ColabFold / training regimes"),
        ("table", "smoke_per_structure_metrics.csv", "RMSD, pLDDT, PH sums per PDB"),
        (
            "table",
            "smoke_nine_run_metrics.csv",
            "3×3: level×stage; TM-score (TM-align if PATH), pLDDT mean/min/std",
        ),
        ("figure", "assets/table_smoke_nine_run.png", "Grid: Level | Stage | TM-score | pLDDT"),
        ("figure", "smoke_layer_a_b_counts.png", "Bar chart: residues vs edges"),
        ("figure", "smoke_persistence_by_dim.png", "Stacked bars H0/H1 counts"),
        ("figure", "assets/*_pd_3d.png", "3D persistence diagrams (shared engine with scripts/figures/)"),
        ("figure", "assets/smoke_residue_graph_pca.png", "Residue graph PCA projection"),
        ("figure", "structure_views/*.png", "py3Dmol cartoon + surface (if not skipped)"),
    ]
    df = pd.DataFrame(rows, columns=["kind", "relative_path", "description"])
    table_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(table_dir / "smoke_artifacts_manifest.csv", index=False)


def _write_smoke_end_to_end_metrics(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    colabfold_out: Path,
    table_dir: Path,
    figure_dir: Path,
    *,
    graph_mode: str,
    max_dimension: int,
) -> None:
    """Per-PDB metrics + full 3×3 grid: TM-score + pLDDT (see smoke_nine_experiments)."""
    pred_dir = colabfold_out / "predictions"
    merged: list[dict[str, object]] = []
    for _, a in df_a.iterrows():
        bmatch = df_b[df_b["pdb_id"] == a["pdb_id"]]
        if bmatch.empty:
            continue
        b = bmatch.iloc[0]
        m = compute_per_structure_metrics(
            mmcif_path=Path(str(a["mmcif_path"])),
            chain_id=str(a["chain_id"]),
            npz_path=Path(str(b["npz_path"])),
            pred_dir=pred_dir,
            pdb_id=str(a["pdb_id"]),
        )
        m["n_edges"] = int(b["n_edges"])
        merged.append(m)
    if not merged:
        return
    dfm = pd.DataFrame(merged)
    table_dir.mkdir(parents=True, exist_ok=True)
    dfm.to_csv(table_dir / "smoke_per_structure_metrics.csv", index=False)

    row0 = merged[0]
    first_a = df_a.iloc[0]
    pred_path = str(row0.get("prediction_pdb_path", "") or "").strip()
    nine_raw: list[dict[str, object]]
    header_note: str | None = None
    if not pred_path or not Path(pred_path).is_file():
        nine_raw = build_empty_nine_experiment_rows(reason=str(row0.get("metric_notes") or "no_prediction_pdb_found"))
    else:
        try:
            native_ca, pred_ca, plddt = load_native_pred_plddt_aligned(
                Path(str(first_a["mmcif_path"])),
                str(first_a["chain_id"]),
                Path(pred_path),
            )
            nine_raw = compute_nine_experiment_metrics(
                native_ca=native_ca,
                pred_ca=pred_ca,
                plddt_per_residue=plddt,
                graph_mode=str(graph_mode),
                max_dimension=int(max_dimension),
            )
        except Exception as exc:  # noqa: BLE001
            nine_raw = build_empty_nine_experiment_rows(reason=repr(exc))
            header_note = "Nine-grid computation failed; see notes on row 1."

    nine_csv = nine_rows_to_legacy_csv_dicts(nine_raw, header_note=header_note)
    pd.DataFrame(nine_csv).to_csv(table_dir / "smoke_nine_run_metrics.csv", index=False)
    asset_dir = figure_dir / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    disp = nine_run_metrics_to_display_table(nine_raw)
    render_nine_run_metrics_table_png(disp, asset_dir / "table_smoke_nine_run.png")


def _m1_inference_status_from_colabfold_csv(path: Path | None) -> tuple[str, str]:
    if path is None or not path.is_file():
        return "not_run", "No layer_c_colabfold_smoke.csv; run with --colabfold-smoke and colabfold_batch on PATH."
    df = pd.read_csv(path)
    if df.empty:
        return "not_run", "empty ColabFold smoke CSV"
    if "regime" in df.columns:
        st = df[df["regime"] == "colabfold_inference_m1"]
    else:
        st = df
    if st.empty:
        return "not_run", "no inference rows in ColabFold smoke CSV"
    ok = (st["status"] == "ok").any()
    if ok:
        return "ok", "At least one colabfold_batch job finished; see layer_c_colabfold_smoke.csv and output/colabfold_smoke/predictions."
    dry = (st["status"] == "dry_run").any()
    if dry:
        return "dry_run", "FASTA only; re-run without --colabfold-dry-run where colabfold_batch exists."
    skip = (st["status"] == "colabfold_batch_not_on_path").any()
    if skip:
        return "skipped", "colabfold_batch not on PATH; install LocalColabFold in WSL2 (wsl_install_localcolabfold.sh)."
    return "failed", "See layer_c_colabfold_smoke.csv and colabfold_batch_stdout_stderr.log."


def _write_pipeline_status(
    out_csv: Path,
    *,
    n_corpus: int,
    n_layer_ab_ok: int,
    n_parse_errors: int,
    graph_mode: str,
    probe: dict[str, object],
    colabfold_smoke_csv: Path | None,
) -> None:
    hint_installed = bool(probe.get("colabfold_batch_on_path"))

    ab_status, ab_notes = _ab_status_strings(
        n_corpus=n_corpus,
        n_layer_ab_ok=n_layer_ab_ok,
        n_parse_errors=n_parse_errors,
        graph_mode=str(graph_mode),
    )

    m1_st, m1_note = _m1_inference_status_from_colabfold_csv(colabfold_smoke_csv)

    train_block = "training_only_not_in_smoke"
    train_notes = (
        "Full ColabFold/JAX fine-tuning with competing objectives is not part of inference smoke; "
        "see README (L_fold + L_W vs L_fold + L_H; depth n is a grid axis, not a fourth recipe)."
    )

    rows = [
        {
            "layer": 1,
            "regime": "structural_data",
            "description": "mmCIF on disk, primary chain, representative coordinates",
            "status": ab_status,
            "notes": ab_notes,
        },
        {
            "layer": 2,
            "regime": "topology_preprocessing",
            "description": "residue graph + clique PH + NPZ cache",
            "status": ab_status,
            "notes": ab_notes,
        },
        {
            "layer": 3,
            "regime": "colabfold_inference_m1",
            "description": "ColabFold (AlphaFold2-class) structure prediction via colabfold_batch (inference smoke)",
            "status": m1_st,
            "notes": m1_note,
        },
        {
            "layer": 3,
            "regime": "train_m2_wasserstein_ph",
            "description": "Fine-tune competing recipe B: L_fold + Wasserstein on diagrams (pred vs native at structural depth n; not persistence landscape)",
            "status": train_block,
            "notes": train_notes,
        },
        {
            "layer": 3,
            "regime": "train_m3_rkhs",
            "description": "Fine-tune competing recipe C: L_fold + L_H (RKHS semimetric on sum-pooled internal VPDs; formal chars/heat; RFF encoding in src/topology)",
            "status": train_block,
            "notes": train_notes,
        },
        {
            "layer": 3,
            "regime": "train_depth_n_axis",
            "description": "Structural depth n (recursive D^(n)) swept within recipes B and C — Cartesian with stage, not a separate training recipe",
            "status": "spec_in_readme",
            "notes": "See README Full differential pipeline; smoke does not sweep n.",
        },
    ]
    if not hint_installed and m1_st in ("not_run", "skipped"):
        rows[2]["notes"] += (
            " No colabfold_batch in probe; install LocalColabFold in WSL2 (scripts/protein/wsl_install_localcolabfold.sh)."
        )

    pd.DataFrame(rows).to_csv(out_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run protein smoke: Layers 1–2 (structure + topology cache), matplotlib + optional "
            "py3Dmol figures, and Layer 3 ColabFold inference (colabfold_batch) unless --no-colabfold-smoke. "
            "Training-only rows (Wasserstein / RKHS / depth axis) appear in smoke_pipeline_status.csv only."
        ),
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "protein" / "mmcif",
        help="Directory containing *.cif / *.mmcif",
    )
    parser.add_argument(
        "--max-corpus-structures",
        type=int,
        default=None,
        help="Process at most N mmCIF files (sorted paths). Default: all files in corpus-dir.",
    )
    parser.add_argument("--max-dimension", type=int, default=1)
    parser.add_argument(
        "--graph-mode",
        type=str,
        choices=("cb_topology", "ca_legacy"),
        default="cb_topology",
    )
    parser.add_argument("--force-topology", action="store_true", help="Rebuild NPZ even if present.")
    parser.add_argument(
        "--skip-structure-figures",
        action="store_true",
        help="Skip py3Dmol+Playwright renders (faster; topology tables still run).",
    )
    parser.add_argument(
        "--figure-width",
        type=int,
        default=1200,
        help="Width/height passed to structure renderer when not skipped.",
    )
    parser.add_argument(
        "--colabfold-smoke",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After Layer A+B, run scripts/protein/run_colabfold_predict_smoke.py (ColabFold / AF2-class M1).",
    )
    parser.add_argument(
        "--colabfold-max-structures",
        type=int,
        default=1,
        help="Forward to ColabFold smoke (keep small; inference is heavy).",
    )
    parser.add_argument(
        "--colabfold-dry-run",
        action="store_true",
        help="Forward --dry-run to ColabFold smoke (FASTA only, no colabfold_batch).",
    )
    parser.add_argument(
        "--colabfold-extra-args",
        type=str,
        default="",
        help="Extra CLI tokens for colabfold_batch (quoted string).",
    )
    parser.add_argument(
        "--colabfold-allow-missing-cli",
        action="store_true",
        help="Allow missing colabfold_batch (skipped status rows). Default: hard error on Linux/WSL.",
    )
    parser.add_argument(
        "--colabfold-allow-windows",
        action="store_true",
        help=(
            "Allow invoking ColabFold smoke on win32 (e.g. colabfold_batch on PATH via WSL interop). "
            "Default blocks this because GPU/JAX is normally WSL-only."
        ),
    )
    parser.add_argument(
        "--colabfold-out-dir",
        type=Path,
        default=None,
        help=(
            "ColabFold working directory (FASTA, predictions/, logs). Use an ext4 path in WSL2 "
            "(e.g. ~/cf_smoke) when the repo is on /mnt/c/ to avoid JAX segfaults on 9p."
        ),
    )
    parser.add_argument(
        "--colabfold-sequential",
        action="store_true",
        help="Forward --sequential to run_colabfold_predict_smoke.py (one GPU job per structure).",
    )
    args = parser.parse_args()

    if (
        args.colabfold_smoke
        and not args.colabfold_dry_run
        and sys.platform == "win32"
        and not args.colabfold_allow_windows
    ):
        raise SystemExit(
            "ColabFold smoke: run from WSL2 — `bash scripts/protein/wsl_run_smoke_from_windows_repo.sh` "
            "(venv + `colabfold_batch` on PATH). Alternatives: `--colabfold-dry-run`, `--no-colabfold-smoke`, "
            "or `--colabfold-allow-windows` if colabfold_batch is available on this host."
        )

    corpus_dir = args.corpus_dir.resolve()
    if not corpus_dir.is_dir():
        raise SystemExit(f"corpus-dir does not exist: {corpus_dir}")

    table_dir = PROJECT_ROOT / "results" / "protein" / "tables" / "smoke"
    figure_dir = PROJECT_ROOT / "results" / "protein" / "figures" / "smoke"
    struct_fig_dir = figure_dir / "structure_views"
    output_json = PROJECT_ROOT / "results" / "protein" / "output" / "smoke_colabfold_probe.json"

    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    mmcifs = _corpus_mmcifs(corpus_dir)
    if args.max_corpus_structures is not None:
        cap = max(0, int(args.max_corpus_structures))
        mmcifs = mmcifs[:cap]
    if not mmcifs:
        raise SystemExit(f"no .cif/.mmcif files under {corpus_dir}")

    probe = _probe_colabfold()
    output_json.write_text(json.dumps(probe, indent=2), encoding="utf-8")

    rows_a: list[dict[str, object]] = []
    rows_b: list[dict[str, object]] = []
    rows_err: list[dict[str, object]] = []

    for mmcif_path in mmcifs:
        pdb_id = mmcif_path.stem.upper()
        try:
            chain = infer_primary_chain_id(mmcif_path)
            sha256 = mmcif_content_fingerprint(mmcif_path)
            size_b = mmcif_path.stat().st_size

            if str(args.graph_mode) == "ca_legacy":
                coords = load_ca_coords_from_mmcif(mmcif_path, chain_id=chain)
            else:
                coords = load_cb_primary_residue_coords_from_mmcif(mmcif_path, chain_id=chain)
            n_res = int(coords.shape[0])
        except Exception as exc:  # noqa: BLE001 — smoke runner; capture all parse failures
            rows_err.append({"pdb_id": pdb_id, "stage": "layer_a_structure", "error": repr(exc)})
            continue

        try:
            npz_path, cpu_s = _ensure_topology_npz(
                mmcif_path,
                chain,
                force=bool(args.force_topology),
                max_dimension=int(args.max_dimension),
                graph_mode=str(args.graph_mode),
            )

            loaded = load_topology_npz(npz_path)
            ef = loaded["edges_filtration"]
            n_edges = int(ef.shape[0])
            n_backbone = int(np.sum(ef <= 1e-12))
            n_spatial = int(n_edges - n_backbone)
            pers = loaded["persistence"]
            dims = pers[:, 0].astype(int)
            n_h0 = int(np.sum(dims == 0))
            n_h1 = int(np.sum(dims == 1))
        except Exception as exc:  # noqa: BLE001
            rows_err.append({"pdb_id": pdb_id, "stage": "layer_b_topology", "error": repr(exc)})
            continue

        rows_a.append(
            {
                "pdb_id": pdb_id,
                "mmcif_path": str(mmcif_path).replace("\\", "/"),
                "chain_id": chain,
                "n_residues": n_res,
                "mmcif_sha256": sha256,
                "file_size_bytes": int(size_b),
            }
        )
        rows_b.append(
            {
                "pdb_id": pdb_id,
                "npz_path": str(npz_path).replace("\\", "/"),
                "topology_policy_id": str(np.asarray(loaded["topology_graph_policy_id"].item())),
                "graph_mode": str(np.asarray(loaded["graph_mode"].item())),
                "max_dimension": int(np.asarray(loaded["max_dimension"])),
                "n_edges": n_edges,
                "n_backbone_edges": n_backbone,
                "n_spatial_edges": n_spatial,
                "n_persistence_h0": n_h0,
                "n_persistence_h1": n_h1,
                "cache_build_cpu_s": round(float(cpu_s), 4) if cpu_s > 0.001 else 0.0,
            }
        )

    df_a = pd.DataFrame(rows_a)
    df_b = pd.DataFrame(rows_b)
    df_err = pd.DataFrame(rows_err)
    df_a.to_csv(table_dir / "layer_a_structure.csv", index=False)
    df_b.to_csv(table_dir / "layer_b_topology.csv", index=False)
    if df_err.empty:
        df_err = pd.DataFrame(columns=["pdb_id", "stage", "error"])
    df_err.to_csv(table_dir / "smoke_parse_errors.csv", index=False)

    if df_a.empty or df_b.empty:
        print("WARNING: no complete Layer A+B rows; skipping summary bar charts.", file=sys.stderr)
    else:
        _figures_from_tables(df_a, df_b, figure_dir)

    colabfold_out = (
        args.colabfold_out_dir.resolve()
        if args.colabfold_out_dir is not None
        else (PROJECT_ROOT / "results" / "protein" / "output" / "colabfold_smoke")
    )
    colabfold_csv = colabfold_out / "layer_c_colabfold_smoke.csv"
    if args.colabfold_smoke and not df_a.empty:
        cf_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "protein" / "run_colabfold_predict_smoke.py"),
            "--layer-a-csv",
            str(table_dir / "layer_a_structure.csv"),
            "--out-dir",
            str(colabfold_out),
            "--max-structures",
            str(int(args.colabfold_max_structures)),
        ]
        if args.colabfold_dry_run:
            cf_cmd.append("--dry-run")
        if args.colabfold_extra_args.strip():
            cf_cmd.extend(["--extra-args", args.colabfold_extra_args.strip()])
        if args.colabfold_allow_missing_cli:
            cf_cmd.append("--allow-missing-cli")
        if args.colabfold_sequential:
            cf_cmd.append("--sequential")
        print("running:", " ".join(cf_cmd))
        subprocess.run(cf_cmd, check=False)
    elif args.colabfold_smoke and df_a.empty:
        print("Skipping ColabFold smoke: no Layer A rows.", file=sys.stderr)

    if not df_a.empty and not df_b.empty:
        _write_smoke_end_to_end_metrics(
            df_a,
            df_b,
            colabfold_out,
            table_dir,
            figure_dir,
            graph_mode=str(args.graph_mode),
            max_dimension=int(args.max_dimension),
        )

    cf_status_csv: Path | None = None
    if args.colabfold_smoke:
        cf_status_csv = colabfold_csv if colabfold_csv.is_file() else None
    elif colabfold_csv.is_file():
        # Prior run / synced artifact (e.g. Windows `--no-colabfold-smoke` but WSL left CSV + predictions).
        cf_status_csv = colabfold_csv

    _write_pipeline_status(
        table_dir / "smoke_pipeline_status.csv",
        n_corpus=len(mmcifs),
        n_layer_ab_ok=len(df_a),
        n_parse_errors=len(rows_err),
        graph_mode=str(args.graph_mode),
        probe=probe,
        colabfold_smoke_csv=cf_status_csv,
    )

    if not df_a.empty and not df_b.empty:
        _figures_persistence_3d_assets(df_b, figure_dir)
        _figures_residue_graph_projections(df_b, figure_dir)
        _write_smoke_artifacts_manifest(table_dir, figure_dir)

    if not args.skip_structure_figures and not df_a.empty:
        wh = int(args.figure_width)
        struct_fig_dir.mkdir(parents=True, exist_ok=True)
        for _, row in df_a.iterrows():
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "protein" / "render_structure_cartoon_surface.py"),
                "--mmcif-path",
                str(row["mmcif_path"]),
                "--pdb-label",
                str(row["pdb_id"]),
                "--chain-id",
                str(row["chain_id"]),
                "--out-dir",
                str(struct_fig_dir),
                "--width",
                str(wh),
                "--height",
                str(wh),
            ]
            print("running:", " ".join(cmd))
            subprocess.run(cmd, check=True)

    print(f"wrote_tables={table_dir.resolve()}")
    print(f"wrote_figures={figure_dir.resolve()}")
    print(f"colabfold_probe={output_json.resolve()}")
    print(f"colabfold_smoke_dir={colabfold_out.resolve()}")


if __name__ == "__main__":
    main()

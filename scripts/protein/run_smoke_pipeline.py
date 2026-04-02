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

from src.protein.dataset_policy import (  # noqa: E402
    CONTACT_GRAPH_RADIUS_MAX_A,
    TOPOLOGY_GRAPH_POLICY_ID,
)
from src.protein.mmcif_io import infer_primary_chain_id  # noqa: E402
from src.protein.residue_points import load_cb_primary_residue_coords_from_mmcif  # noqa: E402
from src.protein.topology_cache import (  # noqa: E402
    build_edges_and_persistence,
    default_topology_npz_path,
    load_topology_npz,
    mmcif_content_fingerprint,
    save_topology_npz,
)


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


def _figures_from_tables(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = df_a["pdb_id"].tolist()

    fig, ax = plt.subplots(figsize=(max(6.0, 0.45 * len(labels)), 4.0))
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, df_a["n_residues"], w, label="residues (CB primary)", color="#4c72b0")
    ax.bar(x + w / 2, df_b["n_edges"], w, label="graph edges", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("count")
    ax.set_title("Smoke corpus — Layer A/B scale")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "smoke_layer_a_b_counts.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(6.0, 0.45 * len(labels)), 4.0))
    h0 = df_b["n_persistence_h0"].to_numpy()
    h1 = df_b["n_persistence_h1"].to_numpy()
    ax.bar(x, h0, label="H0 pairs", color="#55a868")
    ax.bar(x, h1, bottom=h0, label="H1 pairs", color="#c44e52")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("persistence intervals")
    ax.set_title("Clique persistence counts (max_dim from run)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "smoke_persistence_by_dim.png", dpi=150)
    plt.close(fig)


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
    probe: dict[str, object],
    colabfold_smoke_csv: Path | None,
) -> None:
    hint_installed = bool(probe.get("colabfold_batch_on_path"))

    if n_layer_ab_ok == n_corpus and n_parse_errors == 0:
        ab_status = "ok"
        ab_notes = f"all {n_corpus} mmCIF(s) passed frozen CB-primary policy"
    elif n_layer_ab_ok > 0:
        ab_status = "partial"
        ab_notes = (
            f"{n_layer_ab_ok}/{n_corpus} mmCIF(s) passed; "
            f"{n_parse_errors} parse/CB-policy skip(s) — see smoke_parse_errors.csv"
        )
    else:
        ab_status = "failed"
        ab_notes = "no structures passed Layer A+B; see smoke_parse_errors.csv"

    m1_st, m1_note = _m1_inference_status_from_colabfold_csv(colabfold_smoke_csv)

    train_block = "training_only_not_in_smoke"
    train_notes = (
        "Full fine-tuning with auxiliary losses is not part of inference smoke; "
        "implement after ColabFold/AF2 baseline runs (compare native vs predicted coordinates for L_PH / L_RKHS / jet)."
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
            "regime": "train_m2_landscape_ph",
            "description": "Fine-tune: structure loss + topology/landscape loss on persistence features (native vs predicted geometry)",
            "status": train_block,
            "notes": train_notes,
        },
        {
            "layer": 3,
            "regime": "train_m3_rkhs",
            "description": "Fine-tune: add RKHS / semimetric on virtual persistence (epi RKHS port)",
            "status": train_block,
            "notes": train_notes,
        },
        {
            "layer": 3,
            "regime": "train_m4_jet_epilogue",
            "description": "Fine-tune: jet / higher-order multi-scale objective (from epidemiology topological calculus; port TBD)",
            "status": "planned",
            "notes": "Spec only in-repo; hook after M2/M3 when jet-based regularizer is frozen.",
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
            "Training-only regimes M2–M4 appear in smoke_pipeline_status.csv only."
        ),
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "protein" / "mmcif",
        help="Directory containing *.cif / *.mmcif",
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
    args = parser.parse_args()

    if args.colabfold_smoke and not args.colabfold_dry_run and sys.platform == "win32":
        win_path = str(PROJECT_ROOT.resolve())
        if len(win_path) >= 3 and win_path[1] == ":":
            wsl_hint = f"/mnt/{win_path[0].lower()}{win_path[2:].replace(chr(92), '/')}"
        else:
            wsl_hint = win_path.replace("\\", "/")
        raise SystemExit(
            "ColabFold GPU smoke refuses to run under native Windows Python.\n"
            "Use WSL2 + GPU + LocalColabFold (see scripts/protein/wsl_install_localcolabfold.sh), e.g.:\n"
            f"  wsl -e bash -lc 'cd {wsl_hint} && source .venv-wsl-gpu/bin/activate && "
            f"python -u scripts/protein/run_smoke_pipeline.py --skip-structure-figures'\n"
            "From Windows you may still use --colabfold-dry-run (FASTA only, no GPU check)."
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
    if not df_err.empty:
        df_err.to_csv(table_dir / "smoke_parse_errors.csv", index=False)

    if df_a.empty or df_b.empty:
        print("WARNING: no complete Layer A+B rows; skipping summary bar charts.", file=sys.stderr)
    else:
        _figures_from_tables(df_a, df_b, figure_dir)

    colabfold_out = PROJECT_ROOT / "results" / "protein" / "output" / "colabfold_smoke"
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
        print("running:", " ".join(cf_cmd))
        subprocess.run(cf_cmd, check=False)
    elif args.colabfold_smoke and df_a.empty:
        print("Skipping ColabFold smoke: no Layer A rows.", file=sys.stderr)

    cf_status_csv: Path | None = None
    if args.colabfold_smoke:
        cf_status_csv = colabfold_csv if colabfold_csv.is_file() else None

    _write_pipeline_status(
        table_dir / "smoke_pipeline_status.csv",
        n_corpus=len(mmcifs),
        n_layer_ab_ok=len(df_a),
        n_parse_errors=len(df_err),
        probe=probe,
        colabfold_smoke_csv=cf_status_csv,
    )

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

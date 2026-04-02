from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.protein.cuda_guard import ensure_nvidia_gpu_present  # noqa: E402
from src.protein.mmcif_io import extract_protein_sequence_chain  # noqa: E402


def _colabfold_batch_executable() -> str | None:
    root = os.environ.get("LOCALCOLABFOLD_ROOT", "").strip()
    if root:
        p = Path(root).expanduser() / ".pixi" / "envs" / "default" / "bin" / "colabfold_batch"
        if p.is_file():
            return str(p)
    return shutil.which("colabfold_batch")


def _attach_gpu_columns(row: dict[str, object], probe: dict[str, object] | None) -> dict[str, object]:
    if probe is None:
        row["gpu_validation"] = ""
        row["nvidia_gpu_names"] = ""
        return row
    row["gpu_validation"] = "ok"
    row["nvidia_gpu_names"] = ";".join(probe.get("nvidia_smi_gpus", []))
    return row


def _wsl_jax_env() -> dict[str, str]:
    """Recommended exports for LocalColabFold on WSL2 (see localcolabfold README)."""
    return {
        "TF_FORCE_UNIFIED_MEMORY": "1",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "4.0",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
        "TF_FORCE_GPU_ALLOW_GROWTH": "true",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Layer C smoke: FASTA from Layer A rows + colabfold_batch (ColabFold / AlphaFold2-class). "
            "Requires NVIDIA GPU (nvidia-smi); weights download on first colabfold_batch run. "
            "Install LocalColabFold in WSL2 (see scripts/protein/wsl_install_localcolabfold.sh)."
        ),
    )
    parser.add_argument(
        "--layer-a-csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "protein" / "tables" / "smoke" / "layer_a_structure.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "protein" / "output" / "colabfold_smoke",
    )
    parser.add_argument("--max-structures", type=int, default=1)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only write FASTA + status CSV (skip GPU check and colabfold_batch).",
    )
    parser.add_argument(
        "--extra-args",
        type=str,
        default="",
        help=(
            "Extra tokens for colabfold_batch (single string). For low VRAM try e.g. "
            "'--num-recycle 1 --max-msa 128:256' — see colabfold_batch --help and ColabFold/LocalColabFold docs."
        ),
    )
    parser.add_argument(
        "--allow-missing-cli",
        action="store_true",
        help="If colabfold_batch is absent, write skipped rows instead of exiting.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_a = args.layer_a_csv.resolve()
    if not layer_a.is_file():
        raise SystemExit(f"missing Layer A table: {layer_a}")

    df = pd.read_csv(layer_a)
    if df.empty:
        raise SystemExit("Layer A table is empty.")

    failures: list[dict[str, object]] = []
    fasta_lines: list[str] = []
    pdb_order: list[str] = []
    for _, r in df.iterrows():
        if len(pdb_order) >= int(args.max_structures):
            break
        pdb = str(r["pdb_id"])
        mmcif = Path(str(r["mmcif_path"]))
        chain = str(r["chain_id"])
        try:
            seq = extract_protein_sequence_chain(mmcif, chain)
        except Exception as exc:  # noqa: BLE001
            failures.append(
                _attach_gpu_columns(
                    {
                        "pdb_id": pdb,
                        "regime": "colabfold_inference_m1",
                        "status": "failed_sequence",
                        "detail": repr(exc),
                        "fasta_path": "",
                        "colabfold_batch_rc": "",
                    },
                    None,
                )
            )
            continue
        fasta_lines.append(f">{pdb}\n{seq}")
        pdb_order.append(pdb)

    status_csv = out_dir / "layer_c_colabfold_smoke.csv"
    if not pdb_order:
        pd.DataFrame(failures).to_csv(status_csv, index=False)
        print(f"wrote_status_only={status_csv}")
        return

    fasta_path = out_dir / "smoke_colabfold.fasta"
    fasta_path.write_text("\n".join(fasta_lines) + "\n", encoding="utf-8")

    runner = _colabfold_batch_executable()
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        rows = list(failures)
        for pdb in pdb_order:
            rows.append(
                _attach_gpu_columns(
                    {
                        "pdb_id": pdb,
                        "regime": "colabfold_inference_m1",
                        "status": "dry_run",
                        "detail": "FASTA only; colabfold_batch not run.",
                        "fasta_path": str(fasta_path).replace("\\", "/"),
                        "colabfold_batch_rc": "",
                    },
                    None,
                )
            )
        pd.DataFrame(rows).to_csv(status_csv, index=False)
        print(f"wrote={fasta_path}")
        print(f"wrote={status_csv}")
        return

    if runner is None:
        if args.allow_missing_cli:
            rows = list(failures)
            for pdb in pdb_order:
                rows.append(
                    _attach_gpu_columns(
                        {
                            "pdb_id": pdb,
                            "regime": "colabfold_inference_m1",
                            "status": "colabfold_batch_not_on_path",
                            "detail": "Set LOCALCOLABFOLD_ROOT or install LocalColabFold; see wsl_install_localcolabfold.sh",
                            "fasta_path": str(fasta_path).replace("\\", "/"),
                            "colabfold_batch_rc": "",
                        },
                        None,
                    )
                )
            pd.DataFrame(rows).to_csv(status_csv, index=False)
            print(f"wrote={fasta_path}")
            print(f"wrote={status_csv}")
            return
        raise SystemExit(
            "colabfold_batch not found. In WSL2: bash scripts/protein/wsl_install_localcolabfold.sh "
            "or export LOCALCOLABFOLD_ROOT=$HOME/localcolabfold"
        )

    probe = ensure_nvidia_gpu_present(purpose="ColabFold smoke")
    (out_dir / "smoke_cuda_validation.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")

    cmd = [runner, str(fasta_path), str(pred_dir)]
    if args.extra_args.strip():
        cmd.extend(args.extra_args.split())

    env = os.environ.copy()
    env.update(_wsl_jax_env())
    log_path = out_dir / "colabfold_batch_stdout_stderr.log"
    print("running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(out_dir), env=env, capture_output=True, text=True)
    log_path.write_text(proc.stdout + "\n---\n" + proc.stderr, encoding="utf-8")

    rc = int(proc.returncode)
    st = "ok" if rc == 0 else "colabfold_batch_failed"
    rows = list(failures)
    for pdb in pdb_order:
        rows.append(
            _attach_gpu_columns(
                {
                    "pdb_id": pdb,
                    "regime": "colabfold_inference_m1",
                    "status": st,
                    "detail": str(log_path).replace("\\", "/") if rc != 0 else str(pred_dir).replace("\\", "/"),
                    "fasta_path": str(fasta_path).replace("\\", "/"),
                    "colabfold_batch_rc": rc,
                },
                probe,
            )
        )
    pd.DataFrame(rows).to_csv(status_csv, index=False)
    print(f"wrote={status_csv}")
    print(f"wrote={out_dir / 'smoke_cuda_validation.json'}")
    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()

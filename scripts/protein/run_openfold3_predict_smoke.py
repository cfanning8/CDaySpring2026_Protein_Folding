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

from src.protein.cuda_guard import ensure_cuda_for_openfold  # noqa: E402
from src.protein.mmcif_io import extract_protein_sequence_chain  # noqa: E402
from src.protein.openfold3_query import merge_monomer_queries, write_query_json  # noqa: E402


def _which_openfold() -> str | None:
    return shutil.which("run_openfold")


def _attach_gpu_columns(row: dict[str, object], probe: dict[str, object] | None) -> dict[str, object]:
    if probe is None:
        row["gpu_validation"] = ""
        row["nvidia_gpu_names"] = ""
        row["torch_cuda_version"] = ""
        row["torch_device_name"] = ""
        return row
    row["gpu_validation"] = "ok"
    row["nvidia_gpu_names"] = ";".join(probe.get("nvidia_smi_gpus", []))
    row["torch_cuda_version"] = str(probe.get("cuda_version_torch") or "")
    row["torch_device_name"] = str(probe.get("device_name") or "")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Layer C smoke: OpenFold3 query JSON from mmCIF rows + run_openfold predict. "
            "Requires NVIDIA GPU (nvidia-smi + torch CUDA); no CPU fallback."
        ),
    )
    parser.add_argument(
        "--layer-a-csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "protein" / "tables" / "smoke" / "layer_a_structure.csv",
        help="Rows from run_smoke_pipeline (must include pdb_id, mmcif_path, chain_id).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "protein" / "output" / "openfold3_smoke",
    )
    parser.add_argument(
        "--max-structures",
        type=int,
        default=1,
        help="Cap OpenFold3 jobs (inference is heavy); default 1 for smoke.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only write query JSON + status CSV (skips GPU enforcement and run_openfold).",
    )
    parser.add_argument(
        "--extra-args",
        type=str,
        default="",
        help="Extra args passed to run_openfold predict (single string, e.g. '--low_memory').",
    )
    parser.add_argument(
        "--allow-missing-run-openfold",
        action="store_true",
        help="If set, write skipped status when run_openfold is absent (default: exit with error).",
    )
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_a = args.layer_a_csv.resolve()
    if not layer_a.is_file():
        raise SystemExit(f"missing Layer A table: {layer_a} (run run_smoke_pipeline.py first)")

    df = pd.read_csv(layer_a)
    if df.empty:
        raise SystemExit("Layer A table is empty; nothing to predict.")

    failures: list[dict[str, object]] = []
    trio: list[tuple[str, str, str]] = []
    for _, r in df.iterrows():
        if len(trio) >= int(args.max_structures):
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
                        "regime": "openfold3_inference_m1",
                        "status": "failed_sequence",
                        "detail": repr(exc),
                        "query_json": "",
                        "run_openfold_rc": "",
                    },
                    None,
                )
            )
            continue
        trio.append((pdb, chain, seq))

    status_rows: list[dict[str, object]] = list(failures)
    query_path = out_dir / "smoke_queries.json"
    runner = _which_openfold()

    if not trio:
        pd.DataFrame(status_rows).to_csv(out_dir / "layer_c_openfold3_smoke.csv", index=False)
        print(f"wrote_status_only={out_dir / 'layer_c_openfold3_smoke.csv'}")
        return

    payload = merge_monomer_queries(trio)
    write_query_json(query_path, payload)

    if args.dry_run:
        for pdb, chain, _seq in trio:
            status_rows.append(
                _attach_gpu_columns(
                    {
                        "pdb_id": pdb,
                        "regime": "openfold3_inference_m1",
                        "status": "dry_run",
                        "detail": "GPU not validated in --dry-run.",
                        "query_json": str(query_path).replace("\\", "/"),
                        "run_openfold_rc": "",
                    },
                    None,
                )
            )
        pd.DataFrame(status_rows).to_csv(out_dir / "layer_c_openfold3_smoke.csv", index=False)
        print(f"wrote={query_path}")
        print(f"wrote={out_dir / 'layer_c_openfold3_smoke.csv'}")
        return

    if runner is None:
        if args.allow_missing_run_openfold:
            for pdb, _, _ in trio:
                status_rows.append(
                    _attach_gpu_columns(
                        {
                            "pdb_id": pdb,
                            "regime": "openfold3_inference_m1",
                            "status": "run_openfold_not_on_path",
                            "detail": "Install openfold3, run setup_openfold, ensure run_openfold is on PATH.",
                            "query_json": str(query_path).replace("\\", "/"),
                            "run_openfold_rc": "",
                        },
                        None,
                    )
                )
            pd.DataFrame(status_rows).to_csv(out_dir / "layer_c_openfold3_smoke.csv", index=False)
            print(f"wrote={query_path}")
            print(f"wrote={out_dir / 'layer_c_openfold3_smoke.csv'}")
            return
        raise SystemExit(
            "run_openfold not on PATH. Inside WSL: source .venv-wsl-gpu/bin/activate "
            "after bash scripts/protein/wsl_setup_openfold3_gpu.sh && setup_openfold"
        )

    probe = ensure_cuda_for_openfold(purpose="OpenFold3 predict smoke")
    (out_dir / "smoke_cuda_validation.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")

    pred_out = out_dir / "predictions"
    pred_out.mkdir(parents=True, exist_ok=True)
    cmd = [runner, "predict", f"--query_json={query_path}", f"--output_dir={pred_out}"]
    if args.extra_args.strip():
        cmd.extend(args.extra_args.split())
    env = os.environ.copy()
    print("running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(out_dir), env=env, capture_output=True, text=True)
    log_path = out_dir / "run_openfold_stdout_stderr.log"
    log_path.write_text(proc.stdout + "\n---\n" + proc.stderr, encoding="utf-8")

    rc = int(proc.returncode)
    st = "ok" if rc == 0 else "run_openfold_failed"
    for pdb, _, _ in trio:
        status_rows.append(
            _attach_gpu_columns(
                {
                    "pdb_id": pdb,
                    "regime": "openfold3_inference_m1",
                    "status": st,
                    "detail": str(log_path).replace("\\", "/") if rc != 0 else str(pred_out).replace("\\", "/"),
                    "query_json": str(query_path).replace("\\", "/"),
                    "run_openfold_rc": rc,
                },
                probe,
            )
        )
    pd.DataFrame(status_rows).to_csv(out_dir / "layer_c_openfold3_smoke.csv", index=False)
    print(f"wrote={out_dir / 'layer_c_openfold3_smoke.csv'}")
    print(f"wrote={out_dir / 'smoke_cuda_validation.json'}")
    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()

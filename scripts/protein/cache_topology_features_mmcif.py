from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.protein.dataset_policy import (  # noqa: E402
    CONTACT_GRAPH_RADIUS_MAX_A,
    TOPOLOGY_GRAPH_POLICY_ID,
)
from src.protein.mmcif_io import load_ca_coords_from_mmcif  # noqa: E402
from src.protein.residue_points import load_cb_primary_residue_coords_from_mmcif  # noqa: E402
from src.protein.topology_cache import (  # noqa: E402
    build_edges_and_persistence,
    default_topology_npz_path,
    mmcif_content_fingerprint,
    save_topology_npz,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache residue graph + clique persistence features for one mmCIF (NPZ under data/processed/protein/).",
    )
    parser.add_argument("--mmcif-path", type=Path, required=True)
    parser.add_argument("--chain-id", type=str, default=None)
    parser.add_argument("--out-npz", type=Path, default=None)
    parser.add_argument("--radius-max-a", type=float, default=CONTACT_GRAPH_RADIUS_MAX_A)
    parser.add_argument("--max-dimension", type=int, default=1)
    parser.add_argument(
        "--graph-mode",
        type=str,
        choices=("cb_topology", "ca_legacy"),
        default="cb_topology",
        help="cb_topology: C_beta primary + backbone filtration 0; ca_legacy: all edges use distance.",
    )
    parser.add_argument("--policy-id", type=str, default=TOPOLOGY_GRAPH_POLICY_ID)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    chain = args.chain_id
    if chain is not None and len(chain) != 1:
        raise SystemExit("--chain-id must be one character")

    mmcif_path = args.mmcif_path.resolve()
    if args.graph_mode == "cb_topology":
        coords = load_cb_primary_residue_coords_from_mmcif(mmcif_path, chain_id=chain)
        representative = "cb_primary"
    else:
        coords = load_ca_coords_from_mmcif(mmcif_path, chain_id=chain)
        representative = "ca_legacy"

    out_path = args.out_npz
    if out_path is None:
        out_path = default_topology_npz_path(
            mmcif_path,
            chain,
            policy_id=str(args.policy_id),
            representative=representative,
            radius_max_a=float(args.radius_max_a),
            project_root=PROJECT_ROOT,
        )

    if out_path.is_file() and not args.force:
        print(f"skip_existing={out_path.resolve()}")
        return

    sha = mmcif_content_fingerprint(mmcif_path)
    es, et, ef, ptab, _ = build_edges_and_persistence(
        coords,
        float(args.radius_max_a),
        graph_mode=str(args.graph_mode),
        max_dimension=int(args.max_dimension),
    )

    save_topology_npz(
        out_path,
        topology_graph_policy_id=str(args.policy_id),
        mmcif_path=mmcif_path,
        mmcif_sha256=sha,
        chain_id=chain,
        representative=representative,
        radius_max_a=float(args.radius_max_a),
        graph_mode=str(args.graph_mode),
        max_dimension=int(args.max_dimension),
        coords=coords,
        edges_source=es,
        edges_target=et,
        edges_filtration=ef,
        persistence_table=ptab,
    )

    print(f"wrote={out_path.resolve()} n_res={coords.shape[0]} n_edges={es.shape[0]}")


if __name__ == "__main__":
    main()

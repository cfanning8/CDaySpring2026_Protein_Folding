from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.protein.clique_persistence import clique_persistence_from_distance_edges  # noqa: E402
from src.protein.dataset_policy import CONTACT_GRAPH_RADIUS_MAX_A  # noqa: E402
from src.protein.mmcif_io import load_ca_coords_from_mmcif  # noqa: E402
from src.protein.residue_points import load_cb_primary_residue_coords_from_mmcif  # noqa: E402
from src.protein.residue_graph import residue_contact_edges, topology_residue_graph_edges  # noqa: E402
from src.topology.persistence import persistence_pairs_for_dimension  # noqa: E402


def _finite_interval_count(pairs: list[tuple[int, tuple[float, float]]]) -> int:
    count = 0
    for _, interval in pairs:
        birth, death = float(interval[0]), float(interval[1])
        if death == float("inf"):
            continue
        if death > birth:
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a residue contact graph, clique-expand, and summarize persistent homology.",
    )
    parser.add_argument("--mmcif-path", type=Path, required=True)
    parser.add_argument("--chain-id", type=str, default=None)
    parser.add_argument("--radius-max-a", type=float, default=CONTACT_GRAPH_RADIUS_MAX_A)
    parser.add_argument("--max-dimension", type=int, default=1)
    parser.add_argument(
        "--graph-mode",
        type=str,
        choices=("cb_topology", "ca_legacy"),
        default="cb_topology",
    )
    args = parser.parse_args()

    chain = args.chain_id
    if chain is not None and len(chain) != 1:
        raise SystemExit("--chain-id must be a single character")

    if args.graph_mode == "cb_topology":
        coords = load_cb_primary_residue_coords_from_mmcif(args.mmcif_path, chain_id=chain)
        edges = topology_residue_graph_edges(coords, float(args.radius_max_a), backbone_filtration=0.0)
    else:
        coords = load_ca_coords_from_mmcif(args.mmcif_path, chain_id=chain)
        edges = residue_contact_edges(coords, float(args.radius_max_a))
    pairs = clique_persistence_from_distance_edges(edges, max_dimension=int(args.max_dimension))

    h0 = persistence_pairs_for_dimension(pairs, 0)
    h1 = persistence_pairs_for_dimension(pairs, 1)

    print(f"path={args.mmcif_path.resolve()} graph_mode={args.graph_mode}")
    print(f"n_residue={coords.shape[0]} n_edges={len(edges)}")
    print(f"H0_features={len(h0)} H0_finite={_finite_interval_count(h0)}")
    print(f"H1_features={len(h1)} H1_finite={_finite_interval_count(h1)}")


if __name__ == "__main__":
    main()

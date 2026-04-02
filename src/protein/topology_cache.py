from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from src.protein.clique_persistence import clique_persistence_from_distance_edges
from src.protein.dataset_policy import TOPOLOGY_GRAPH_POLICY_ID
from src.protein.residue_graph import residue_contact_edges, topology_residue_graph_edges


def processed_protein_dir(project_root: Path | None = None) -> Path:
    root = project_root if project_root is not None else Path(__file__).resolve().parents[2]
    return root / "data" / "processed" / "protein"


def default_topology_npz_path(
    mmcif_path: Path,
    chain_id: str | None,
    *,
    policy_id: str = TOPOLOGY_GRAPH_POLICY_ID,
    representative: str,
    radius_max_a: float,
    project_root: Path | None = None,
) -> Path:
    stem = mmcif_path.resolve().stem.upper()
    ch = chain_id if chain_id is not None else "_"
    tag = hashlib.sha256(
        f"{policy_id}|{representative}|{radius_max_a:.6f}|{stem}|{ch}".encode("ascii"),
        usedforsecurity=False,
    ).hexdigest()[:12]
    return processed_protein_dir(project_root) / f"{stem}_{ch}_{tag}.npz"


def mmcif_content_fingerprint(mmcif_path: Path) -> str:
    data = Path(mmcif_path).read_bytes()
    return hashlib.sha256(data, usedforsecurity=False).hexdigest()


def build_edges_and_persistence(
    coords: np.ndarray,
    radius_max_a: float,
    *,
    graph_mode: str,
    max_dimension: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[int, tuple[float, float]]]]:
    if graph_mode == "cb_topology":
        edges_df = topology_residue_graph_edges(coords, float(radius_max_a), backbone_filtration=0.0)
    elif graph_mode == "ca_legacy":
        edges_df = residue_contact_edges(coords, float(radius_max_a))
    else:
        raise ValueError("graph_mode must be cb_topology or ca_legacy")

    pairs = clique_persistence_from_distance_edges(edges_df, max_dimension=int(max_dimension))
    src = np.asarray(edges_df["source"].to_numpy(dtype=np.int32), dtype=np.int32)
    tgt = np.asarray(edges_df["target"].to_numpy(dtype=np.int32), dtype=np.int32)
    if "filtration" in edges_df.columns:
        filt = np.asarray(edges_df["filtration"].to_numpy(dtype=np.float64), dtype=np.float64)
    else:
        filt = np.asarray(edges_df["distance"].to_numpy(dtype=np.float64), dtype=np.float64)

    dims = np.array([int(p[0]) for p in pairs], dtype=np.int32)
    births = np.array([float(p[1][0]) for p in pairs], dtype=np.float64)
    deaths = np.array([float(p[1][1]) for p in pairs], dtype=np.float64)
    return src, tgt, filt, np.stack([dims, births, deaths], axis=1), pairs


def save_topology_npz(
    out_path: Path,
    *,
    topology_graph_policy_id: str,
    mmcif_path: Path,
    mmcif_sha256: str,
    chain_id: str | None,
    representative: str,
    radius_max_a: float,
    graph_mode: str,
    max_dimension: int,
    coords: np.ndarray,
    edges_source: np.ndarray,
    edges_target: np.ndarray,
    edges_filtration: np.ndarray,
    persistence_table: np.ndarray,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        topology_graph_policy_id=np.array(topology_graph_policy_id),
        mmcif_path=np.array(str(mmcif_path.resolve())),
        mmcif_sha256=np.array(mmcif_sha256),
        chain_id=np.array("" if chain_id is None else chain_id),
        representative=np.array(representative),
        radius_max_a=np.float64(radius_max_a),
        graph_mode=np.array(graph_mode),
        max_dimension=np.int32(max_dimension),
        coords=np.asarray(coords, dtype=np.float64),
        edges_source=np.asarray(edges_source, dtype=np.int32),
        edges_target=np.asarray(edges_target, dtype=np.int32),
        edges_filtration=np.asarray(edges_filtration, dtype=np.float64),
        persistence=np.asarray(persistence_table, dtype=np.float64),
    )


def load_topology_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}

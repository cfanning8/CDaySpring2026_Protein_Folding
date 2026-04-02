from __future__ import annotations

from pathlib import Path

import numpy as np


def _virtual_cb_from_n_ca_c(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    """ProteinMPNN-style virtual C_beta from backbone N, CA, C (Angstrom coordinates)."""
    b = ca - n
    cvec = c - ca
    bn = float(np.linalg.norm(b))
    cn = float(np.linalg.norm(cvec))
    if bn < 1e-6 or cn < 1e-6:
        raise ValueError("degenerate backbone segment for virtual C_beta")
    b = b / bn
    cvec = cvec / cn
    a = np.cross(b, cvec)
    an = float(np.linalg.norm(a))
    if an < 1e-8:
        raise ValueError("collinear N, CA, C for virtual C_beta")
    a = a / an
    return (
        ca
        - 0.58273431 * b
        + 0.56802827 * cvec
        - 0.54067466 * a
    )


def _atom_coord_for_name(coords, atom_names, target: str) -> np.ndarray | None:
    mask = atom_names == target
    if not np.any(mask):
        return None
    idx = int(np.flatnonzero(mask)[0])
    return np.asarray(coords[idx], dtype=np.float64)


def load_cb_primary_residue_coords_from_mmcif(
    mmcif_path: Path,
    chain_id: str | None,
) -> np.ndarray:
    import biotite.structure as struc
    import biotite.structure.io.pdbx as pdbx

    path = Path(mmcif_path)
    if not path.is_file():
        raise FileNotFoundError(str(path))

    cif_file = pdbx.CIFFile.read(str(path))
    structure = pdbx.get_structure(cif_file, model=1)
    if chain_id is not None:
        if len(chain_id) != 1:
            raise ValueError("chain_id must be one character when set")
        structure = structure[structure.chain_id.astype(str) == chain_id]

    if structure.array_length() == 0:
        raise ValueError("empty structure after chain filter")

    coords_out: list[np.ndarray] = []
    for res in struc.residue_iter(structure):
        res_name = str(res.res_name[0]).strip().upper()
        res_coords = res.coord
        names = np.char.strip(res.atom_name.astype(str))

        ca = _atom_coord_for_name(res_coords, names, "CA")
        if ca is None:
            raise ValueError("residue missing CA")

        if res_name == "GLY":
            coords_out.append(ca)
            continue

        cb = _atom_coord_for_name(res_coords, names, "CB")
        if cb is not None:
            coords_out.append(cb)
            continue

        n_coord = _atom_coord_for_name(res_coords, names, "N")
        c_coord = _atom_coord_for_name(res_coords, names, "C")
        if n_coord is None or c_coord is None:
            raise ValueError("missing N or C needed for virtual C_beta")
        coords_out.append(_virtual_cb_from_n_ca_c(n_coord, ca, c_coord))

    if not coords_out:
        raise ValueError("no residues extracted")

    stacked = np.stack(coords_out, axis=0)
    if stacked.ndim != 2 or stacked.shape[1] != 3:
        raise ValueError("unexpected coordinate shape")
    return stacked.astype(np.float64, copy=False)

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_ca_coords_from_mmcif(mmcif_path: Path, chain_id: str | None = None) -> np.ndarray:
    import biotite.structure.io.pdbx as pdbx

    path = Path(mmcif_path)
    if not path.is_file():
        raise FileNotFoundError(str(path))

    if chain_id is not None and len(chain_id) != 1:
        raise ValueError("chain_id must be a single character when provided")

    cif_file = pdbx.CIFFile.read(str(path))
    structure = pdbx.get_structure(cif_file, model=1)
    mask = structure.atom_name == "CA"
    if chain_id is not None:
        chain_ids = structure.chain_id.astype(str)
        mask = mask & (chain_ids == chain_id)

    ca = structure[mask]
    if len(ca) == 0:
        raise ValueError("no CA atoms found after parsing with the given filters")

    coords = np.asarray(ca.coord, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("unexpected CA coordinate shape")

    return coords


def infer_primary_chain_id(mmcif_path: Path) -> str:
    """Pick a single chain id from CA atoms: sole chain, else 'A' if present, else lexicographically first."""
    import biotite.structure.io.pdbx as pdbx

    path = Path(mmcif_path)
    if not path.is_file():
        raise FileNotFoundError(str(path))

    cif_file = pdbx.CIFFile.read(str(path))
    structure = pdbx.get_structure(cif_file, model=1)
    ca = structure[structure.atom_name == "CA"]
    if len(ca) == 0:
        raise ValueError("no CA atoms")
    chains = sorted({str(c) for c in ca.chain_id.astype(str)})
    if len(chains) == 1:
        return chains[0]
    if "A" in chains:
        return "A"
    return chains[0]


def extract_protein_sequence_chain(mmcif_path: Path, chain_id: str) -> str:
    """One-letter protein sequence for a single chain (mmCIF model 1), via biotite.

    Drops hetero/ligand/solvent atoms **before** ``to_sequence`` so mmCIF rows that mix
    MSE/NAG/waters into the same chain id do not trigger ``BadStructureError``.
    """
    import biotite.structure as struc
    import biotite.structure.io.pdbx as pdbx
    from biotite.sequence import ProteinSequence

    if len(chain_id) != 1:
        raise ValueError("chain_id must be one character")
    path = Path(mmcif_path)
    if not path.is_file():
        raise FileNotFoundError(str(path))

    cif_file = pdbx.CIFFile.read(str(path))
    structure = pdbx.get_structure(cif_file, model=1)
    structure = structure[structure.chain_id.astype(str) == chain_id]
    if structure.array_length() == 0:
        raise ValueError("empty structure after chain filter")

    aa_mask = struc.filter_amino_acids(structure)
    structure = structure[aa_mask]
    if structure.array_length() == 0:
        raise ValueError("no standard amino acid atoms in chain after hetero filter")

    block, _chain_ids = struc.to_sequence(structure)
    parts: list[str] = []
    if isinstance(block, list):
        for item in block:
            if isinstance(item, ProteinSequence):
                parts.append(str(item))
    elif isinstance(block, ProteinSequence):
        parts.append(str(block))
    if not parts:
        raise ValueError("no protein sequence extracted after amino-acid filtering")
    return "".join(parts)

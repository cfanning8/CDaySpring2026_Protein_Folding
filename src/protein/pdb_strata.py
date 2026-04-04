from __future__ import annotations

from src.protein import dataset_policy as pol

# Core (low-noise) X-ray band for the primary supervised stratum.
CORE_XRAY_RESOLUTION_MAX = 2.5


def resolution_band_value(resolution: float | None) -> str:
    if resolution is None or not (resolution == resolution):
        return "unknown"
    r = float(resolution)
    if r < 2.5:
        return "lt_2p5"
    if r <= 3.5:
        return "2p5_to_3p5"
    return "gt_3p5"


def modality_label(experimental_method: str) -> str:
    method = experimental_method.strip().upper()
    if method == pol.XRAY_METHOD.upper():
        return "xray"
    if method == pol.CRYO_EM_METHOD.upper():
        return "cryoem"
    return "other"


def ligand_state_label(has_nonpolymer_components: bool) -> str:
    return "ligand_or_cofactor_bound" if has_nonpolymer_components else "apo_like"


def assign_pdb_stratum(
    *,
    modality: str,
    resolution: float | None,
    is_membrane_annotated: bool,
    ligand_state: str,
    deposited_protein_instance_count: int | None,
    chain_length_metadata: int | None,
) -> str:
    inst = deposited_protein_instance_count
    length = chain_length_metadata
    res = float(resolution) if resolution is not None and (resolution == resolution) else None

    stress_multimer = inst is not None and inst != 1
    stress_membrane = is_membrane_annotated
    stress_ligand = ligand_state != "apo_like"
    stress_long = length is not None and length > 600
    stress_xray_moderate = modality == "xray" and res is not None and res > CORE_XRAY_RESOLUTION_MAX

    if stress_multimer or stress_membrane or stress_ligand or stress_long or stress_xray_moderate:
        return "stress"

    if modality == "cryoem":
        return "cryo_em_robustness"

    if modality == "xray" and res is not None and res <= CORE_XRAY_RESOLUTION_MAX:
        return "core_xray"

    if modality == "xray":
        return "stress"

    return "unassigned"

from __future__ import annotations

import json
from typing import Any

import requests

from src.protein import dataset_policy as pol
from src.protein.pdb_manifest_schema import FROZEN_PDB_MANIFEST_VERSION, PDB_MANIFEST_COLUMNS
from src.protein.pdb_strata import (
    assign_pdb_stratum,
    ligand_state_label,
    modality_label,
    resolution_band_value,
)
from src.protein.validation_urls import mmcif_deposited_download_url, validation_report_pdf_candidates

CORE_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
POLYMER_ENTITY_URL = "https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}"


def _first_resolution(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            return None
        value = value[0]
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not (out == out) or out <= 0:
        return None
    return out


def _exptl_method(core: dict) -> str:
    exptl = core.get("exptl")
    if isinstance(exptl, list) and exptl:
        method = exptl[0].get("method")
        if method:
            return str(method)
    return "unknown"


def _database_status(core: dict) -> dict:
    status = core.get("pdbx_database_status")
    if isinstance(status, list) and status:
        first = status[0]
        if isinstance(first, dict):
            return first
    if isinstance(status, dict):
        return status
    return {}


def _keywords_text(core: dict) -> str:
    sk = core.get("struct_keywords") or {}
    parts = []
    for key in ("pdbx_keywords", "text"):
        val = sk.get(key)
        if val:
            parts.append(str(val))
    return " ".join(parts).upper()


def _vrpt_payload(core: dict) -> dict:
    payload: dict[str, Any] = {}
    summary = core.get("pdbx_vrpt_summary")
    if isinstance(summary, dict):
        for key in ("report_creation_date", "attempted_validation_steps", "ligands_for_buster_report"):
            if summary.get(key) is not None:
                payload[f"summary.{key}"] = summary.get(key)

    geom = core.get("pdbx_vrpt_summary_geometry")
    if isinstance(geom, list) and geom:
        g0 = geom[0]
        if isinstance(g0, dict):
            for key in (
                "clashscore",
                "percent_ramachandran_outliers",
                "percent_rotamer_outliers",
                "bonds_rmsz",
                "angles_rmsz",
            ):
                if g0.get(key) is not None:
                    payload[f"geometry.{key}"] = g0.get(key)

    em = core.get("pdbx_vrpt_summary_em")
    if isinstance(em, list) and em:
        e0 = em[0]
        if isinstance(e0, dict):
            for key in (
                "qscore",
                "atom_inclusion_all_atoms",
                "atom_inclusion_backbone",
                "author_provided_fsc_resolution_by_cutoff_pt143",
            ):
                if e0.get(key) is not None:
                    payload[f"em.{key}"] = e0.get(key)

    return payload


def _clashscore_from_core(core: dict) -> str:
    geom = core.get("pdbx_vrpt_summary_geometry")
    if isinstance(geom, list) and geom:
        g0 = geom[0]
        if isinstance(g0, dict) and g0.get("clashscore") is not None:
            return str(float(g0["clashscore"]))
    return ""


def _qscore_from_core(core: dict) -> str:
    em = core.get("pdbx_vrpt_summary_em")
    if isinstance(em, list) and em:
        e0 = em[0]
        if isinstance(e0, dict) and e0.get("qscore") is not None:
            return str(float(e0["qscore"]))
    return ""


def fetch_core_entry(pdb_id: str, session: requests.Session, timeout_s: int = 120) -> dict:
    url = CORE_ENTRY_URL.format(pdb_id=pdb_id.strip().upper())
    response = session.get(url, timeout=timeout_s)
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise ValueError("core entry response must be a JSON object")
    return body


def fetch_first_protein_polymer_entity(
    pdb_id: str,
    entity_id: str,
    session: requests.Session,
    timeout_s: int = 120,
) -> dict:
    url = POLYMER_ENTITY_URL.format(pdb_id=pdb_id.strip().upper(), entity_id=str(entity_id))
    response = session.get(url, timeout=timeout_s)
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise ValueError("polymer entity response must be a JSON object")
    return body


def _organism_from_polymer(polymer: dict) -> str:
    src = polymer.get("rcsb_entity_source_organism")
    if isinstance(src, list) and src:
        name = src[0].get("scientific_name") or src[0].get("ncbi_scientific_name")
        if name:
            return str(name)
    gen = polymer.get("entity_src_gen")
    if isinstance(gen, list) and gen:
        name = gen[0].get("pdbx_gene_src_scientific_name")
        if name:
            return str(name)
    return ""


def _to_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def build_pdb_manifest_row(
    *,
    selection_spec_id: str,
    pdb_id: str,
    session: requests.Session,
) -> dict[str, str]:
    pid = pdb_id.strip().upper()
    row: dict[str, str] = {col: "" for col in PDB_MANIFEST_COLUMNS}
    row["frozen_pdb_manifest_version"] = FROZEN_PDB_MANIFEST_VERSION
    row["selection_spec_id"] = selection_spec_id
    row["pdb_id"] = pid
    row["download_status"] = "pending"
    row["parse_status"] = "pending"
    row["graph_status"] = "pending"
    row["topology_status"] = "pending"

    try:
        core = fetch_core_entry(pid, session=session)
    except requests.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else "unknown"
        row["entry_exclusion_code"] = f"rcsb_http_{code}"
        row["validation_available"] = "N"
        return row
    except requests.RequestException:
        row["entry_exclusion_code"] = "rcsb_network_error"
        row["validation_available"] = "N"
        return row

    exclusion: list[str] = []

    acc = core.get("rcsb_accession_info") or {}
    release = acc.get("initial_release_date") or ""
    if release:
        row["release_date"] = str(release)

    status = _database_status(core)
    status_code = str(status.get("status_code") or "")

    ei = core.get("rcsb_entry_info") or {}
    methodology = str(ei.get("structure_determination_methodology") or "")

    resolution = _first_resolution(ei.get("resolution_combined"))
    if resolution is not None:
        row["resolution_combined"] = str(resolution)

    exp_method = _exptl_method(core)
    row["experimental_method"] = exp_method
    modality = modality_label(exp_method)
    row["modality"] = modality
    row["resolution_band"] = resolution_band_value(resolution)

    pep_i = _to_int_or_none(ei.get("polymer_entity_count_protein"))
    if pep_i is not None:
        row["polymer_entity_count"] = str(pep_i)

    inst_i = _to_int_or_none(ei.get("deposited_polymer_entity_instance_count"))
    if inst_i is not None:
        row["deposited_protein_instance_count"] = str(inst_i)

    npc = _to_int_or_none(ei.get("nonpolymer_entity_count"))
    if npc is not None:
        row["num_nonpolymer_ligands"] = str(npc)

    components = ei.get("nonpolymer_bound_components")
    comp_joined = ""
    if isinstance(components, list):
        comp_joined = ",".join(str(x) for x in components)
        row["nonpolymer_bound_components"] = comp_joined
    elif components is not None:
        comp_joined = str(components)
        row["nonpolymer_bound_components"] = comp_joined

    has_nonpoly = (npc is not None and npc > 0) or (bool(comp_joined))
    lig_state = ligand_state_label(has_nonpoly)
    row["ligand_state"] = lig_state

    metal_bonds = _to_int_or_none(ei.get("inter_mol_metalic_bond_count"))
    row["has_metal"] = "Y" if metal_bonds is not None and metal_bonds > 0 else "N"

    kw = _keywords_text(core)
    is_membrane = "MEMBRANE" in kw
    row["is_membrane_annotated"] = "Y" if is_membrane else "N"

    vrpt = _vrpt_payload(core)
    row["validation_quality_fields"] = json.dumps(vrpt, sort_keys=True, separators=(",", ":"))
    has_summary_block = isinstance(core.get("pdbx_vrpt_summary"), dict)
    row["validation_available"] = "Y" if (vrpt or has_summary_block) else "N"

    row["qscore"] = _qscore_from_core(core)
    row["clashscore"] = _clashscore_from_core(core)

    row["mmcif_deposited_url"] = mmcif_deposited_download_url(pid)
    candidates = validation_report_pdf_candidates(pid)
    row["validation_url_primary"] = candidates[0] if candidates else ""

    if status_code != "REL":
        exclusion.append(f"status_{status_code or 'missing'}")
    if methodology != "experimental":
        exclusion.append(f"methodology_{methodology or 'missing'}")
    if pep_i is None:
        exclusion.append("polymer_count_unparsed")
    elif pep_i < 1:
        exclusion.append("no_protein_polymer")
    if inst_i is not None and inst_i != 1:
        exclusion.append("multiple_polymer_instances")

    chain_len_meta: int | None = None
    identifiers = core.get("rcsb_entry_container_identifiers") or {}
    polymer_ids = identifiers.get("polymer_entity_ids")
    polymer: dict = {}
    if not isinstance(polymer_ids, list) or not polymer_ids:
        exclusion.append("no_polymer_entity_ids")
    else:
        entity_id = str(polymer_ids[0])
        try:
            polymer = fetch_first_protein_polymer_entity(pid, entity_id, session=session)
        except requests.RequestException:
            exclusion.append("polymer_entity_fetch_failed")
            polymer = {}

        if polymer:
            organism = _organism_from_polymer(polymer)
            if organism:
                row["organism"] = organism

            epoly = polymer.get("entity_poly") or {}
            strand_raw = epoly.get("pdbx_strand_id")
            strands: list[str] = []
            if isinstance(strand_raw, str) and strand_raw.strip():
                for piece in strand_raw.split(","):
                    s = piece.strip()
                    if s:
                        strands.append(s)
            strands = sorted(set(strands))
            row["protein_chains_in_entity"] = ",".join(strands)
            if strands:
                row["chain_id"] = strands[0]

            sample_len = _to_int_or_none(epoly.get("rcsb_sample_sequence_length"))
            if sample_len is not None:
                chain_len_meta = sample_len
                row["chain_length_metadata"] = str(chain_len_meta)

    if chain_len_meta is not None:
        if chain_len_meta < pol.MIN_DOMAIN_LENGTH or chain_len_meta > pol.MAX_DOMAIN_LENGTH:
            exclusion.append(f"length_outside_{pol.MIN_DOMAIN_LENGTH}_{pol.MAX_DOMAIN_LENGTH}")

    stratum = assign_pdb_stratum(
        modality=modality,
        resolution=resolution,
        is_membrane_annotated=is_membrane,
        ligand_state=lig_state,
        deposited_protein_instance_count=inst_i,
        chain_length_metadata=chain_len_meta,
    )
    row["pdb_stratum"] = stratum

    if exclusion:
        row["entry_exclusion_code"] = ";".join(sorted(set(exclusion)))

    return row


def normalize_manifest_row(row: dict[str, str]) -> dict[str, str]:
    return {col: str(row.get(col, "") or "") for col in PDB_MANIFEST_COLUMNS}


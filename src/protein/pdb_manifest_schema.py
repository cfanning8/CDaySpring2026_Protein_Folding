from __future__ import annotations

# Frozen PDB extraction manifest schema (metadata-first; coordinates are pass 2).
FROZEN_PDB_MANIFEST_VERSION = "2026-04-02-pdb-neurips-v1"

# Column order for CSV interchange and audit trails.
PDB_MANIFEST_COLUMNS = [
    "frozen_pdb_manifest_version",
    "selection_spec_id",
    "pdb_id",
    "release_date",
    "experimental_method",
    "modality",
    "resolution_combined",
    "resolution_band",
    "polymer_entity_count",
    "deposited_protein_instance_count",
    "chain_id",
    "protein_chains_in_entity",
    "chain_length_metadata",
    "chain_length_parsed",
    "num_ca_atoms",
    "fraction_ca_present",
    "num_nonpolymer_ligands",
    "nonpolymer_bound_components",
    "ligand_state",
    "has_metal",
    "organism",
    "is_membrane_annotated",
    "qscore",
    "clashscore",
    "validation_available",
    "validation_quality_fields",
    "mmcif_deposited_url",
    "validation_url_primary",
    "pdb_stratum",
    "entry_exclusion_code",
    "mmcif_sha256",
    "validation_pdf_sha256",
    "download_status",
    "parse_status",
    "graph_status",
    "topology_status",
]

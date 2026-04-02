from __future__ import annotations

# Frozen dataset selection spec for metadata-first corpus construction.
FROZEN_SELECTION_SPEC_ID = "protein_domain_cath_rcsb_v1"

# Domain extent (residue count) from CATH CLF column 11.
MIN_DOMAIN_LENGTH = 80
MAX_DOMAIN_LENGTH = 800

# RCSB filters (structure determination in the mmCIF entry).
XRAY_METHOD = "X-RAY DIFFRACTION"
CRYO_EM_METHOD = "ELECTRON MICROSCOPY"
XRAY_RESOLUTION_MAX = 3.0
CRYOEM_RESOLUTION_MAX = 4.0

# Single polymer instance in the deposited asymmetric unit (first benchmark policy).
DEPOSITED_POLYMER_ENTITY_INSTANCE_COUNT = 1

# CATH CLF resolution column: 999.000 marks NMR; 1000.000 marks obsolete PDB entries.
# Eligibility uses `res >= 999.0` in `eligible_domains.domain_passes_policy`.

# Optional additional cap on the CATH-reported resolution column (Angstroms).
CATH_RESOLUTION_MAX = 4.0

# For the first automated pipeline, restrict to whole-chain CATH domains (domain id ends in "00").
REQUIRE_WHOLE_CHAIN_DOMAIN_SUFFIX = "00"

# Skip CATH chain placeholder that means "no chain field" until a dedicated loader exists.
EXCLUDE_CATH_CHAIN_PLACEHOLDER_ZERO = True

# Primary topology graph (C_beta / virtual C_beta representatives; see residue_points.py).
# Literature ranges for CA/CB contact caps are often ~7-8 A (representation-dependent).
TOPOLOGY_GRAPH_POLICY_ID = "topology_v2_cb_backbone_filtration0_r8"

# Spatial-contact cap R_max for primary topology graph (Angstroms).
CONTACT_GRAPH_RADIUS_MAX_A = 8.0

# Random sampling seed for stratified smoke manifests.
SMOKE_SAMPLE_RANDOM_SEED = 14


def rcsb_policy_query_group() -> dict:
    return {
        "type": "group",
        "logical_operator": "and",
        "nodes": [
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entry_info.structure_determination_methodology",
                    "operator": "exact_match",
                    "value": "experimental",
                },
            },
            {
                "type": "group",
                "logical_operator": "or",
                "nodes": [
                    {
                        "type": "group",
                        "logical_operator": "and",
                        "nodes": [
                            {
                                "type": "terminal",
                                "service": "text",
                                "parameters": {
                                    "attribute": "exptl.method",
                                    "operator": "exact_match",
                                    "value": XRAY_METHOD,
                                },
                            },
                            {
                                "type": "terminal",
                                "service": "text",
                                "parameters": {
                                    "attribute": "rcsb_entry_info.resolution_combined",
                                    "operator": "less_or_equal",
                                    "value": XRAY_RESOLUTION_MAX,
                                },
                            },
                        ],
                    },
                    {
                        "type": "group",
                        "logical_operator": "and",
                        "nodes": [
                            {
                                "type": "terminal",
                                "service": "text",
                                "parameters": {
                                    "attribute": "exptl.method",
                                    "operator": "exact_match",
                                    "value": CRYO_EM_METHOD,
                                },
                            },
                            {
                                "type": "terminal",
                                "service": "text",
                                "parameters": {
                                    "attribute": "rcsb_entry_info.resolution_combined",
                                    "operator": "less_or_equal",
                                    "value": CRYOEM_RESOLUTION_MAX,
                                },
                            },
                        ],
                    },
                ],
            },
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entry_info.deposited_polymer_entity_instance_count",
                    "operator": "equals",
                    "value": DEPOSITED_POLYMER_ENTITY_INSTANCE_COUNT,
                },
            },
        ],
    }

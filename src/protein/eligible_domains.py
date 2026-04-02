from __future__ import annotations

from src.protein import dataset_policy as pol


def domain_passes_policy(row: dict, rcsb_ids_upper: set[str]) -> bool:
    if row["pdb_id"] not in rcsb_ids_upper:
        return False
    if row["domain_id"][5:7] != pol.REQUIRE_WHOLE_CHAIN_DOMAIN_SUFFIX:
        return False
    if pol.EXCLUDE_CATH_CHAIN_PLACEHOLDER_ZERO and row["chain_id"] == "0":
        return False
    length = int(row["domain_length"])
    if length < pol.MIN_DOMAIN_LENGTH or length > pol.MAX_DOMAIN_LENGTH:
        return False
    res = float(row["cath_resolution"])
    if not (res == res) or res <= 0:
        return False
    if res >= 999.0:
        return False
    if res > pol.CATH_RESOLUTION_MAX:
        return False
    return True

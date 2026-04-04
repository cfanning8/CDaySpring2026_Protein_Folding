from __future__ import annotations

# Deposited entry mmCIF via RCSB file service (asymmetric unit; not biological assembly).
def mmcif_deposited_download_url(pdb_id: str) -> str:
    pid = pdb_id.strip().upper()
    if len(pid) != 4:
        raise ValueError("pdb_id must be four characters")
    return f"https://files.rcsb.org/download/{pid}.cif"


def mmcif_deposited_gz_url(pdb_id: str) -> str:
    low = pdb_id.strip().lower()
    if len(low) != 4:
        raise ValueError("pdb_id must be four characters")
    mid = low[1:3]
    return f"https://files.rcsb.org/pub/pdb/data/structures/divided/mm_cif/{mid}/{low}.cif.gz"


def validation_report_pdf_candidates(pdb_id: str) -> list[str]:
    low = pdb_id.strip().lower()
    if len(low) != 4:
        raise ValueError("pdb_id must be four characters")
    mid = low[1:3]
    base = f"{low}_validation.pdf"
    return [
        f"https://www.ebi.ac.uk/pdbe/entry-files/{base}",
        f"https://files.rcsb.org/pub/pdb/validation_reports/{mid}/{low}/{base}",
        f"https://ftp.pdbj.org/pub/pdb/validation_reports/{mid}/{low}/{base}",
    ]

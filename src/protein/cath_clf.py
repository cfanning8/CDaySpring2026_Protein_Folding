from __future__ import annotations

import gzip
from collections.abc import Iterator
from pathlib import Path


def _open_text(path: Path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    if str(path).lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="ascii", errors="strict")
    return path.open("r", encoding="ascii", errors="strict")


def iter_cath_clf_rows(domain_list_path: Path) -> Iterator[dict]:
    with _open_text(domain_list_path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 12:
                raise ValueError("CATH CLF row has fewer than 12 columns")

            domain_id = parts[0]
            if len(domain_id) != 7:
                raise ValueError(f"unexpected CATH domain id length: {domain_id!r}")

            yield {
                "domain_id": domain_id,
                "pdb_id": domain_id[0:4].upper(),
                "chain_id": domain_id[4],
                "class_num": parts[1],
                "arch_num": parts[2],
                "topology_num": parts[3],
                "homologous_superfamily_num": parts[4],
                "superfamily": f"{parts[1]}.{parts[2]}.{parts[3]}.{parts[4]}",
                "s35_cluster": parts[5],
                "domain_length": int(parts[10]),
                "cath_resolution": float(parts[11]),
            }

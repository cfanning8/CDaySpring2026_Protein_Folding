from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.protein.validation_urls import (  # noqa: E402
    mmcif_deposited_gz_url,
    mmcif_deposited_download_url,
    validation_report_pdf_candidates,
)


def _get(session: requests.Session, url: str, timeout_s: int) -> bytes:
    response = session.get(url, timeout=timeout_s)
    response.raise_for_status()
    return response.content


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pass 2: download deposited entry mmCIF and validation PDF for manifest rows.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument(
        "--mmcif-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "protein" / "mmcif",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "protein" / "validation",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--allow-excluded",
        action="store_true",
        help="Download even when entry_exclusion_code is non-empty (not recommended).",
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--sleep-s", type=float, default=0.2)
    args = parser.parse_args()

    with args.manifest.open("r", encoding="ascii", errors="strict") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("manifest has no header")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    args.mmcif_dir.mkdir(parents=True, exist_ok=True)
    args.validation_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "cdays-protein-pipeline/1.0 (+https://files.rcsb.org)"})

    processed = 0
    stamp = datetime.now(timezone.utc).isoformat()
    try:
        for row in rows:
            if args.max_rows is not None and processed >= args.max_rows:
                break
            excl = (row.get("entry_exclusion_code") or "").strip()
            if excl and not args.allow_excluded:
                continue

            pdb_id = (row.get("pdb_id") or "").strip().upper()
            if len(pdb_id) != 4:
                row["download_status"] = "skipped_bad_pdb_id"
                continue
            processed += 1

            mmcif_path = args.mmcif_dir / f"{pdb_id}.cif"
            val_path = args.validation_dir / f"{pdb_id}_validation.pdf"

            notes: list[str] = []

            try:
                if args.skip_existing and mmcif_path.is_file():
                    notes.append("mmcif_skip_existing")
                    row["mmcif_sha256"] = hashlib.sha256(mmcif_path.read_bytes()).hexdigest()
                else:
                    url = (row.get("mmcif_deposited_url") or "").strip() or mmcif_deposited_download_url(pdb_id)
                    mmcif_bytes = _get(session, url, args.timeout_s)
                    mmcif_path.write_bytes(mmcif_bytes)
                    row["mmcif_sha256"] = hashlib.sha256(mmcif_bytes).hexdigest()
                    notes.append("mmcif_ok")
            except requests.RequestException as exc:
                try:
                    gz_url = mmcif_deposited_gz_url(pdb_id)
                    body = _get(session, gz_url, args.timeout_s)
                    dec = gzip.decompress(body)
                    mmcif_path.write_bytes(dec)
                    row["mmcif_sha256"] = hashlib.sha256(dec).hexdigest()
                    notes.append("mmcif_ok_gz")
                except (requests.RequestException, OSError, EOFError):
                    row["mmcif_sha256"] = ""
                    row["download_status"] = f"mmcif_failed:{exc}"
                    time.sleep(float(args.sleep_s))
                    continue

            if args.skip_existing and val_path.is_file():
                notes.append("validation_skip_existing")
                row["validation_pdf_sha256"] = hashlib.sha256(val_path.read_bytes()).hexdigest()
            else:
                got = False
                primary = (row.get("validation_url_primary") or "").strip()
                tried: list[str] = []
                for cand in ([primary] if primary else []) + validation_report_pdf_candidates(pdb_id):
                    if not cand or cand in tried:
                        continue
                    tried.append(cand)
                    try:
                        pdf_bytes = _get(session, cand, args.timeout_s)
                        val_path.write_bytes(pdf_bytes)
                        row["validation_pdf_sha256"] = hashlib.sha256(pdf_bytes).hexdigest()
                        got = True
                        notes.append("validation_ok")
                        break
                    except requests.RequestException:
                        continue
                if not got:
                    notes.append("validation_missing")

            row["download_status"] = ";".join(notes) + f";ts={stamp}"
            time.sleep(float(args.sleep_s))
    finally:
        session.close()

    with args.out_manifest.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote={args.out_manifest.resolve()} processed={processed}")


if __name__ == "__main__":
    main()

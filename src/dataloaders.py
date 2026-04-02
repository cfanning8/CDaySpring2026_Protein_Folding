from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict
import xml.etree.ElementTree as ET

import pandas as pd
from pandas.errors import ParserError

DATA_DIR_NAME = "data"
MALAWI_DIR_NAME = "tnet_malawi_pilot.csv"
SUPPORTED_SUFFIXES = {".csv", ".txt", ".dat", ".json", ".gexf"}
NUMERIC_TOKEN = re.compile(r"^-?\d+(\.\d+)?$")


def get_default_data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / DATA_DIR_NAME


def load_all_datasets(data_dir: Path | None = None) -> Dict[str, pd.DataFrame]:
    root = data_dir or get_default_data_dir()
    datasets: Dict[str, pd.DataFrame] = {}

    malawi_dir = root / MALAWI_DIR_NAME
    if malawi_dir.is_dir():
        datasets[f"{MALAWI_DIR_NAME}/parsed_rows"] = load_malawi_filename_rows(malawi_dir)

    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file():
            continue
        if "__MACOSX" in file_path.parts:
            continue
        if file_path.name.startswith("._"):
            continue
        if _is_malawi_fragment(file_path):
            continue
        if file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        rel_key = str(file_path.relative_to(root))
        datasets[rel_key] = load_single_dataset(file_path)

    return datasets


def load_single_dataset(file_path: Path) -> pd.DataFrame:
    if file_path.name.lower().startswith("readme"):
        return _load_plain_text_lines(file_path)
    suffix = file_path.suffix.lower()
    if suffix == ".json":
        return _load_interval_json(file_path)
    if suffix == ".gexf":
        return _load_gexf_edges(file_path)
    return _load_text_table(file_path)


def _load_text_table(file_path: Path) -> pd.DataFrame:
    delimiter = _detect_delimiter(file_path)
    has_header = _detect_header(file_path, delimiter)
    read_kwargs = {
        "sep": delimiter,
        "header": 0 if has_header else None,
        "engine": "python",
    }
    try:
        frame = pd.read_csv(file_path, encoding="utf-8", **read_kwargs)
    except UnicodeDecodeError:
        try:
            frame = pd.read_csv(file_path, encoding="latin-1", **read_kwargs)
        except ParserError:
            return _load_plain_text_lines(file_path)
    except ParserError:
        return _load_plain_text_lines(file_path)

    if frame.shape[1] == 1 and delimiter == ",":
        try:
            frame = pd.read_csv(file_path, sep=r"\s+", header=0 if has_header else None, engine="python")
        except ParserError:
            pass

    if not has_header:
        frame.columns = _default_column_names(frame.shape[1])
    return frame


def _detect_delimiter(file_path: Path) -> str:
    sample_lines = _sample_lines(file_path, line_count=10)
    sample_text = "\n".join(sample_lines)
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=",\t;|")
        if dialect.delimiter in {",", "\t", ";", "|"}:
            return dialect.delimiter
    except csv.Error:
        pass
    if "," in sample_text:
        return ","
    if "\t" in sample_text:
        return "\t"
    return r"\s+"


def _detect_header(file_path: Path, delimiter: str) -> bool:
    sample_lines = _sample_lines(file_path, line_count=8)
    if not sample_lines:
        return False
    sample_text = "\n".join(sample_lines)
    if delimiter != r"\s+":
        try:
            return bool(csv.Sniffer().has_header(sample_text))
        except csv.Error:
            pass
    first_row = _split_row(sample_lines[0], delimiter)
    if not first_row:
        return False
    first_numeric_ratio = _numeric_ratio(first_row)
    if len(sample_lines) == 1:
        return first_numeric_ratio < 0.5
    second_row = _split_row(sample_lines[1], delimiter)
    if not second_row:
        return first_numeric_ratio < 0.5
    second_numeric_ratio = _numeric_ratio(second_row)
    return first_numeric_ratio < second_numeric_ratio


def _sample_lines(file_path: Path, line_count: int) -> list[str]:
    lines: list[str] = []
    with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for _ in range(line_count):
            line = handle.readline()
            if not line:
                break
            lines.append(line.rstrip("\n"))
    return lines


def _split_row(line: str, delimiter: str) -> list[str]:
    if delimiter == ",":
        return [value.strip().strip('"') for value in next(csv.reader([line]))]
    if delimiter == "\t":
        return [value.strip().strip('"') for value in line.split("\t")]
    return [value.strip().strip('"') for value in line.split()]


def _numeric_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    numeric_count = 0
    for token in tokens:
        if NUMERIC_TOKEN.match(token):
            numeric_count += 1
    return numeric_count / len(tokens)


def _default_column_names(size: int) -> list[str]:
    return [f"col_{index}" for index in range(size)]


def _load_plain_text_lines(file_path: Path) -> pd.DataFrame:
    with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
        lines = [line.rstrip("\n") for line in handle]
    return pd.DataFrame({"text": lines})


def _load_interval_json(file_path: Path) -> pd.DataFrame:
    raw = json.loads(file_path.read_text(encoding="utf-8"))
    rows: list[dict[str, int]] = []
    for source, neighbors in raw.items():
        if not isinstance(neighbors, dict):
            continue
        for target, intervals in neighbors.items():
            for interval in intervals:
                if len(interval) != 2:
                    continue
                rows.append(
                    {
                        "source": int(source),
                        "target": int(target),
                        "start": int(interval[0]),
                        "end": int(interval[1]),
                    }
                )
    return pd.DataFrame(rows)


def _load_gexf_edges(file_path: Path) -> pd.DataFrame:
    root = ET.parse(file_path).getroot()
    rows: list[dict[str, str]] = []
    for edge in root.findall(".//{*}edge"):
        rows.append(
            {
                "id": edge.get("id", ""),
                "source": edge.get("source", ""),
                "target": edge.get("target", ""),
                "weight": edge.get("weight", ""),
            }
        )
    return pd.DataFrame(rows)


def _is_malawi_fragment(file_path: Path) -> bool:
    return file_path.parent.name == MALAWI_DIR_NAME and file_path.suffix == ""


def load_malawi_filename_rows(malawi_dir: Path) -> pd.DataFrame:
    rows: list[list[str]] = []
    columns: list[str] | None = None

    for fragment in sorted(malawi_dir.iterdir()):
        if not fragment.is_file():
            continue
        values = next(csv.reader([fragment.name]))
        if values and values[0] == "":
            columns = [column if column else "row_id" for column in values]
            continue
        rows.append(values)

    if columns is None:
        columns = [f"col_{index}" for index in range(len(rows[0]) if rows else 0)]

    expected_size = len(columns)
    normalized_rows: list[list[str]] = []
    for row in rows:
        if len(row) < expected_size:
            normalized_rows.append(row + [""] * (expected_size - len(row)))
        else:
            normalized_rows.append(row[:expected_size])

    frame = pd.DataFrame(normalized_rows, columns=columns)
    for column in frame.columns:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError):
            continue
    return frame

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class EdgePreparationResult:
    events: pd.DataFrame
    canonical_edges: pd.DataFrame
    rule: str


@dataclass(frozen=True)
class TemporalEventResult:
    events: pd.DataFrame
    rule: str


def prepare_edges_for_dataset(dataset_key: str, frame: pd.DataFrame) -> EdgePreparationResult | None:
    events, rule = _infer_event_edges(dataset_key, frame)
    if events is None or events.empty:
        return None

    edges = events.copy()
    edges["source"] = edges["source"].astype(str)
    edges["target"] = edges["target"].astype(str)
    edges["duration_seconds"] = pd.to_numeric(edges["duration_seconds"], errors="coerce").fillna(0.0)
    edges = edges[edges["source"] != edges["target"]]
    edges = edges[edges["duration_seconds"] > 0]
    edges = edges.drop_duplicates(subset=["source", "target", "duration_seconds"])
    if edges.empty:
        return None

    # Treat contacts as undirected for the default TDA edge object.
    order_mask = edges["source"] <= edges["target"]
    edges["u"] = edges["source"].where(order_mask, edges["target"])
    edges["v"] = edges["target"].where(order_mask, edges["source"])

    canonical = (
        edges.groupby(["u", "v"], as_index=False)
        .agg(
            duration_seconds=("duration_seconds", "sum"),
            event_count=("duration_seconds", "size"),
        )
        .rename(columns={"u": "source", "v": "target"})
    )

    return EdgePreparationResult(events=edges, canonical_edges=canonical, rule=rule)


def extract_temporal_events_for_dataset(dataset_key: str, frame: pd.DataFrame) -> TemporalEventResult | None:
    events, rule = _infer_temporal_events(dataset_key, frame)
    if events is None or events.empty:
        return None
    parsed = events.copy()
    parsed["source"] = parsed["source"].astype(str)
    parsed["target"] = parsed["target"].astype(str)
    parsed["duration_seconds"] = pd.to_numeric(parsed["duration_seconds"], errors="coerce")
    parsed["t_start"] = pd.to_numeric(parsed["t_start"], errors="coerce")
    parsed = parsed.dropna(subset=["source", "target", "duration_seconds", "t_start"])
    parsed = parsed[parsed["source"] != parsed["target"]]
    parsed = parsed[(parsed["duration_seconds"] > 0) & (parsed["t_start"] >= 0)]
    parsed = parsed.drop_duplicates(subset=["source", "target", "t_start", "duration_seconds"])
    if parsed.empty:
        return None
    return TemporalEventResult(events=parsed, rule=rule)


def _infer_event_edges(dataset_key: str, frame: pd.DataFrame) -> tuple[pd.DataFrame | None, str]:
    lower_key = _normalized_key(dataset_key)
    if any(token in lower_key for token in ("metadata", "readme", "variables_dictionary")):
        return None, "excluded_metadata_or_readme"

    columns = {str(column).strip().lower(): column for column in frame.columns}

    if _is_kenya_contacts(columns):
        events = pd.DataFrame(
            {
                "source": frame[columns["h1"]].astype(str) + "_" + frame[columns["m1"]].astype(str),
                "target": frame[columns["h2"]].astype(str) + "_" + frame[columns["m2"]].astype(str),
                "duration_seconds": pd.to_numeric(frame[columns["duration"]], errors="coerce"),
            }
        )
        return events, "kenya_duration_column"

    if "tnet_malawi_pilot.csv/parsed_rows" in lower_key and {"id1", "id2"}.issubset(columns):
        events = pd.DataFrame(
            {
                "source": frame[columns["id1"]].astype(str),
                "target": frame[columns["id2"]].astype(str),
                "duration_seconds": 20.0,
            }
        )
        return events, "malawi_20_second_rows"

    if {"duration_sec", "indid1", "indid2"}.issubset(columns):
        events = pd.DataFrame(
            {
                "source": frame[columns["indid1"]].astype(str),
                "target": frame[columns["indid2"]].astype(str),
                "duration_seconds": pd.to_numeric(frame[columns["duration_sec"]], errors="coerce"),
            }
        )
        return events, "duration_sec_column"

    if {"source", "target", "start", "end"}.issubset(columns):
        start_values = pd.to_numeric(frame[columns["start"]], errors="coerce")
        end_values = pd.to_numeric(frame[columns["end"]], errors="coerce")
        events = pd.DataFrame(
            {
                "source": frame[columns["source"]].astype(str),
                "target": frame[columns["target"]].astype(str),
                "duration_seconds": (end_values - start_values).clip(lower=0),
            }
        )
        return events, "interval_end_minus_start"

    if {"t", "i", "j"}.issubset(columns) and _is_known_temporal(dataset_key):
        events = pd.DataFrame(
            {
                "source": frame[columns["i"]].astype(str),
                "target": frame[columns["j"]].astype(str),
                "duration_seconds": 20.0,
            }
        )
        return events, "20_second_temporal_contacts"

    if {"source", "target", "weight"}.issubset(columns):
        weight = pd.to_numeric(frame[columns["weight"]], errors="coerce")
        events = pd.DataFrame(
            {
                "source": frame[columns["source"]].astype(str),
                "target": frame[columns["target"]].astype(str),
                "duration_seconds": weight.where(weight.notna(), 0.0),
            }
        )
        return events, "gexf_weight_as_duration_proxy"

    if {"col_0", "col_1", "col_2"}.issubset(columns) and _is_known_temporal(dataset_key):
        events = pd.DataFrame(
            {
                "source": frame[columns["col_1"]].astype(str),
                "target": frame[columns["col_2"]].astype(str),
                "duration_seconds": 20.0,
            }
        )
        return events, "20_second_temporal_contacts_generic_columns"

    if {"col_0", "col_1", "col_2"}.issubset(columns) and _is_known_static_weighted(dataset_key):
        events = pd.DataFrame(
            {
                "source": frame[columns["col_0"]].astype(str),
                "target": frame[columns["col_1"]].astype(str),
                "duration_seconds": pd.to_numeric(frame[columns["col_2"]], errors="coerce"),
            }
        )
        return events, "static_weight_proxy"

    if {"i", "j"}.issubset(columns):
        events = pd.DataFrame(
            {
                "source": frame[columns["i"]].astype(str),
                "target": frame[columns["j"]].astype(str),
                "duration_seconds": 1.0,
            }
        )
        return events, "static_edge_unit_weight"

    if {"col_0", "col_1"}.issubset(columns) and _is_known_static_unweighted(dataset_key):
        events = pd.DataFrame(
            {
                "source": frame[columns["col_0"]].astype(str),
                "target": frame[columns["col_1"]].astype(str),
                "duration_seconds": 1.0,
            }
        )
        return events, "static_edge_unit_weight_generic_columns"

    return None, "not_inferable"


def _is_kenya_contacts(columns: dict[str, str]) -> bool:
    required = {"h1", "m1", "h2", "m2", "duration"}
    return required.issubset(columns)


def _is_known_temporal(dataset_key: str) -> bool:
    lower_key = _normalized_key(dataset_key)
    tokens = (
        "infectious/sciencegallery_infectious_contacts",
        "hypertext/ht2009_contact_list.dat",
        "high_school/highschool2013_proximity_net.csv",
        "primary_school/primaryschool.csv",
        "workplace/workplace_invs15_tij.dat",
        "ward/hospital_lyon_contacts.dat",
        "sfhh_tij.dat",
    )
    return any(token in lower_key for token in tokens)


def _is_known_static_weighted(dataset_key: str) -> bool:
    lower_key = _normalized_key(dataset_key)
    return any(
        token in lower_key
        for token in (
            "highschool2013_facebook_known_pairs",
            "highschool2013_contactdiaries_network",
        )
    )


def _is_known_static_unweighted(dataset_key: str) -> bool:
    lower_key = _normalized_key(dataset_key)
    return "highschool2013_friendship_network" in lower_key


def _infer_temporal_events(dataset_key: str, frame: pd.DataFrame) -> tuple[pd.DataFrame | None, str]:
    lower_key = _normalized_key(dataset_key)
    if any(token in lower_key for token in ("metadata", "readme", "variables_dictionary")):
        return None, "excluded_metadata_or_readme"

    columns = {str(column).strip().lower(): column for column in frame.columns}

    if _is_kenya_contacts(columns) and {"day", "hour"}.issubset(columns):
        day_values = pd.to_numeric(frame[columns["day"]], errors="coerce").fillna(0)
        hour_values = pd.to_numeric(frame[columns["hour"]], errors="coerce").fillna(0)
        events = pd.DataFrame(
            {
                "source": frame[columns["h1"]].astype(str) + "_" + frame[columns["m1"]].astype(str),
                "target": frame[columns["h2"]].astype(str) + "_" + frame[columns["m2"]].astype(str),
                "duration_seconds": pd.to_numeric(frame[columns["duration"]], errors="coerce"),
                "t_start": ((day_values - day_values.min()) * 24 * 3600) + (hour_values * 3600),
            }
        )
        return events, "kenya_day_hour_proxy_time"

    if "tnet_malawi_pilot.csv/parsed_rows" in lower_key and {"id1", "id2", "contact_time"}.issubset(columns):
        events = pd.DataFrame(
            {
                "source": frame[columns["id1"]].astype(str),
                "target": frame[columns["id2"]].astype(str),
                "duration_seconds": 20.0,
                "t_start": pd.to_numeric(frame[columns["contact_time"]], errors="coerce"),
            }
        )
        return events, "malawi_contact_time"

    if {"duration_sec", "indid1", "indid2", "t"}.issubset(columns):
        time_values = pd.to_datetime(frame[columns["t"]], errors="coerce")
        t_seconds = (time_values - time_values.min()).dt.total_seconds()
        events = pd.DataFrame(
            {
                "source": frame[columns["indid1"]].astype(str),
                "target": frame[columns["indid2"]].astype(str),
                "duration_seconds": pd.to_numeric(frame[columns["duration_sec"]], errors="coerce"),
                "t_start": t_seconds,
            }
        )
        return events, "duration_sec_with_timestamp"

    if {"source", "target", "start", "end"}.issubset(columns):
        start_values = pd.to_numeric(frame[columns["start"]], errors="coerce")
        end_values = pd.to_numeric(frame[columns["end"]], errors="coerce")
        events = pd.DataFrame(
            {
                "source": frame[columns["source"]].astype(str),
                "target": frame[columns["target"]].astype(str),
                "duration_seconds": (end_values - start_values).clip(lower=0),
                "t_start": start_values,
            }
        )
        return events, "interval_start_end"

    if {"t", "i", "j"}.issubset(columns) and _is_known_temporal(dataset_key):
        events = pd.DataFrame(
            {
                "source": frame[columns["i"]].astype(str),
                "target": frame[columns["j"]].astype(str),
                "duration_seconds": 20.0,
                "t_start": pd.to_numeric(frame[columns["t"]], errors="coerce"),
            }
        )
        return events, "20_second_with_t_column"

    if {"col_0", "col_1", "col_2"}.issubset(columns) and _is_known_temporal(dataset_key):
        events = pd.DataFrame(
            {
                "source": frame[columns["col_1"]].astype(str),
                "target": frame[columns["col_2"]].astype(str),
                "duration_seconds": 20.0,
                "t_start": pd.to_numeric(frame[columns["col_0"]], errors="coerce"),
            }
        )
        return events, "20_second_generic_t_column"

    return None, "not_temporal"


def default_canonical_output_dir(project_root: Path) -> Path:
    return project_root / "temp" / "canonical_edges"


def _normalized_key(dataset_key: str) -> str:
    return Path(dataset_key).as_posix().lower()

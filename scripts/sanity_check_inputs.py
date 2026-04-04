from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataloaders import load_all_datasets
from src.edge_preparation import prepare_edges_for_dataset


def main() -> None:
    datasets = load_all_datasets(PROJECT_ROOT / "data")
    issues: list[str] = []
    summary_rows: list[dict[str, object]] = []

    for key, frame in datasets.items():
        basic = _basic_checks(key, frame)
        issues.extend(basic["issues"])

        result = prepare_edges_for_dataset(key, frame)
        if result is None:
            summary_rows.append(
                {
                    "dataset_key": key,
                    "rows": len(frame),
                    "cols": len(frame.columns),
                    "edge_inferred": False,
                    "rule": "not_inferable",
                    "nodes": 0,
                    "edges": 0,
                    "total_duration_seconds": 0.0,
                    "self_loop_ratio": 0.0,
                }
            )
            continue

        edges = result.canonical_edges
        node_count = len(set(edges["source"]).union(set(edges["target"])))
        self_loop_ratio = float((edges["source"] == edges["target"]).mean()) if len(edges) else 0.0
        total_duration = float(edges["duration_seconds"].sum())

        if node_count < 5:
            issues.append(f"{key}: very small inferred graph ({node_count} nodes)")
        if len(edges) < 10:
            issues.append(f"{key}: very small inferred graph ({len(edges)} edges)")
        if total_duration <= 0:
            issues.append(f"{key}: non-positive total canonical duration")
        if self_loop_ratio > 0.20:
            issues.append(f"{key}: high self-loop ratio ({self_loop_ratio:.3f})")

        summary_rows.append(
            {
                "dataset_key": key,
                "rows": len(frame),
                "cols": len(frame.columns),
                "edge_inferred": True,
                "rule": result.rule,
                "nodes": node_count,
                "edges": len(edges),
                "total_duration_seconds": total_duration,
                "self_loop_ratio": self_loop_ratio,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("dataset_key")
    output_dir = PROJECT_ROOT / "temp"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "sanity_summary.csv"
    issues_path = output_dir / "sanity_issues.txt"
    summary.to_csv(summary_path, index=False)
    issues_path.write_text("\n".join(issues) + ("\n" if issues else ""), encoding="utf-8")

    print(f"Datasets checked: {len(datasets)}")
    print(f"Summary file: {summary_path}")
    print(f"Issues file: {issues_path}")
    print(f"Issues found: {len(issues)}")
    if issues:
        for issue in issues[:30]:
            print(f"- {issue}")
        raise SystemExit(1)

    print("All sanity checks passed.")


def _basic_checks(dataset_key: str, frame: pd.DataFrame) -> dict[str, list[str]]:
    issues: list[str] = []
    row_count = len(frame)
    col_count = len(frame.columns)

    if row_count == 0:
        issues.append(f"{dataset_key}: empty table")
    if col_count == 0:
        issues.append(f"{dataset_key}: zero columns")

    null_fraction = float(frame.isna().mean().mean()) if row_count and col_count else 0.0
    if null_fraction > 0.50:
        issues.append(f"{dataset_key}: high missing-value fraction ({null_fraction:.3f})")

    if dataset_key.lower().endswith("readme.txt"):
        if "text" not in [str(column).lower() for column in frame.columns]:
            issues.append(f"{dataset_key}: README should load as plain text lines")

    # Generic parse smell: single-column tables where many rows appear delimited.
    if col_count == 1 and row_count > 10:
        sample = frame.iloc[:50, 0].astype(str)
        delimited_ratio = float(sample.str.contains(r"[,\\t ]").mean())
        lower_key = dataset_key.lower()
        if (
            delimited_ratio > 0.80
            and "variables_dictionary" not in lower_key
            and "readme" not in lower_key
            and "metadata" not in lower_key
        ):
            issues.append(f"{dataset_key}: possible delimiter misparse (single delimited text column)")

    return {"issues": issues}


if __name__ == "__main__":
    main()

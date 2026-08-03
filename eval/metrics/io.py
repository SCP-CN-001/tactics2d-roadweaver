# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Evaluation CSV input/output implementation."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def save_results(output_dir: Path, name: str, agg: dict, per_map: list | None = None):
    """Save per-map evaluation results to CSV + aggregate summary.

    Writes two files:
      * ``{name}.csv``        — one row per map
      * ``{name}_summary.csv`` — aggregate (mean ± std per metric)

    Args:
        output_dir: Output directory.
        name: Base filename.
        agg: Aggregated metrics dict (keys like ``lcc_mean``, ``lcc_std``, …).
        per_map: List of per-map metric dicts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if per_map:
        # Collect all keys across all maps
        all_keys: list[str] = []
        seen = set()
        for m in per_map:
            for k in m:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

        # Normalise: node_count/num_nodes → n_nodes, edge_count → n_edges
        col_map = {"node_count": "n_nodes", "num_nodes": "n_nodes", "edge_count": "n_edges"}
        cols = [col_map.get(k, k) for k in all_keys]

        with open(output_dir / f"{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for m in per_map:
                row = []
                for k in all_keys:
                    v = m.get(k, "")
                    if isinstance(v, (np.floating,)):
                        v = float(v)
                    elif isinstance(v, (np.integer,)):
                        v = int(v)
                    row.append(v)
                w.writerow(row)

    # Summary (aggregate)
    if agg:
        with open(output_dir / f"{name}_summary.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "mean", "std"])
            for k, v in agg.items():
                if k.endswith("_mean"):
                    base = k[:-5]
                    std_val = agg.get(f"{base}_std", "")
                    w.writerow([base, v, std_val])
                elif k in ("n_maps", "avg_generation_time", "osm_reference_avg_degree"):
                    w.writerow([k, v, ""])


def print_results_table(title: str, results: dict, fields: list[tuple[str, str, str]]):
    """Pretty-print a results table.

    Args:
        title: Section title.
        results: Dict with '{key}_mean' and '{key}_std'.
        fields: List of (key, display_name, format_spec) like ('lcc', 'LCC', '.4f').
    """
    print(f"\n  Results:")
    for key, name, fmt in fields:
        mean = results.get(f"{key}_mean", 0)
        std = results.get(f"{key}_std", 0)
        print(f"    {name:20s}  {mean:{fmt}} ± {std:{fmt}}")


def load_csv_rows(csv_path: Path) -> list[dict]:
    """Load a CSV file as a list of dicts (values remain strings)."""
    if not Path(csv_path).exists():
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def save_binned_summary(
    output_dir: Path, name: str, per_map: list[dict], bin_col: str = "node_bin"
) -> None:
    """Aggregate per-map metrics by node-count bin and write a summary CSV.

    Writes ``{name}_by_bin.csv`` with one row per bin (ascending).  Numeric
    metric columns are aggregated as ``{metric}_mean`` / ``{metric}_std``;
    non-numeric ID columns are excluded.  Accepts dicts with native numeric
    values or string values read back from a CSV (``load_csv_rows``).
    """
    # Columns to exclude from the metric summary.
    exclude = {bin_col, "map_id", "wn", "attempt", "seed", "map_config", "pickle_idx", "scale"}

    # Metric columns: numeric-valued columns in first-appearance order.
    metric_cols: list[str] = []
    for m in per_map:
        for k, v in m.items():
            if k in exclude or k in metric_cols:
                continue
            if isinstance(v, (int, float, np.integer, np.floating)) or (
                isinstance(v, str) and v.replace(".", "", 1).lstrip("-").isdigit()
            ):
                metric_cols.append(k)

    # Group per-map rows by bin.
    groups: dict = defaultdict(list)
    for m in per_map:
        groups[m.get(bin_col)].append(m)

    def _sort_key(b):
        try:
            return (0, float(b))
        except (TypeError, ValueError):
            return (1, str(b))

    header = (
        [bin_col, "n_maps"] + [f"{c}_mean" for c in metric_cols] + [f"{c}_std" for c in metric_cols]
    )
    rows = []
    for b in sorted(groups, key=_sort_key):
        g = groups[b]
        rec = {bin_col: b, "n_maps": len(g)}
        for c in metric_cols:
            vals = []
            for m in g:
                v = m.get(c)
                if v is None or v == "":
                    continue
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    continue
            rec[f"{c}_mean"] = float(np.mean(vals)) if vals else ""
            rec[f"{c}_std"] = float(np.std(vals)) if vals else ""
        rows.append(rec)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{name}_by_bin.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for rec in rows:
            w.writerow([rec.get(c, "") for c in header])


def load_csv_keys(csv_path: Path, key_cols: list[str]) -> set[str]:
    """Load existing CSV and return set of ``'|'.join(key_cols)`` seen keys.

    Used for resume: before generating a map, check if
    ``f"{row[col1]}|{row[col2]}"`` is already in the CSV.  If yes, skip it.
    """
    if not csv_path.exists():
        return set()
    seen: set[str] = set()
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parts = [str(row.get(c, "")).strip() for c in key_cols]
                seen.add("|".join(parts))
    except Exception:
        pass
    return seen


def append_csv_row(csv_path: Path, columns: list[str], row: dict):
    """Append a single row to CSV.  Writes header only if file is new.

    Args:
        csv_path: Path to the CSV file (append mode).
        columns: Ordered list of column names.
        row: Dict of column → value for this row.
    """
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(columns)
        w.writerow([row.get(c, "") for c in columns])


def save_system_info(output_dir: Path, label: str, init_stats: dict, peak_stats: dict, n_maps: int):
    """Save system resource info to ``system.csv``."""
    out = output_dir / "system.csv"
    exists = out.exists()
    with open(out, "a" if exists else "w", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(
                [
                    "baseline",
                    "n_maps",
                    "cpu_init",
                    "cpu_peak",
                    "mem_init_mb",
                    "mem_peak_mb",
                    "gpu_init_mb",
                    "gpu_peak_mb",
                ]
            )
        w.writerow(
            [
                label,
                n_maps,
                init_stats.get("cpu_percent", ""),
                peak_stats.get("cpu_percent", ""),
                init_stats.get("mem_mb", ""),
                peak_stats.get("mem_mb", ""),
                init_stats.get("gpu_mem_mb", ""),
                peak_stats.get("gpu_mem_mb", ""),
            ]
        )

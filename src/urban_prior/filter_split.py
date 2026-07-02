"""
Urban Structural Prior Dataset — Filtering and Train/Val/Test Split.

Reads the flat Parquet from build_dataset.py, applies quality filters,
computes per-feature percentile thresholds, and writes stratified Parquet splits.

Outputs (under output-dir):
    train.parquet              — 80% stratified
    val.parquet                — 10% stratified
    test.parquet               — 10% stratified
    split_summary.json         — counts, pattern distribution, filter reasons

Usage:
    python -m src.urban_prior.filter_split \
        --parquet data/urban_prior/urban_prior.parquet \
        --output-dir data/urban_prior/splits \
        --seed 42
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

from src.skeleton_generator.utils import PATTERN_NAMES
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
NUMERIC_FILTER_COLS = [
    "road_density_km_per_km2", "block_count",
    "skeleton_node_count", "skeleton_edge_count",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Filter and split Urban Prior Dataset")
    parser.add_argument("--parquet", default="data/urban_prior/urban_prior.parquet", help="Flat Parquet path")
    parser.add_argument("--output-dir", default="data/urban_prior/splits", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args(argv)


def _compute_percentile_mask(df, col, low=1.0, high=99.0):
    vals = df[col].dropna()
    if len(vals) < 100:
        return np.ones(len(df), dtype=bool)
    p_low, p_high = np.percentile(vals, low), np.percentile(vals, high)
    return (df[col] >= p_low) & (df[col] <= p_high)


def _stratified_split(df, stratify_col, ratios, seed):
    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True)

    order = np.zeros(len(df), dtype=int)
    for _, grp_idx in df.groupby(stratify_col, group_keys=False).groups.items():
        perm = rng.permutation(len(grp_idx))
        order[list(grp_idx)] = perm

    df = df.copy()
    df["_order"] = order

    splits = {name: [] for name in ratios}

    for grp_name, grp_df in df.groupby(stratify_col):
        grp_sorted = grp_df.sort_values("_order")
        n = len(grp_sorted)
        n_train = int(n * ratios["train"])
        n_val = int(n * ratios["val"])
        splits["train"].append(grp_sorted.iloc[:n_train])
        splits["val"].append(grp_sorted.iloc[n_train:n_train + n_val])
        splits["test"].append(grp_sorted.iloc[n_train + n_val:])

    return {name: pd.concat(pieces, ignore_index=True).drop(columns=["_order"])
            for name, pieces in splits.items()}


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load data
    print(f"[INFO] Loading parquet: {args.parquet}")
    df = pd.read_parquet(args.parquet)
    n_total = len(df)
    print(f"[INFO] Total records: {n_total}")

    # 2. Hard quality filters
    filter_counts = Counter()
    mask_valid = df["quality_valid_graph"] == True
    dropped_valid = n_total - mask_valid.sum()
    filter_counts["quality_valid_graph_false"] = int(dropped_valid)
    print(f"[FILTER] quality_valid_graph == false: {dropped_valid}")

    mask_block = df["block_prior_available"] == True
    dropped_block = n_total - mask_block.sum()
    filter_counts["block_prior_unavailable"] = int(dropped_block)
    print(f"[FILTER] block_prior_available == false: {dropped_block}")

    mask_conf = df["confidence"] >= 0.4
    dropped_conf = n_total - mask_conf.sum()
    filter_counts["confidence_below_0.4"] = int(dropped_conf)
    print(f"[FILTER] confidence < 0.4: {dropped_conf}")

    hard_mask = mask_valid & mask_block & mask_conf
    df_hard = df[hard_mask].copy()
    print(f"[INFO] After hard filters: {len(df_hard)} / {n_total}")

    # 3. Percentile-based filters
    filter_mask = pd.Series(True, index=df_hard.index)
    for col in NUMERIC_FILTER_COLS:
        vals = df_hard[col].dropna()
        if len(vals) < 100:
            continue
        p_low, p_high = np.percentile(vals, 1), np.percentile(vals, 99)
        col_mask = (df_hard[col] >= p_low) & (df_hard[col] <= p_high)
        n_dropped = (~col_mask).sum()
        filter_counts[f"{col}_outside_1_99_pct"] = int(n_dropped)
        print(f"[FILTER] {col} outside [p1, p99] = [{p_low:.4f}, {p_high:.4f}]: {n_dropped}")
        filter_mask = filter_mask & col_mask

    df_clean = df_hard[filter_mask].copy()
    n_clean = len(df_clean)
    print(f"[INFO] Clean records: {n_clean}")

    # 4. Stratified split
    splits = _stratified_split(df_clean, "top_pattern_name", SPLIT_RATIOS, args.seed)

    for name, split_df in splits.items():
        actual_ratio = len(split_df) / n_clean
        print(f"[SPLIT] {name}: {len(split_df)} ({actual_ratio:.1%}) vs expected {SPLIT_RATIOS[name]:.0%}")

    # 5. Write split Parquets
    for name in ["train", "val", "test"]:
        path = os.path.join(args.output_dir, f"{name}.parquet")
        splits[name].to_parquet(path, index=False)
        print(f"[OUTPUT] {path} — {len(splits[name])} records")

    # 6. Write summary JSON
    pattern_dist = {}
    for name in ["train", "val", "test"]:
        counts = splits[name]["top_pattern_name"].value_counts()
        pattern_dist[name] = {str(k): int(v) for k, v in counts.items()}

    summary = {
        "seed": args.seed,
        "total_original": int(n_total),
        "total_clean": int(n_clean),
        "total_filtered": int(n_total - n_clean),
        "filter_reasons": {k: int(v) for k, v in sorted(filter_counts.items())},
        "splits": {name: {"count": int(len(split_df)),
                          "ratio": round(len(split_df) / n_clean, 4) if n_clean > 0 else 0.0}
                   for name, split_df in splits.items()},
        "pattern_distribution_per_split": pattern_dist,
        "overall_pattern_distribution": {
            str(k): int(v) for k, v in df_clean["top_pattern_name"].value_counts().items()
        },
    }

    summary_path = os.path.join(args.output_dir, "split_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OUTPUT] {summary_path}")

    # Balance check
    print()
    print("Pattern distribution per split:")
    print(f"{'Pattern':<20} {'Overall':>10} {'Train':>10} {'Val':>10} {'Test':>10}")
    print("-" * 60)
    overall_counts = df_clean["top_pattern_name"].value_counts()
    for p in PATTERN_NAMES:
        print(f"{p:<20} {overall_counts.get(p, 0):>10} "
              f"{splits['train']['top_pattern_name'].value_counts().get(p, 0):>10} "
              f"{splits['val']['top_pattern_name'].value_counts().get(p, 0):>10} "
              f"{splits['test']['top_pattern_name'].value_counts().get(p, 0):>10}")

    print()
    print("=" * 55)
    print("  FILTER & SPLIT COMPLETE")
    print("=" * 55)
    print(f"  Original: {n_total}")
    print(f"  Clean:    {n_clean}")
    print(f"  Filtered: {n_total - n_clean}")
    print(f"  Train:    {len(splits['train'])}")
    print(f"  Val:      {len(splits['val'])}")
    print(f"  Test:     {len(splits['test'])}")


if __name__ == "__main__":
    main()

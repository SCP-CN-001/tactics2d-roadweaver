"""Urban prior dataset filtering and splitting."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
NUMERIC_FILTER_COLS = [
    "road_density_km_per_km2",
    "block_count",
    "skeleton_node_count",
    "skeleton_edge_count",
]


def parse_args(argv=None):
    """Parse command-line arguments for filtering."""
    parser = argparse.ArgumentParser(description="Filter and split Urban Prior Dataset")
    parser.add_argument("--parquet", default="data/urban_prior/urban_prior.parquet")
    parser.add_argument("--output-dir", default="data/urban_prior/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--style-predictions",
        type=str,
        default=None,
        help="Style predictions JSON (from predict.py)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum confidence for style filter (default: 0.7)",
    )
    return parser.parse_args(argv)


def _extract_patch_id(path: str) -> str:
    """Extract patch ID from a file path like 'Abidjan_0.png' or '/path/to/Abidjan_0.png'."""
    return os.path.splitext(os.path.basename(path))[0]


def main():
    """Filter and split the urban prior dataset."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] Loading parquet: {args.parquet}")
    df = pd.read_parquet(args.parquet)
    n_total = len(df)
    print(f"[INFO] Total records: {n_total}")

    filter_counts = Counter()

    # 1. Hard quality filters
    mask_valid = df["quality_valid_graph"] == True
    dropped_valid = n_total - mask_valid.sum()
    filter_counts["quality_valid_graph_false"] = int(dropped_valid)
    print(f"[FILTER] quality_valid_graph == false: {dropped_valid}")

    mask_block = df["block_prior_available"] == True
    dropped_block = n_total - mask_block.sum()
    filter_counts["block_prior_unavailable"] = int(dropped_block)
    print(f"[FILTER] block_prior_available == false: {dropped_block}")

    hard_mask = mask_valid & mask_block
    df_hard = df[hard_mask].copy()
    print(f"[INFO] After hard filters: {len(df_hard)} / {n_total}")

    # 2. Style predictions merge (optional)
    if args.style_predictions and os.path.isfile(args.style_predictions):
        print(f"[INFO] Loading style predictions: {args.style_predictions}")
        with open(args.style_predictions) as f:
            predictions = json.load(f)

        pred_df = pd.DataFrame(predictions)
        pred_df["patch_id"] = pred_df["image_path"].apply(_extract_patch_id)

        df_hard["patch_id"] = df_hard["crhd_image_path"].apply(_extract_patch_id)

        before = len(df_hard)
        df_hard = df_hard.merge(pred_df, on="patch_id", how="left", suffixes=("", "_pred"))
        print(
            f"[INFO] Merged style predictions: {len(df_hard)} records ({len(df_hard) - before} added)"
        )

        # Add style vector columns
        for i in range(6):
            df_hard[f"style_vector_{i}"] = df_hard["style_vector"].apply(
                lambda v: v[i] if isinstance(v, list) and len(v) > i else None
            )

        df_hard["confidence"] = df_hard["confidence"].fillna(0.0)
        df_hard["top_pattern_name"] = df_hard.get("top_pattern_name", "Unknown").fillna("Unknown")

        # Confidence filter
        mask_conf = df_hard["confidence"] >= args.confidence_threshold
        dropped_conf = (~mask_conf).sum()
        filter_counts[f"confidence_below_{args.confidence_threshold}"] = int(dropped_conf)
        print(f"[FILTER] confidence < {args.confidence_threshold}: {dropped_conf}")
        df_hard = df_hard[mask_conf].copy()

        # Drop intermediate merge columns
        drop_cols = [
            c
            for c in [
                "image_path",
                "style_vector",
                "style_dim",
                "top_pattern",
                "checkpoint_loaded",
                "patch_id",
            ]
            if c in df_hard.columns
        ]
        df_hard = df_hard.drop(columns=drop_cols, errors="ignore")

        print(f"[INFO] After style filter: {len(df_hard)} records")

    # 3. Percentile-based numeric filters
    filter_mask = pd.Series(True, index=df_hard.index)
    for col in NUMERIC_FILTER_COLS:
        if col not in df_hard.columns:
            continue
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

    if n_clean == 0:
        print("[WARN] No records left after filtering!")
        return

    # 4. Random split
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_clean)
    n_train = int(n_clean * SPLIT_RATIOS["train"])
    n_val = int(n_clean * SPLIT_RATIOS["val"])
    splits = {
        "train": df_clean.iloc[perm[:n_train]],
        "val": df_clean.iloc[perm[n_train : n_train + n_val]],
        "test": df_clean.iloc[perm[n_train + n_val :]],
    }

    for name, split_df in splits.items():
        actual_ratio = len(split_df) / n_clean
        print(
            f"[SPLIT] {name}: {len(split_df)} ({actual_ratio:.1%}) vs expected {SPLIT_RATIOS[name]:.0%}"
        )

    # 5. Write split Parquets
    for name in ["train", "val", "test"]:
        path = os.path.join(args.output_dir, f"{name}.parquet")
        splits[name].to_parquet(path, index=False)
        print(f"[OUTPUT] {path} — {len(splits[name])} records")

    # 6. Summary
    summary = {
        "seed": args.seed,
        "confidence_threshold": args.confidence_threshold if args.style_predictions else None,
        "total_original": int(n_total),
        "total_clean": int(n_clean),
        "total_filtered": int(n_total - n_clean),
        "filter_reasons": {k: int(v) for k, v in sorted(filter_counts.items())},
        "splits": {
            name: {
                "count": int(len(split_df)),
                "ratio": round(len(split_df) / n_clean, 4) if n_clean > 0 else 0.0,
            }
            for name, split_df in splits.items()
        },
    }

    if "top_pattern_name" in df_clean.columns:
        summary["pattern_distribution"] = {
            str(k): int(v) for k, v in df_clean["top_pattern_name"].value_counts().items()
        }

    summary_path = os.path.join(args.output_dir, "split_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OUTPUT] {summary_path}")

    print("\n" + "=" * 55)
    print("  FILTER & SPLIT COMPLETE")
    print("=" * 55)
    print(f"  Original: {n_total}")
    print(f"  Clean:    {n_clean}")
    print(f"  Filtered: {n_total - n_clean}")
    print(f"  Train:    {len(splits['train'])}")
    print(f"  Val:      {len(splits['val'])}")
    print(f"  Test:     {len(splits['test'])}")
    if "top_pattern_name" in df_clean.columns:
        print("\n  Pattern distribution:")
        for p, c in df_clean["top_pattern_name"].value_counts().items():
            print(f"    {p}: {c}")


if __name__ == "__main__":
    main()

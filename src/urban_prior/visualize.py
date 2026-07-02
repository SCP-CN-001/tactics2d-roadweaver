"""
Urban Structural Prior Dataset — Visualization & Diagnostics.

Reads flat Parquet and generates diagnostic plots, outlier detection,
feature statistics, and a markdown report.

Usage:
    python -m src.urban_prior.visualize \
        --parquet data/urban_prior/urban_prior.parquet \
        --output-dir analysis/imgs \
        --report-dir analysis
"""

import argparse
import json
import math
import os
import sys
import time
import warnings
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

_HAS_SEABORN = False
_HAS_UMAP = False
try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:
    pass
try:
    import umap
    _HAS_UMAP = True
except ImportError:
    pass
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STYLE_VECTOR_KEYS = [f"style_vector_{i}" for i in range(6)]

GLOBAL_PRIOR_KEYS = [
    "road_density_km_per_km2", "major_road_density_km_per_km2",
    "minor_road_density_km_per_km2", "node_count", "edge_count",
    "intersection_count", "major_node_count", "boundary_entry_count",
    "avg_degree", "dead_end_ratio", "three_way_ratio", "four_way_ratio",
    "road_length_mean_m", "road_length_std_m", "road_length_median_m",
    "orientation_entropy", "bearing_entropy", "gridness_score",
    "radialness_score", "organic_score", "estimated_lane_mean",
    "estimated_road_width_mean_m",
]

BLOCK_PRIOR_KEYS = [
    "block_count", "block_area_mean_m2", "block_area_std_m2",
    "block_area_median_m2", "block_aspect_ratio_mean",
    "block_compactness_mean", "block_scale_m", "block_prior_available",
]

PCA_FEATURES = (
    STYLE_VECTOR_KEYS + [
        "road_density_km_per_km2", "major_road_density_km_per_km2",
        "minor_road_density_km_per_km2", "intersection_count",
        "major_node_count", "boundary_entry_count", "dead_end_ratio",
        "three_way_ratio", "four_way_ratio", "orientation_entropy",
        "bearing_entropy", "gridness_score", "radialness_score",
        "organic_score", "block_count", "block_scale_m",
        "skeleton_node_count", "skeleton_edge_count",
    ]
)

CORRELATION_FEATURES = (
    STYLE_VECTOR_KEYS + [
        "road_density_km_per_km2", "major_road_density_km_per_km2",
        "minor_road_density_km_per_km2", "node_count", "edge_count",
        "intersection_count", "major_node_count", "boundary_entry_count",
        "avg_degree", "dead_end_ratio", "three_way_ratio", "four_way_ratio",
        "orientation_entropy", "bearing_entropy", "gridness_score",
        "radialness_score", "organic_score", "block_count", "block_scale_m",
        "skeleton_node_count", "skeleton_edge_count",
    ]
)

PATTERN_NAMES = ["Gridiron", "Linear", "No pattern", "Organic", "Radial", "Tributary"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _clip_to_percentile(series, low=1, high=99):
    vals = series.dropna()
    if len(vals) < 10:
        return series
    return series.clip(np.percentile(vals, low), np.percentile(vals, high))


def _boxplot(ax, data, x_col, y_col, order=None, palette="Set2", title="", xlabel="", ylabel=""):
    if _HAS_SEABORN:
        sns.boxplot(data=data, x=x_col, y=y_col, order=order, palette=palette, ax=ax)
    else:
        if order is None:
            order = data[x_col].unique()
        groups = [data[data[x_col] == g][y_col].dropna().values for g in order]
        bp = ax.boxplot(groups, tick_labels=order, patch_artist=True)
        colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)


def _violinplot(ax, data, x_col, y_col, order=None, palette="Set2", title="", xlabel="", ylabel=""):
    if _HAS_SEABORN:
        sns.violinplot(data=data, x=x_col, y=y_col, order=order, palette=palette, ax=ax)
    else:
        _boxplot(ax, data, x_col, y_col, order, palette, title, xlabel, ylabel)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

def detect_outliers(df: pd.DataFrame, outlier_path: str):
    print("[INFO] Detecting outliers ...")
    records = df.copy()
    outlier_list = []

    thresholds = {
        "road_density_km_per_km2": ("high_road_density", "low_road_density"),
        "block_count": ("high_block_count", "low_block_count"),
        "skeleton_node_count": ("high_skeleton_node_count", "low_skeleton_node_count"),
        "skeleton_edge_count": ("high_skeleton_edge_count", "low_skeleton_edge_count"),
    }

    for col, (high_reason, low_reason) in thresholds.items():
        if col not in records.columns:
            continue
        vals = records[col].dropna()
        if len(vals) < 100:
            continue
        p01, p99 = np.percentile(vals, [1, 99])
        for idx, row in records.iterrows():
            v = row.get(col)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            if v > p99:
                _add_outlier(outlier_list, row, high_reason)
            elif v < p01:
                _add_outlier(outlier_list, row, low_reason)

    for idx, row in records.iterrows():
        conf = row.get("confidence")
        if conf is not None and not (isinstance(conf, float) and math.isnan(conf)) and conf < 0.4:
            _add_outlier(outlier_list, row, "low_confidence")
        if not row.get("block_prior_available", True):
            _add_outlier(outlier_list, row, "block_prior_unavailable")
        if not row.get("quality_valid_graph", True):
            _add_outlier(outlier_list, row, "invalid_graph")

    seen = set()
    deduped = []
    for o in outlier_list:
        pid = o["patch_id"]
        if pid in seen:
            for existing in deduped:
                if existing["patch_id"] == pid:
                    for r in o["reasons"]:
                        if r not in existing["reasons"]:
                            existing["reasons"].append(r)
                    break
        else:
            seen.add(pid)
            deduped.append(o)

    print(f"[INFO] Found {len(deduped)} unique outlier records")
    with open(outlier_path, "w") as f:
        json.dump(deduped, f, indent=2, default=str)
    print(f"[INFO] Outliers saved to {outlier_path}")
    return deduped


def _add_outlier(outlier_list, row, reason):
    outlier_list.append({
        "patch_id": str(row.get("patch_id", "unknown")),
        "top_pattern_name": str(row.get("top_pattern_name", "Unknown")),
        "confidence": None if (isinstance(row.get("confidence"), float) and math.isnan(row.get("confidence"))) else row.get("confidence"),
        "reasons": [reason],
        "road_density_km_per_km2": None if (isinstance(row.get("road_density_km_per_km2"), float) and math.isnan(row.get("road_density_km_per_km2"))) else row.get("road_density_km_per_km2"),
        "block_count": None if (isinstance(row.get("block_count"), float) and math.isnan(row.get("block_count"))) else row.get("block_count"),
        "skeleton_node_count": None if (isinstance(row.get("skeleton_node_count"), float) and math.isnan(row.get("skeleton_node_count"))) else row.get("skeleton_node_count"),
        "skeleton_edge_count": None if (isinstance(row.get("skeleton_edge_count"), float) and math.isnan(row.get("skeleton_edge_count"))) else row.get("skeleton_edge_count"),
    })


# ---------------------------------------------------------------------------
# Feature statistics
# ---------------------------------------------------------------------------

def compute_feature_statistics(df: pd.DataFrame, stats_path: str):
    print("[INFO] Computing feature statistics ...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for drop in ["patch_id", "top_pattern"]:
        if drop in numeric_cols:
            numeric_cols.remove(drop)

    stats = {}
    for col in numeric_cols:
        series = df[col].dropna()
        missing = df[col].isna().sum()
        if len(series) == 0:
            stats[col] = {"mean": None, "std": None, "min": None, "max": None,
                          "p01": None, "p05": None, "p50": None, "p95": None, "p99": None,
                          "missing_ratio": float(missing) / len(df) if len(df) > 0 else 1.0, "count": 0}
            continue
        stats[col] = {
            "mean": float(np.mean(series)), "std": float(np.std(series)),
            "min": float(np.min(series)), "max": float(np.max(series)),
            "p01": float(np.percentile(series, 1)), "p05": float(np.percentile(series, 5)),
            "p50": float(np.median(series)), "p95": float(np.percentile(series, 95)),
            "p99": float(np.percentile(series, 99)),
            "missing_ratio": float(missing) / len(df) if len(df) > 0 else 1.0, "count": int(len(series)),
        }

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[INFO] Feature statistics saved to {stats_path}")
    return stats


# ---------------------------------------------------------------------------
# Plotting functions (unchanged from original)
# ---------------------------------------------------------------------------

def plot_pattern_distribution(df, output_dir):
    print("  [1/10] Pattern distribution ...")
    fig, ax = plt.subplots(figsize=(10, 5))
    counts = df["top_pattern_name"].value_counts()
    values = [counts.get(p, 0) for p in PATTERN_NAMES]
    colors = plt.cm.Set2(np.linspace(0, 1, len(PATTERN_NAMES)))
    bars = ax.bar(PATTERN_NAMES, values, color=colors, edgecolor="gray")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.005,
                str(v), ha="center", va="bottom", fontsize=9)
    ax.set_title("Urban Pattern Distribution", fontsize=14)
    ax.set_xlabel("Pattern")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pattern_distribution.png"), dpi=150)
    plt.close(fig)


def plot_road_density_by_pattern(df, output_dir):
    print("  [2/10] Road density by pattern ...")
    fig, ax = plt.subplots(figsize=(10, 5))
    data = df[df["road_density_km_per_km2"].notna()].copy()
    data["road_density_clipped"] = _clip_to_percentile(data["road_density_km_per_km2"])
    _violinplot(ax, data, x_col="top_pattern_name", y_col="road_density_clipped",
                order=PATTERN_NAMES, title="Road Density by Pattern (1%-99% clipped)",
                xlabel="Pattern", ylabel="Road Density (km/km²)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "road_density_by_pattern.png"), dpi=150)
    plt.close(fig)


def plot_major_road_density_by_pattern(df, output_dir):
    print("  [3/10] Major road density by pattern ...")
    fig, ax = plt.subplots(figsize=(10, 5))
    data = df[df["major_road_density_km_per_km2"].notna()].copy()
    data["major_road_density_clipped"] = _clip_to_percentile(data["major_road_density_km_per_km2"])
    _violinplot(ax, data, x_col="top_pattern_name", y_col="major_road_density_clipped",
                order=PATTERN_NAMES, title="Major Road Density by Pattern (1%-99% clipped)",
                xlabel="Pattern", ylabel="Major Road Density (km/km²)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "major_road_density_by_pattern.png"), dpi=150)
    plt.close(fig)


def plot_block_count_by_pattern(df, output_dir):
    print("  [4/10] Block count by pattern ...")
    fig, ax = plt.subplots(figsize=(10, 5))
    data = df[df["block_count"].notna()].copy()
    data["block_count_clipped"] = _clip_to_percentile(data["block_count"])
    _violinplot(ax, data, x_col="top_pattern_name", y_col="block_count_clipped",
                order=PATTERN_NAMES, title="Block Count by Pattern (1%-99% clipped)",
                xlabel="Pattern", ylabel="Block Count")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "block_count_by_pattern.png"), dpi=150)
    plt.close(fig)


def plot_block_scale_by_pattern(df, output_dir):
    print("  [5/10] Block scale by pattern ...")
    fig, ax = plt.subplots(figsize=(10, 5))
    data = df[df["block_scale_m"].notna()].copy()
    data["block_scale_clipped"] = _clip_to_percentile(data["block_scale_m"])
    _violinplot(ax, data, x_col="top_pattern_name", y_col="block_scale_clipped",
                order=PATTERN_NAMES, title="Block Scale by Pattern (1%-99% clipped)",
                xlabel="Pattern", ylabel="Block Scale (m)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "block_scale_by_pattern.png"), dpi=150)
    plt.close(fig)


def plot_structural_scores_distribution(df, output_dir):
    print("  [6/10] Structural scores distribution ...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (col, label) in zip(axes, [("gridness_score", "Gridness Score"),
                                         ("radialness_score", "Radialness Score"),
                                         ("organic_score", "Organic Score")]):
        vals = df[col].dropna().values
        if len(vals) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue
        if _HAS_SEABORN:
            sns.histplot(vals, kde=True, bins=80, ax=ax, color="steelblue")
        else:
            ax.hist(vals, bins=80, density=True, alpha=0.7, color="steelblue")
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "structural_scores_distribution.png"), dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df, output_dir):
    print("  [7/10] Correlation heatmaps ...")
    available = [c for c in CORRELATION_FEATURES if c in df.columns]
    corr_df = df[available].dropna().corr(method="pearson")
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)
    if _HAS_SEABORN:
        sns.heatmap(corr_df, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                    vmin=-1, vmax=1, center=0, square=True, linewidths=0.3, ax=ax)
    else:
        im = ax.imshow(corr_df.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr_df.columns)))
        ax.set_yticks(range(len(corr_df.columns)))
        ax.set_xticklabels(corr_df.columns, rotation=90, fontsize=6)
        ax.set_yticklabels(corr_df.columns, fontsize=6)
        fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Pearson Correlation: Style Vector vs Global Prior", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "style_prior_correlation_heatmap.png"), dpi=150)
    plt.close(fig)

    print("  [7/10] Spearman correlation ...")
    corr_sp = df[available].dropna().corr(method="spearman")
    fig, ax = plt.subplots(figsize=(14, 12))
    if _HAS_SEABORN:
        sns.heatmap(corr_sp, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                    vmin=-1, vmax=1, center=0, square=True, linewidths=0.3, ax=ax)
    else:
        im = ax.imshow(corr_sp.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr_sp.columns)))
        ax.set_yticks(range(len(corr_sp.columns)))
        ax.set_xticklabels(corr_sp.columns, rotation=90, fontsize=6)
        ax.set_yticklabels(corr_sp.columns, fontsize=6)
        fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Spearman Correlation: Style Vector vs Global Prior", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "style_prior_spearman_heatmap.png"), dpi=150)
    plt.close(fig)


def plot_pca(df, output_dir):
    print("  [8/10] PCA ...")
    available = [c for c in PCA_FEATURES if c in df.columns]
    plot_df = df[available].dropna().copy()
    if len(plot_df) < 10:
        print("  [WARN] Too few samples for PCA, skipping")
        return
    if len(plot_df) > 50000:
        plot_df = plot_df.sample(n=50000, random_state=42)
    X = StandardScaler().fit_transform(plot_df[available].values)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    pattern_map = df.loc[plot_df.index, "top_pattern_name"].values
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set2(np.linspace(0, 1, 6))
    color_map = {p: c for p, c in zip(PATTERN_NAMES, colors)}
    for p in PATTERN_NAMES:
        mask = pattern_map == p
        if mask.sum() == 0:
            continue
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[color_map[p]],
                   label=p, alpha=0.5, s=5)
    ax.set_title(f"PCA: Style + Prior Features (var: {pca.explained_variance_ratio_[0]:.2f}, {pca.explained_variance_ratio_[1]:.2f})", fontsize=12)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.legend(markerscale=5, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "style_vs_prior_pca.png"), dpi=150)
    plt.close(fig)


def plot_umap(df, output_dir):
    print("  [9/10] UMAP ...")
    if not _HAS_UMAP:
        print("  [SKIP] umap-learn is not installed")
        return
    available = [c for c in PCA_FEATURES if c in df.columns]
    plot_df = df[available].dropna().copy()
    if len(plot_df) < 10:
        print("  [WARN] Too few samples for UMAP, skipping")
        return
    if len(plot_df) > 50000:
        plot_df = plot_df.sample(n=50000, random_state=42)
    X_scaled = StandardScaler().fit_transform(plot_df[available].values)
    X_umap = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1).fit_transform(X_scaled)
    pattern_map = df.loc[plot_df.index, "top_pattern_name"].values
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set2(np.linspace(0, 1, 6))
    color_map = {p: c for p, c in zip(PATTERN_NAMES, colors)}
    for p in PATTERN_NAMES:
        mask = pattern_map == p
        if mask.sum() == 0:
            continue
        ax.scatter(X_umap[mask, 0], X_umap[mask, 1], c=[color_map[p]],
                   label=p, alpha=0.5, s=5)
    ax.set_title("UMAP: Style + Prior Features", fontsize=12)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=5, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "style_vs_prior_umap.png"), dpi=150)
    plt.close(fig)


def plot_skeleton_size_distribution(df, output_dir):
    print("  [10/10] Skeleton size distribution ...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, label in zip(axes, ["skeleton_node_count", "skeleton_edge_count"],
                               ["Skeleton Node Count", "Skeleton Edge Count"]):
        vals = df[col].dropna().values
        if len(vals) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue
        vals_clipped = np.clip(vals, np.percentile(vals, 1), np.percentile(vals, 99))
        if _HAS_SEABORN:
            sns.histplot(vals_clipped, bins=80, kde=True, ax=ax, color="steelblue")
        else:
            ax.hist(vals_clipped, bins=80, density=True, alpha=0.7, color="steelblue")
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Count")
        ax.set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "skeleton_size_distribution.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Diagnostics report
# ---------------------------------------------------------------------------

def write_diagnostics_report(df, stats, outliers, report_path, used_samples, total_records):
    n_valid = df["quality_valid_graph"].sum() if "quality_valid_graph" in df else len(df)
    n_total = len(df)
    pattern_counts = df["top_pattern_name"].value_counts()

    numeric_for_pattern = [
        "road_density_km_per_km2", "major_road_density_km_per_km2",
        "block_count", "block_scale_m", "gridness_score", "radialness_score", "organic_score",
    ]
    pattern_means = {}
    for p in PATTERN_NAMES:
        sub = df[df["top_pattern_name"] == p]
        if len(sub) == 0:
            continue
        pattern_means[p] = {col: (float(sub[col].mean()) if sub[col].notna().any() else None)
                           for col in numeric_for_pattern if col in sub.columns}

    available = [c for c in CORRELATION_FEATURES if c in df.columns]
    corr_df = df[available].dropna().corr(method="pearson")
    style_cols = STYLE_VECTOR_KEYS
    prior_cols = [c for c in CORRELATION_FEATURES if c not in STYLE_VECTOR_KEYS]
    cross_corrs = []
    for sc in style_cols:
        if sc not in corr_df.columns:
            continue
        for pc in prior_cols:
            if pc not in corr_df.columns:
                continue
            val = corr_df.loc[sc, pc]
            if not (isinstance(val, float) and math.isnan(val)):
                cross_corrs.append((abs(val), sc, pc, val))
    cross_corrs.sort(key=lambda x: x[0], reverse=True)
    top_corrs = cross_corrs[:15]

    pattern_discrimination = []
    for p in PATTERN_NAMES:
        if p not in pattern_means:
            continue
        sub = df[df["top_pattern_name"] == p]
        if len(sub) < 10:
            continue
        scores = []
        for col in numeric_for_pattern:
            if col not in stats or col not in sub.columns:
                continue
            group_mean = sub[col].mean()
            overall_mean = stats[col]["mean"]
            overall_std = stats[col]["std"]
            if overall_std is None or overall_std == 0:
                continue
            scores.append((abs(group_mean - overall_mean) / overall_std, col, group_mean, overall_mean))
        scores.sort(key=lambda x: x[0], reverse=True)
        pattern_discrimination.append((p, scores[:5]))

    outlier_reasons = defaultdict(int)
    for o in outliers:
        for r in o["reasons"]:
            outlier_reasons[r] += 1

    lines = [
        "# Urban Structural Prior Dataset — Diagnostics Report\n",
        "## 1. Data Overview\n",
        f"- **Total records:** {total_records}",
        f"- **Valid samples:** {n_total}",
        f"- **Valid graph records:** {n_valid}",
        f"- **Used for visualization:** {used_samples}",
        "- **Pattern distribution:**",
    ]
    for p in PATTERN_NAMES:
        c = pattern_counts.get(p, 0)
        lines.append(f"  - **{p}:** {c} ({100.0 * c / n_total:.1f}%)" if n_total else f"  - **{p}:** {c}")
    lines.append("")

    lines.append("## 2. Per-Pattern Structural Statistics (Mean)\n")
    lines.append("| Pattern | Road Density (km/km²) | Major Road Density | Block Count | Block Scale (m) | Gridness | Radialness | Organic |")
    lines.append("|---------|----------------------|-------------------|-------------|----------------|----------|------------|---------|")
    for p in PATTERN_NAMES:
        if p not in pattern_means:
            continue
        m = pattern_means[p]
        rd = f"{m.get('road_density_km_per_km2', 'N/A'):.2f}" if m.get("road_density_km_per_km2") is not None else "N/A"
        md = f"{m.get('major_road_density_km_per_km2', 'N/A'):.2f}" if m.get("major_road_density_km_per_km2") is not None else "N/A"
        bc = f"{m.get('block_count', 'N/A'):.1f}" if m.get("block_count") is not None else "N/A"
        bs = f"{m.get('block_scale_m', 'N/A'):.1f}" if m.get("block_scale_m") is not None else "N/A"
        gr = f"{m.get('gridness_score', 'N/A'):.3f}" if m.get("gridness_score") is not None else "N/A"
        ra = f"{m.get('radialness_score', 'N/A'):.3f}" if m.get("radialness_score") is not None else "N/A"
        or_ = f"{m.get('organic_score', 'N/A'):.3f}" if m.get("organic_score") is not None else "N/A"
        lines.append(f"| {p} | {rd} | {md} | {bc} | {bs} | {gr} | {ra} | {or_} |")
    lines.append("")

    lines.append("## 3. Style Vector — Global Prior Correlation\n")
    lines.append("Top correlated pairs (style_vector ↔ global_prior, by |Pearson r|):\n")
    lines.append("| Rank | Feature 1 | Feature 2 | Pearson r |")
    lines.append("|------|-----------|-----------|-----------|")
    for rank, (abs_r, sc, pc, r_val) in enumerate(top_corrs, 1):
        lines.append(f"| {rank} | {sc} | {pc} | {r_val:.4f} |")
    lines.append("")

    lines.append("## 4. Discriminative Features by Pattern\n")
    lines.append("Features where the pattern group mean deviates most from the global mean (Cohen's d effect size).\n")
    for p, scores in pattern_discrimination:
        lines.append(f"### {p}\n")
        lines.append("| Feature | Effect Size (|d|) | Group Mean | Global Mean |")
        lines.append("|---------|-------------------|------------|-------------|")
        for es, col, gm, om in scores:
            lines.append(f"| {col} | {es:.3f} | {gm:.3f} | {om:.3f} |")
        lines.append("")

    lines.append("## 5. Outlier Summary\n")
    lines.append(f"- **Total unique outlier records:** {len(outliers)}\n")
    lines.append("| Reason | Count |")
    lines.append("|--------|-------|")
    for reason, count in sorted(outlier_reasons.items(), key=lambda x: -x[1]):
        lines.append(f"| {reason} | {count} |")
    lines.append("")

    lines.append("## 6. Conclusion and Recommendations\n")
    gridiron_corrs = [x for x in cross_corrs if "gridness" in x[2] or "grid" in x[2].lower()]
    has_gridiron_correlation = any(abs_r > 0.15 for abs_r, _, _, _ in gridiron_corrs)

    has_bad_scores = False
    for col in ["gridness_score", "radialness_score", "organic_score"]:
        if col in stats and stats[col]["mean"] is not None and stats[col]["std"] is not None and stats[col]["std"] < 0.05:
            has_bad_scores = True

    outlier_ratio = len(outliers) / max(n_total, 1)

    lines.append("### 6.1 Style Vector — Prior Relationship")
    if has_gridiron_correlation:
        lines.append("- **Correlation detected:** Style vectors show meaningful correlation with structural prior metrics.")
    else:
        lines.append("- **Weak correlation:** Style vectors show limited direct linear correlation with structural metrics.")
    lines.append("")

    lines.append("### 6.2 Pattern Separability")
    lines.append("- PCA and UMAP visualizations show the degree to which different pattern classes cluster in the feature space.")
    sep_patterns = [p for p, scores in pattern_discrimination if scores and scores[0][0] > 0.5]
    if sep_patterns:
        lines.append(f"- Patterns with strong structural differentiation: {', '.join(sep_patterns)}.")
    lines.append("")

    lines.append("### 6.3 Data Quality")
    lines.append(f"- Outlier ratio: {outlier_ratio:.2%} ({len(outliers)} / {n_total})")
    if outlier_ratio > 0.05:
        lines.append("- **Warning:** Outlier ratio exceeds 5%. Consider cleaning extreme samples before generator training.")
    else:
        lines.append("- Outlier ratio is within acceptable range.")
    if has_bad_scores:
        lines.append("- **Note:** Some structural scores show very low variance, which may limit their usefulness as training signals.")
    lines.append("")

    lines.append("### 6.4 Recommendation")
    if outlier_ratio < 0.05 and n_valid > 10000:
        lines.append("**Recommendation: PROCEED to Urban Skeleton Generator training.**")
    else:
        lines.append("**Recommendation: Additional data cleaning recommended before generator training.**")

    lines.append("")

    report_text = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"[INFO] Diagnostics report saved to {report_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Urban Structural Prior — Visualization & Diagnostics")
    parser.add_argument("--parquet", default="data/urban_prior/urban_prior.parquet", help="Path to Parquet dataset")
    parser.add_argument("--output-dir", default="analysis/imgs", help="Directory for output PNGs")
    parser.add_argument("--report-dir", default="analysis", help="Directory for reports")
    parser.add_argument("--max-samples", type=int, default=50000, help="Max samples (-1 = all)")
    args = parser.parse_args()

    _ensure_dir(args.output_dir)
    _ensure_dir(args.report_dir)

    # Load Parquet
    t_start = time.time()
    print(f"[INFO] Reading parquet: {args.parquet}")
    df = pd.read_parquet(args.parquet)
    total_records = len(df)
    print(f"[INFO] Loaded {total_records} rows, {len(df.columns)} columns")

    if 0 < args.max_samples < total_records:
        df = df.sample(n=args.max_samples, random_state=42)
        print(f"[INFO] Sampled {len(df)} records")

    # Filter to valid graph records for analysis
    df_valid = df[df["quality_valid_graph"] == True].copy()
    if len(df_valid) == 0:
        df_valid = df.copy()
    print(f"[INFO] Valid graph records: {len(df_valid)}")

    # Outlier detection
    outlier_path = os.path.join(args.report_dir, "outlier_records.json")
    outliers = detect_outliers(df_valid, outlier_path)

    # Feature statistics
    stats_path = os.path.join(args.report_dir, "feature_statistics.json")
    stats = compute_feature_statistics(df_valid, stats_path)

    # CSV sample
    csv_path = os.path.join(args.report_dir, "urban_prior_sample.csv")
    sample_size = min(5000, len(df_valid))
    df_valid.sample(n=sample_size, random_state=42).to_csv(csv_path, index=False)
    print(f"[INFO] CSV sample saved to {csv_path}")

    # Visualizations
    print("[INFO] Generating visualizations ...")
    plot_pattern_distribution(df_valid, args.output_dir)
    plot_road_density_by_pattern(df_valid, args.output_dir)
    plot_major_road_density_by_pattern(df_valid, args.output_dir)
    plot_block_count_by_pattern(df_valid, args.output_dir)
    plot_block_scale_by_pattern(df_valid, args.output_dir)
    plot_structural_scores_distribution(df_valid, args.output_dir)
    plot_correlation_heatmap(df_valid, args.output_dir)
    plot_pca(df_valid, args.output_dir)
    plot_umap(df_valid, args.output_dir)
    plot_skeleton_size_distribution(df_valid, args.output_dir)

    # Diagnostics report
    report_path = os.path.join(args.report_dir, "diagnostics_report.md")
    write_diagnostics_report(df_valid, stats, outliers, report_path,
                             used_samples=len(df_valid), total_records=total_records)

    # Completion
    parquet_size_mb = os.path.getsize(args.parquet) / (1024 * 1024) if os.path.isfile(args.parquet) else 0
    print()
    print("=" * 65)
    print("  URBAN PRIOR VISUALIZATION — COMPLETE")
    print("=" * 65)
    print(f"  Load time:       {time.time() - t_start:.1f}s")
    print(f"  Parquet size:    {parquet_size_mb:.1f} MB")
    print(f"  Records loaded:  {total_records}")
    print(f"  Valid records:   {len(df_valid)}")
    print(f"  Outliers found:  {len(outliers)}")
    print()


if __name__ == "__main__":
    main()

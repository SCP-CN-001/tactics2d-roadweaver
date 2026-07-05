"""
Spatial Conditional Skeleton Generator — Visualization Utilities.

Supports plotting skeleton graphs from generated or ground-truth data,
comparison grids (GT | teacher-forcing | free-run), and bulk sample grids.
"""

import os
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NODE_COLORS = {
    0: "#4CAF50",  # Waypoint
    1: "#2196F3",  # Junction
    2: "#FF9800",  # Roundabout
    3: "#F44336",  # Cross
    4: "#9C27B0",  # Terminal
}

DEFAULT_NODE_COLOR = "#888888"


def _ensure_dir(path):
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────
# Single skeleton
# ──────────────────────────────────────────────────────────────────────────


def plot_skeleton(
    coords: np.ndarray,             # (N, 2)
    edge_index: np.ndarray,         # (2, E)
    node_types: np.ndarray = None,   # (N,) optional
    save_path: str = None,
    title: str = "",
    show_grid: bool = False,
    ax: plt.Axes = None,
    node_size: float = 15,
    edge_width: float = 0.8,
):
    """Plot a skeleton graph (single panel)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    N = coords.shape[0]

    # Draw edges (deduplicate)
    seen = set()
    if edge_index.shape[1] > 0:
        for e in range(edge_index.shape[1]):
            i, j = int(edge_index[0, e]), int(edge_index[1, e])
            if i >= N or j >= N or i == j:
                continue
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            ax.plot(
                [coords[i, 0], coords[j, 0]],
                [coords[i, 1], coords[j, 1]],
                color="#888888", linewidth=edge_width, alpha=0.5, zorder=1,
            )

    # Draw nodes
    for i in range(N):
        if node_types is not None and i < len(node_types):
            color = NODE_COLORS.get(int(node_types[i]), DEFAULT_NODE_COLOR)
        else:
            color = DEFAULT_NODE_COLOR
        ax.scatter(
            coords[i, 0], coords[i, 1],
            c=color, s=node_size, zorder=2, edgecolors="black", linewidths=0.3,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    if show_grid:
        ax.grid(True, alpha=0.3)

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


# (plot_skeleton_comparison removed — use plot_skeleton_grid instead)


# ──────────────────────────────────────────────────────────────────────────
# Comparison grid  (rows = samples, cols = GT | TF | FR)
# ──────────────────────────────────────────────────────────────────────────


def plot_skeleton_grid(
    samples: List[Dict],
    save_path: str,
    max_rows: int = 8,
):
    """
    Grid of sample rows, each row: GT | Teacher-Forcing | Free-Run.

    Each element of `samples` is a dict:
        {
            "gt": Dict,
            "teacher_forcing": Optional[Dict],
            "free_run": Optional[Dict],
            "title": str (optional),
        }
    """
    n = min(len(samples), max_rows)
    if n == 0:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        ax.text(0.5, 0.5, "No samples", ha="center", va="center", transform=ax.transAxes)
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    cols = 3  # GT, TF, FR
    fig, axes = plt.subplots(n, cols, figsize=(cols * 4, n * 4), squeeze=False)
    # axes is always (n, cols) with squeeze=False

    panel_labels = ["GT", "Teacher\nForcing", "Free\nRun"]

    for row_idx in range(n):
        sample = samples[row_idx]
        for col_idx, key in enumerate(["gt", "teacher_forcing", "free_run"]):
            ax = axes[row_idx, col_idx]
            graph = sample.get(key)

            if graph is None or graph["coords"].shape[0] == 0:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(panel_labels[col_idx], fontsize=8)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                continue

            coords = graph["coords"]
            edge_index = graph["edge_index"]
            node_types = graph.get("node_types", None)
            N = coords.shape[0]
            E = edge_index.shape[1]

            info = [f"N={N}", f"E={E}"]
            if "avg_degree" in graph:
                info.append(f"deg={graph['avg_degree']:.2f}")
            if "edge_f1" in graph:
                info.append(f"F1={graph['edge_f1']:.3f}")
            if "connected_components" in graph:
                info.append(f"CC={graph['connected_components']}")

            plot_skeleton(
                coords, edge_index, node_types,
                ax=ax, title=f"{panel_labels[col_idx]}\n{' '.join(info)}",
                node_size=10, edge_width=0.5,
            )

        # Row title
        sample_title = sample.get("title", f"Sample {row_idx}")
        axes[row_idx, 0].set_ylabel(sample_title, fontsize=7, rotation=0, labelpad=15, va="center")

    plt.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Generated framework grid  (all free-run samples, one row per sample)
# ──────────────────────────────────────────────────────────────────────────


def plot_generated_framework_grid(
    generated_samples: List[Dict],
    save_path: str,
    max_cols: int = 4,
):
    """
    Grid of free-run generated skeletons only.

    Each element of ``generated_samples`` is a dict:
        {
            "coords": np.ndarray (N, 2),
            "edge_index": np.ndarray (2, E),
            "node_types": np.ndarray (N,),
            "style_label": str,
            "free_run_score": float or None,
            "connected_components": int,
            "isolated_node_ratio": float,
        }
    """
    n = len(generated_samples)
    if n == 0:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        ax.text(0.5, 0.5, "No generated samples", ha="center", va="center", transform=ax.transAxes)
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return
    cols = min(max_cols, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False)
    # axes is always (rows, cols) with squeeze=False

    flat = axes.flatten()
    for idx in range(len(flat)):
        ax = flat[idx]

        if idx >= n:
            ax.set_visible(False)
            continue

        sample = generated_samples[idx]
        coords = sample.get("coords", np.zeros((0, 2)))
        edge_index = sample.get("edge_index", np.zeros((2, 0)))
        node_types = sample.get("node_types", None)
        N = coords.shape[0]
        E = edge_index.shape[1]

        parts = [sample.get("style_label", f"#{idx}"), f"N={N}", f"E={E}"]
        cc = sample.get("connected_components")
        if cc is not None:
            parts.append(f"CC={cc}")
        iso = sample.get("isolated_node_ratio")
        if iso is not None:
            parts.append(f"iso={iso:.2f}")
        frs = sample.get("free_run_score")
        if frs is not None:
            parts.append(f"score={frs:.3f}")

        plot_skeleton(
            coords, edge_index, node_types,
            ax=ax, title=" ".join(parts),
            node_size=8, edge_width=0.4,
        )

    plt.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

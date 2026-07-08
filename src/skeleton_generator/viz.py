"""
Visualization utilities for road skeleton graph generation.

Provides reusable plotting patterns:
  - field_grid:          N conditions × M seeds road field grid
  - field_to_graph_viz:  field vs extracted graph side-by-side
  - diversity_bars:      N/E counts per seed bar chart
"""

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NODE_COLORS = {
    0: "#4CAF50",  # Skeleton waypoint
    1: "#2196F3",  # Junction
    2: "#FF9800",  # Roundabout
    3: "#F44336",  # Major intersection
    4: "#9C27B0",  # Dead end
}


def field_grid(
    fields: Sequence[np.ndarray],
    titles: Sequence[str],
    n_cols: int = 4,
    figsize: Tuple[int, int] = (16, 10),
    suptitle: str = "",
    cmap: str = "gray_r",
) -> plt.Figure:
    """
    Plot a grid of road fields.

    Args:
        fields: list of (H, W) road field arrays
        titles: per-panel titles
        n_cols: number of columns
        figsize: figure dimensions
        suptitle: overall figure title
        cmap: colormap for imshow
    Returns:
        matplotlib Figure (not yet saved or closed)
    """
    n_rows = (len(fields) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

    for i in range(len(fields)):
        axes[i].imshow(fields[i], cmap=cmap, vmin=0, vmax=1)
        axes[i].set_title(titles[i], fontsize=9)
        axes[i].axis("off")

    for i in range(len(fields), len(axes)):
        axes[i].axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    return fig


def field_to_graph_viz(
    fields: Sequence[np.ndarray],
    coords_list: Sequence[np.ndarray],
    edge_indices: Sequence[np.ndarray],
    node_types_list: Sequence[np.ndarray],
    labels: Sequence[str],
    figsize: Tuple[int, int] = (10, 8),
    suptitle: str = "",
) -> plt.Figure:
    """
    Side-by-side field (left) and extracted graph (right) for each condition.

    Args:
        fields: list of (H, W) road field arrays
        coords_list: list of (N, 2) node coordinate arrays
        edge_indices: list of (E, 2) edge index arrays
        node_types_list: list of (N,) node type arrays
        labels: per-row labels
        figsize: figure dimensions
        suptitle: overall title
    Returns:
        matplotlib Figure
    """
    n = len(fields)
    fig, axes = plt.subplots(n, 2, figsize=figsize)

    for i in range(n):
        # Left: field
        axes[i, 0].imshow(fields[i], cmap="gray_r", vmin=0, vmax=1)
        axes[i, 0].set_title(f"Field {labels[i]}", fontsize=9)
        axes[i, 0].axis("off")

        # Right: graph
        rc, re, rt = coords_list[i], edge_indices[i], node_types_list[i]
        if len(rc) > 0:
            for ii, jj in re:
                axes[i, 1].plot(
                    [rc[ii, 0], rc[jj, 0]], [rc[ii, 1], rc[jj, 1]],
                    color="#666", lw=0.8, alpha=0.5, zorder=1,
                )
            for n_idx in range(len(rc)):
                axes[i, 1].scatter(
                    rc[n_idx, 0], rc[n_idx, 1],
                    c=NODE_COLORS.get(int(rt[n_idx]), "#888"),
                    s=12, zorder=2, edgecolors="black", lw=0.3,
                )
        axes[i, 1].set_xlim(0, 1)
        axes[i, 1].set_ylim(0, 1)
        axes[i, 1].set_aspect("equal")
        axes[i, 1].set_title(f"Graph N={len(rc)}", fontsize=9)
        axes[i, 1].axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    return fig


def diversity_bars(
    n_values: Sequence[int],
    e_values: Sequence[int],
    seed_labels: Optional[Sequence[str]] = None,
    title: str = "",
    figsize: Tuple[int, int] = (8, 3),
) -> plt.Figure:
    """
    Bar chart of N (nodes) and E (edges) per seed.

    Args:
        n_values: list of node counts per seed
        e_values: list of edge counts per seed
        seed_labels: optional x-tick labels (default: "Seed 0" ...)
        title: plot title
        figsize: figure dimensions
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    x = np.arange(len(n_values))
    width = 0.35
    ax.bar(x - width / 2, n_values, width, label="N", alpha=0.7)
    ax.bar(x + width / 2, e_values, width, label="E", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(seed_labels or [f"S{i}" for i in range(len(n_values))])
    ax.set_xlabel("Seed")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig

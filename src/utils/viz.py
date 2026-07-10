"""
Shared visualization utilities for road skeleton graph generation and refinement.

Provides reusable plotting patterns:
  - plot_field:                single road field → PNG
  - plot_graph:                single skeleton graph → PNG
  - compare_field_graph:       field vs extracted graph side-by-side → PNG
  - field_grid:                N conditions × M seeds road field grid
  - field_to_graph_viz:        field vs raw vs refined graph grid
  - diversity_bars:            N/E counts per seed bar chart
  - plot_refine_comparison:    before/after refinement overlay
  - plot_score_breakdown:      bar chart of scoring dimensions
  - plot_pick_best_summary:    top-k scores across seeds grid
"""

import os
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

# ---------------------------------------------------------------------------
# Building block: single-field plot
# ---------------------------------------------------------------------------

def plot_field(
    field: np.ndarray,
    save_path: str,
    title: str = "",
    cmap: str = "gray_r",
    figsize: Tuple[int, int] = (5, 5),
    dpi: int = 150,
) -> None:
    """
    Plot a single road field and save to PNG.

    Args:
        field: (H, W) road probability / binary centerline array in [0, 1]
        save_path: output PNG path (created directory if needed)
        title: optional title above the field
        cmap: colormap for imshow
        figsize: figure dimensions
        dpi: output resolution
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(field, cmap=cmap, vmin=0, vmax=1)
    if title:
        ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Building block: single-graph plot
# ---------------------------------------------------------------------------

def plot_graph(
    coords: np.ndarray,
    edge_index: np.ndarray,
    node_types: np.ndarray,
    save_path: str,
    title: str = "",
    figsize: Tuple[int, int] = (5, 5),
    dpi: int = 150,
    show_labels: bool = True,
) -> None:
    """
    Plot a single skeleton graph and save to PNG.

    Args:
        coords: (N, 2) node coordinates in [0, 1]
        edge_index: (E, 2) edge index pairs
        node_types: (N,) node type codes (0=waypoint, 1=junction, 4=dead-end)
        save_path: output PNG path
        title: optional title above the graph
        figsize: figure dimensions
        dpi: output resolution
        show_labels: whether to show "N=..." in the title area
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    _draw_graph(ax, coords, edge_index, node_types)
    label = title
    if show_labels and len(coords) > 0:
        label = f"{title}  N={len(coords)}" if title else f"N={len(coords)}"
    if label:
        ax.set_title(label, fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _draw_graph(
    ax: plt.Axes,
    coords: np.ndarray,
    edge_index: np.ndarray,
    node_types: np.ndarray,
) -> None:
    """Internal helper: draw edges + nodes on a given Axes."""
    if len(coords) == 0:
        return
    for ii, jj in edge_index:
        ax.plot(
            [coords[ii, 0], coords[jj, 0]],
            [coords[ii, 1], coords[jj, 1]],
            color="#666", lw=0.8, alpha=0.5, zorder=1,
        )
    for n_idx in range(len(coords)):
        ax.scatter(
            coords[n_idx, 0], coords[n_idx, 1],
            c=NODE_COLORS.get(int(node_types[n_idx]), "#888"),
            s=12, zorder=2, edgecolors="black", lw=0.3,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")


# ---------------------------------------------------------------------------
# Building block: field + graph side-by-side comparison → single PNG
# ---------------------------------------------------------------------------

def compare_field_graph(
    field: np.ndarray,
    coords: np.ndarray,
    edge_index: np.ndarray,
    node_types: np.ndarray,
    save_path: str,
    title: str = "",
    figsize: Tuple[int, int] = (10, 5),
    dpi: int = 150,
) -> None:
    """
    Side-by-side: road field (left) and extracted graph (right), saved to PNG.

    Args:
        field: (H, W) road field array
        coords: (N, 2) node coordinates
        edge_index: (E, 2) edge index pairs
        node_types: (N,) node type codes
        save_path: output PNG path
        title: overall suptitle
        figsize: figure dimensions
        dpi: output resolution
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=figsize)

    # Left: field
    ax_l.imshow(field, cmap="gray_r", vmin=0, vmax=1)
    ax_l.set_title("Road Field", fontsize=10)
    ax_l.axis("off")

    # Right: graph
    _draw_graph(ax_r, coords, edge_index, node_types)
    ax_r.set_title(f"Graph  N={len(coords)}", fontsize=10)

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Grid-of-fields (unchanged semantics)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Backward-compatible grid: rows of (field, graph) pairs
# ---------------------------------------------------------------------------

def field_to_graph_viz(
    fields: Sequence[np.ndarray],
    coords_list: Sequence[np.ndarray],
    edge_indices: Sequence[np.ndarray],
    node_types_list: Sequence[np.ndarray],
    labels: Sequence[str],
    figsize: Tuple[int, int] = (10, 8),
    suptitle: str = "",
    ref_coords_list: Optional[Sequence[np.ndarray]] = None,
    ref_edge_indices: Optional[Sequence[np.ndarray]] = None,
    ref_node_types_list: Optional[Sequence[np.ndarray]] = None,
) -> plt.Figure:
    """
    Side-by-side field (left) and extracted graph (right) for each condition.

    When *ref_coords_list* is provided, adds a third column showing the
    refined graph (via ``graph_refiner``).

    Args:
        fields: list of (H, W) road field arrays
        coords_list: list of (N, 2) raw node coordinate arrays
        edge_indices: list of (E, 2) raw edge index arrays
        node_types_list: list of (N,) raw node type arrays
        labels: per-row labels
        figsize: figure dimensions
        suptitle: overall title
        ref_coords_list: optional list of (N', 2) refined coords
        ref_edge_indices: optional list of (E', 2) refined edges
        ref_node_types_list: optional list of (N',) refined types
    Returns:
        matplotlib Figure
    """
    n = len(fields)
    n_cols = 3 if ref_coords_list is not None else 2
    fig, axes = plt.subplots(n, n_cols, figsize=figsize)

    for i in range(n):
        # Left: field
        axes[i, 0].imshow(fields[i], cmap="gray_r", vmin=0, vmax=1)
        axes[i, 0].set_title(f"Field {labels[i]}", fontsize=9)
        axes[i, 0].axis("off")

        # Middle: raw graph
        rc, re, rt = coords_list[i], edge_indices[i], node_types_list[i]
        _draw_graph(axes[i, 1], rc, re, rt)
        axes[i, 1].set_title(f"Raw N={len(rc)}", fontsize=9)

        # Right: refined graph (if provided)
        if ref_coords_list is not None:
            rrc, rre, rrt = ref_coords_list[i], ref_edge_indices[i], ref_node_types_list[i]
            _draw_graph(axes[i, 2], rrc, rre, rrt)
            axes[i, 2].set_title(f"Refined N={len(rrc)}", fontsize=9)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Diversity bar chart (unchanged)
# ---------------------------------------------------------------------------

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


# ===========================================================================
# Graph Refinement Visualization (new)
# ===========================================================================

def plot_refine_comparison(
    road_field: np.ndarray,
    raw_graph: Dict,
    ref_graph: Dict,
    save_path: str,
    title: str = "",
    dpi: int = 150,
) -> None:
    """
    Overlay raw and refined graphs on the road field, side-by-side.

    Left: raw skeleton graph on field
    Right: refined graph on field

    Args:
        road_field: (H, W) road probability / field array
        raw_graph: dict with ``coords``, ``edge_index``, ``node_types``
        ref_graph: dict with ``coords``, ``edge_index``, ``node_types``
        save_path: output PNG path
        title: suptitle
        dpi: output resolution
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    H, W = road_field.shape
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 5))

    for ax, g, lbl in zip(
        [ax_l, ax_r],
        [raw_graph, ref_graph],
        ["Raw", "Refined"],
    ):
        ax.imshow(road_field, cmap="gray_r", vmin=0, vmax=1, alpha=0.4)
        cc, ee, tt = g["coords"], g["edge_index"], g["node_types"]
        if len(cc) > 0:
            px = cc * [W, H]
            for ii, jj in ee:
                ax.plot(
                    [px[ii, 0], px[jj, 0]], [px[ii, 1], px[jj, 1]],
                    color="#e74c3c", lw=1.0, alpha=0.6, zorder=2,
                )
            colors = [
                NODE_COLORS.get(int(t), "#888") for t in tt
            ]
            ax.scatter(
                px[:, 0], px[:, 1], c=colors, s=18, zorder=3,
                edgecolors="black", lw=0.3,
            )
        n_w = int((tt == 0).sum()) if len(tt) > 0 else 0
        n_j = int((tt == 1).sum()) if len(tt) > 0 else 0
        ax.set_title(f"{lbl}  N={len(cc)} (W={n_w} J={n_j})", fontsize=10)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_score_breakdown(
    breakdown: Dict[str, float],
    save_path: str,
    title: str = "Score breakdown",
    figsize: Tuple[int, int] = (8, 4),
    dpi: int = 150,
) -> None:
    """
    Bar chart of score dimensions (from ``score_graph``).

    Args:
        breakdown: dict of dimension name → score [0, 1]
        save_path: output PNG path
        title: plot title
        figsize: figure dimensions
        dpi: output resolution
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    names = list(breakdown.keys())
    values = list(breakdown.values())
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    bars = ax.bar(names, values, color=colors, edgecolor="gray", alpha=0.8)
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{v:.2f}", ha="center", va="bottom", fontsize=8,
        )
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_pick_best_summary(
    all_scores: Sequence[float],
    all_n_values: Sequence[int],
    all_j_values: Sequence[int],
    best_idx: int,
    save_path: str,
    title: str = "Seeds ranking",
    figsize: Tuple[int, int] = (10, 4),
    dpi: int = 150,
) -> None:
    """
    Bar chart of all seeds with score, N, and highlight for best.

    Args:
        all_scores: list of scores (one per seed)
        all_n_values: list of N (node counts, one per seed)
        all_j_values: list of J (junction counts, one per seed)
        best_idx: index of the best seed
        save_path: output PNG path
        title: plot title
        figsize: figure dimensions
        dpi: output resolution
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    n = len(all_scores)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    colors = ["#e74c3c" if i == best_idx else "#3498db" for i in range(n)]
    bars = ax.bar(x, all_scores, color=colors, alpha=0.7, edgecolor="gray", lw=0.5)

    # Annotate N and J on each bar
    for i, (bar, s, nv, jv) in enumerate(zip(bars, all_scores, all_n_values, all_j_values)):
        label = f"N={nv}" if i == best_idx else ""
        if label:
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                label, ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"S{i}" for i in range(n)], fontsize=7)
    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=11)
    if best_idx >= 0:
        ax.legend(
            [plt.Rectangle((0, 0), 1, 1, color="#e74c3c"),
             plt.Rectangle((0, 0), 1, 1, color="#3498db")],
            [f"Best (seed {best_idx})", "Other"],
            fontsize=8, loc="lower right",
        )
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_refine_comparison_grid(
    fields: Sequence[np.ndarray],
    raw_graphs: Sequence[Dict],
    ref_graphs: Sequence[Dict],
    labels: Sequence[str],
    save_path: str,
    suptitle: str = "",
    figsize: Tuple[int, int] = (14, 10),
    dpi: int = 150,
) -> None:
    """
    Grid of (field, raw, refined) triples for multiple conditions.

    Each row: road field | raw skeleton | refined graph
    Useful for visual validation of the refine pipeline.

    Args:
        fields: list of (H, W) road field arrays
        raw_graphs: list of raw graph dicts
        ref_graphs: list of refined graph dicts
        labels: per-row labels
        save_path: output PNG path
        suptitle: overall suptitle
        figsize: figure dimensions
        dpi: output resolution
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    n = len(fields)
    fig, axes = plt.subplots(n, 3, figsize=figsize)

    for i in range(n):
        H, W = fields[i].shape

        # Field
        axes[i, 0].imshow(fields[i], cmap="gray_r", vmin=0, vmax=1)
        axes[i, 0].set_title(f"Field {labels[i]}", fontsize=9)
        axes[i, 0].axis("off")

        # Raw graph overlay
        axes[i, 1].imshow(fields[i], cmap="gray_r", vmin=0, vmax=1, alpha=0.3)
        rc, re, rt = raw_graphs[i]["coords"], raw_graphs[i]["edge_index"], raw_graphs[i]["node_types"]
        if len(rc) > 0:
            px = rc * [W, H]
            for ii, jj in re:
                axes[i, 1].plot(
                    [px[ii, 0], px[jj, 0]], [px[ii, 1], px[jj, 1]],
                    color="#e74c3c", lw=0.8, alpha=0.5, zorder=2,
                )
            colors = [NODE_COLORS.get(int(t), "#888") for t in rt]
            axes[i, 1].scatter(
                px[:, 0], px[:, 1], c=colors, s=12, zorder=3,
                edgecolors="black", lw=0.3,
            )
        nw = int((rt == 0).sum()) if len(rt) > 0 else 0
        nj = int((rt == 1).sum()) if len(rt) > 0 else 0
        axes[i, 1].set_title(f"Raw  N={len(rc)} (W={nw} J={nj})", fontsize=9)
        axes[i, 1].axis("off")

        # Refined graph overlay
        axes[i, 2].imshow(fields[i], cmap="gray_r", vmin=0, vmax=1, alpha=0.3)
        rrc, rre, rrt = ref_graphs[i]["coords"], ref_graphs[i]["edge_index"], ref_graphs[i]["node_types"]
        if len(rrc) > 0:
            px = rrc * [W, H]
            for ii, jj in rre:
                axes[i, 2].plot(
                    [px[ii, 0], px[jj, 0]], [px[ii, 1], px[jj, 1]],
                    color="#e74c3c", lw=0.8, alpha=0.5, zorder=2,
                )
            colors = [NODE_COLORS.get(int(t), "#888") for t in rrt]
            axes[i, 2].scatter(
                px[:, 0], px[:, 1], c=colors, s=12, zorder=3,
                edgecolors="black", lw=0.3,
            )
        nw = int((rrt == 0).sum()) if len(rrt) > 0 else 0
        nj = int((rrt == 1).sum()) if len(rrt) > 0 else 0
        axes[i, 2].set_title(f"Refined  N={len(rrc)} (W={nw} J={nj})", fontsize=9)
        axes[i, 2].axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

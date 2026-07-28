"""
Shared visualization utilities for road skeleton graph generation and refinement.

Provides reusable plotting patterns:
  - plot_field:                single road field → PNG
  - plot_graph:                single skeleton graph → PNG
  - compare_field_graph:       field vs extracted graph side-by-side → PNG
  - field_grid:                N conditions x M seeds road field grid
  - field_to_graph_viz:        field vs raw vs refined graph grid
  - diversity_bars:            N/E counts per seed bar chart
  - plot_refine_comparison:    before/after refinement overlay
  - plot_score_breakdown:      bar chart of scoring dimensions
  - plot_pick_best_summary:    top-k scores across seeds grid
  - plot_refine_comparison_grid: grid of (field, raw, refined) triples
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
    dpi: int = 300,
) -> None:
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
    dpi: int = 300,
    show_labels: bool = True,
) -> None:
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
    ax: plt.Axes, coords: np.ndarray, edge_index: np.ndarray, node_types: np.ndarray
) -> None:
    if len(coords) == 0:
        return
    for ii, jj in edge_index:
        ax.plot(
            [coords[ii, 0], coords[jj, 0]],
            [coords[ii, 1], coords[jj, 1]],
            color="#666",
            lw=0.8,
            alpha=0.5,
            zorder=1,
        )
    for n_idx in range(len(coords)):
        ax.scatter(
            coords[n_idx, 0],
            coords[n_idx, 1],
            c=NODE_COLORS.get(int(node_types[n_idx]), "#888"),
            s=12,
            zorder=2,
            edgecolors="black",
            lw=0.3,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")


# ---------------------------------------------------------------------------
# Building block: field + graph side-by-side comparison -> single PNG
# ---------------------------------------------------------------------------


def compare_field_graph(
    field: np.ndarray,
    coords: np.ndarray,
    edge_index: np.ndarray,
    node_types: np.ndarray,
    save_path: str,
    title: str = "",
    figsize: Tuple[int, int] = (10, 5),
    dpi: int = 300,
) -> None:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=figsize)

    ax_l.imshow(field, cmap="gray_r", vmin=0, vmax=1)
    ax_l.set_title("Road Field", fontsize=10)
    ax_l.axis("off")

    _draw_graph(ax_r, coords, edge_index, node_types)
    ax_r.set_title(f"Graph  N={len(coords)}", fontsize=10)

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Grid-of-fields
# ---------------------------------------------------------------------------


def field_grid(
    fields: Sequence[np.ndarray],
    titles: Sequence[str],
    n_cols: int = 4,
    figsize: Tuple[int, int] = (16, 10),
    suptitle: str = "",
    cmap: str = "gray_r",
) -> plt.Figure:
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
    n = len(fields)
    n_cols = 3 if ref_coords_list is not None else 2
    fig, axes = plt.subplots(n, n_cols, figsize=figsize)

    for i in range(n):
        axes[i, 0].imshow(fields[i], cmap="gray_r", vmin=0, vmax=1)
        axes[i, 0].set_title(f"Field {labels[i]}", fontsize=9)
        axes[i, 0].axis("off")

        rc, re, rt = coords_list[i], edge_indices[i], node_types_list[i]
        _draw_graph(axes[i, 1], rc, re, rt)
        axes[i, 1].set_title(f"Raw N={len(rc)}", fontsize=9)

        if ref_coords_list is not None:
            rrc, rre, rrt = ref_coords_list[i], ref_edge_indices[i], ref_node_types_list[i]
            _draw_graph(axes[i, 2], rrc, rre, rrt)
            axes[i, 2].set_title(f"Refined N={len(rrc)}", fontsize=9)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Graph Refinement Visualization
# ---------------------------------------------------------------------------


def plot_refine_comparison(
    road_field: np.ndarray,
    raw_graph: Dict,
    ref_graph: Dict,
    save_path: str,
    title: str = "",
    dpi: int = 300,
) -> None:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    H, W = road_field.shape
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 5))

    for ax, g, lbl in zip([ax_l, ax_r], [raw_graph, ref_graph], ["Raw", "Refined"]):
        ax.imshow(road_field, cmap="gray_r", vmin=0, vmax=1, alpha=0.4)
        cc, ee, tt = g["coords"], g["edge_index"], g["node_types"]
        if len(cc) > 0:
            px = cc * [W, H]
            for ii, jj in ee:
                ax.plot(
                    [px[ii, 0], px[jj, 0]],
                    [px[ii, 1], px[jj, 1]],
                    color="#e74c3c",
                    lw=1.0,
                    alpha=0.6,
                    zorder=2,
                )
            colors = [NODE_COLORS.get(int(t), "#888") for t in tt]
            ax.scatter(px[:, 0], px[:, 1], c=colors, s=18, zorder=3, edgecolors="black", lw=0.3)
        n_w = int((tt == 0).sum()) if len(tt) > 0 else 0
        n_j = int((tt == 1).sum()) if len(tt) > 0 else 0
        ax.set_title(f"{lbl}  N={len(cc)} (W={n_w} J={n_j})", fontsize=10)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=12)
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
    dpi: int = 300,
) -> None:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    n = len(fields)
    fig, axes = plt.subplots(n, 3, figsize=figsize)

    for i in range(n):
        H, W = fields[i].shape

        axes[i, 0].imshow(fields[i], cmap="gray_r", vmin=0, vmax=1)
        axes[i, 0].set_title(f"Field {labels[i]}", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(fields[i], cmap="gray_r", vmin=0, vmax=1, alpha=0.3)
        rc, re, rt = (
            raw_graphs[i]["coords"],
            raw_graphs[i]["edge_index"],
            raw_graphs[i]["node_types"],
        )
        if len(rc) > 0:
            px = rc * [W, H]
            for ii, jj in re:
                axes[i, 1].plot(
                    [px[ii, 0], px[jj, 0]],
                    [px[ii, 1], px[jj, 1]],
                    color="#e74c3c",
                    lw=0.8,
                    alpha=0.5,
                    zorder=2,
                )
            colors = [NODE_COLORS.get(int(t), "#888") for t in rt]
            axes[i, 1].scatter(
                px[:, 0], px[:, 1], c=colors, s=12, zorder=3, edgecolors="black", lw=0.3
            )
        nw = int((rt == 0).sum()) if len(rt) > 0 else 0
        nj = int((rt == 1).sum()) if len(rt) > 0 else 0
        axes[i, 1].set_title(f"Raw  N={len(rc)} (W={nw} J={nj})", fontsize=9)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(fields[i], cmap="gray_r", vmin=0, vmax=1, alpha=0.3)
        rrc, rre, rrt = (
            ref_graphs[i]["coords"],
            ref_graphs[i]["edge_index"],
            ref_graphs[i]["node_types"],
        )
        if len(rrc) > 0:
            px = rrc * [W, H]
            for ii, jj in rre:
                axes[i, 2].plot(
                    [px[ii, 0], px[jj, 0]],
                    [px[ii, 1], px[jj, 1]],
                    color="#e74c3c",
                    lw=0.8,
                    alpha=0.5,
                    zorder=2,
                )
            colors = [NODE_COLORS.get(int(t), "#888") for t in rrt]
            axes[i, 2].scatter(
                px[:, 0], px[:, 1], c=colors, s=12, zorder=3, edgecolors="black", lw=0.3
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

"""Road graph and field visualization utilities."""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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


def _draw_field(
    ax: plt.Axes,
    field: np.ndarray,
    *,
    cmap: str = "gray_r",
    origin: str = "upper",
    extent: tuple[float, float, float, float] | None = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    """Render a road field onto an existing axis."""
    ax.imshow(field, cmap=cmap, vmin=vmin, vmax=vmax, origin=origin, extent=extent)


def plot_field(
    field: np.ndarray,
    save_path: str,
    title: str = "",
    cmap: str = "gray_r",
    figsize: tuple[int, int] = (5, 5),
    dpi: int = 300,
) -> None:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    _draw_field(ax, field, cmap=cmap)
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
    figsize: tuple[int, int] = (5, 5),
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
    ax: plt.Axes,
    coords: np.ndarray,
    edge_index: np.ndarray,
    node_types: np.ndarray,
    *,
    edge_color: str = "#666",
    edge_lw: float = 0.8,
    edge_alpha: float = 0.5,
    node_color: str | None = None,
    node_size: int | Callable[[int], int] = 12,
    fixed_limits: tuple[float, float, float, float] | None = (0.0, 0.0, 1.0, 1.0),
    hide_axes: bool = True,
) -> None:
    """Render a graph (edges + nodes) onto an existing axis.

    ``node_color`` overrides per-type ``NODE_COLORS`` with a single color;
    ``node_size`` may be an int or a callable of ``len(coords)``;
    ``fixed_limits=(x0, y0, x1, y1)`` fixes axis limits, ``None`` auto-pads.
    """
    if len(coords) == 0:
        return
    for ii, jj in edge_index:
        ax.plot(
            [coords[ii, 0], coords[jj, 0]],
            [coords[ii, 1], coords[jj, 1]],
            color=edge_color,
            lw=edge_lw,
            alpha=edge_alpha,
            zorder=1,
        )
    colors = (
        [node_color] * len(coords)
        if node_color is not None
        else [NODE_COLORS.get(int(t), "#888") for t in node_types]
    )
    size = node_size(len(coords)) if callable(node_size) else node_size
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=size, zorder=2, edgecolors="black", lw=0.3)
    if fixed_limits is None:
        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0)
        padx = max((x1 - x0) * 0.05, 1e-9)
        pady = max((y1 - y0) * 0.05, 1e-9)
        ax.set_xlim(x0 - padx, x1 + padx)
        ax.set_ylim(y0 - pady, y1 + pady)
    else:
        ax.set_xlim(fixed_limits[0], fixed_limits[2])
        ax.set_ylim(fixed_limits[1], fixed_limits[3])
    ax.set_aspect("equal")
    if hide_axes:
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
    figsize: tuple[int, int] = (10, 5),
    dpi: int = 300,
) -> None:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=figsize)

    _draw_field(ax_l, field)
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
    figsize: tuple[int, int] = (16, 10),
    suptitle: str = "",
    cmap: str = "gray_r",
) -> plt.Figure:
    n_rows = (len(fields) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

    for i in range(len(fields)):
        _draw_field(axes[i], fields[i], cmap=cmap)
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
    figsize: tuple[int, int] = (10, 8),
    suptitle: str = "",
    ref_coords_list: Sequence[np.ndarray] | None = None,
    ref_edge_indices: Sequence[np.ndarray] | None = None,
    ref_node_types_list: Sequence[np.ndarray] | None = None,
) -> plt.Figure:
    n = len(fields)
    n_cols = 3 if ref_coords_list is not None else 2
    fig, axes = plt.subplots(n, n_cols, figsize=figsize)

    for i in range(n):
        _draw_field(axes[i, 0], fields[i])
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
    raw_graph: dict,
    ref_graph: dict,
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
    raw_graphs: Sequence[dict],
    ref_graphs: Sequence[dict],
    labels: Sequence[str],
    save_path: str,
    suptitle: str = "",
    figsize: tuple[int, int] = (14, 10),
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


# ---------------------------------------------------------------------------
# Compressed intersection graph panel (graph_utils node types: 1=junction,
# 3=roundabout, 4=endpoint) — used by the per-map pipeline visualisation.
# ---------------------------------------------------------------------------


def draw_intersection_graph(
    ax: plt.Axes,
    coords: np.ndarray,
    edge_index: np.ndarray,
    node_types: np.ndarray,
    lanes_per_dir: np.ndarray | None = None,
    road_class: np.ndarray | None = None,
    geoms: Sequence[np.ndarray] | None = None,
    legend: bool = True,
) -> list:
    """Draw a compressed intersection graph onto *ax*.

    Edges are coloured by road class (HUSL: primary green / secondary purple /
    local orange) with line width proportional to lane count; nodes are drawn
    by type (junction blue / roundabout red / endpoint yellow).  Returns the
    legend handles so the caller can re-use them on a combined figure.
    """
    import husl
    from matplotlib.lines import Line2D

    rc_colors = {
        1: husl.husl_to_hex(150, 65, 55),  # green
        2: husl.husl_to_hex(280, 65, 55),  # purple
        3: husl.husl_to_hex(40, 65, 55),  # orange
    }
    ax.set_facecolor("#f5f5f5")
    n_j = int((node_types == 1).sum())
    n_r = int((node_types == 3).sum())
    n_e = int((node_types == 4).sum())

    if len(coords):
        for j, (u, v) in enumerate(edge_index):
            lw = min(3.0, 0.3 + 0.35 * int(lanes_per_dir[j])) if j < len(lanes_per_dir) else 0.5
            rc = int(road_class[j]) if road_class is not None and j < len(road_class) else 2
            color = rc_colors.get(rc, "#888")
            geom = geoms[j] if geoms is not None and j < len(geoms) else None
            if geom is not None and len(geom) > 2:
                ax.plot(geom[:, 0], geom[:, 1], color=color, lw=lw, alpha=0.7)
            else:
                ax.plot(
                    [coords[u, 0], coords[v, 0]],
                    [coords[u, 1], coords[v, 1]],
                    color=color,
                    lw=lw,
                    alpha=0.7,
                )

        if n_j:
            ax.scatter(
                coords[node_types == 1, 0],
                coords[node_types == 1, 1],
                c="#1565C0",
                s=40,
                edgecolors="black",
                lw=0.5,
                zorder=3,
            )
        if n_r:
            ax.scatter(
                coords[node_types == 3, 0],
                coords[node_types == 3, 1],
                c="#E53935",
                s=40,
                edgecolors="black",
                lw=0.5,
                zorder=3,
            )
        if n_e:
            ax.scatter(
                coords[node_types == 4, 0],
                coords[node_types == 4, 1],
                c="#FDD835",
                s=16,
                edgecolors="black",
                lw=0.3,
                zorder=3,
            )

        xs, ys = coords[:, 0], coords[:, 1]
        mx = max(0.02, (xs.max() - xs.min()) * 0.05)
        my = max(0.02, (ys.max() - ys.min()) * 0.05)
        ax.set_xlim(xs.min() - mx, xs.max() + mx)
        ax.set_ylim(ys.min() - my, ys.max() + my)

    items = []
    if n_j:
        items.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#1565C0",
                markeredgecolor="black",
                markersize=8,
                label=f"Junction ({n_j})",
            )
        )
    if n_r:
        items.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#E53935",
                markeredgecolor="black",
                markersize=8,
                label=f"Roundabout ({n_r})",
            )
        )
    if n_e:
        items.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#FDD835",
                markeredgecolor="black",
                markersize=6,
                label=f"Endpoint ({n_e})",
            )
        )
    for rc_label, rc_val, hue in [
        ("Primary (1)", 1, 150),
        ("Secondary (2)", 2, 280),
        ("Local (3)", 3, 40),
    ]:
        items.append(Line2D([0], [0], color=husl.husl_to_hex(hue, 65, 55), lw=2, label=rc_label))
    for lw_val, label in [(0.65, "1 lane"), (1.0, "2 lanes"), (1.35, "3+ lanes")]:
        items.append(Line2D([0], [0], color="#555", lw=lw_val, label=label))
    if legend:
        ax.legend(
            handles=items,
            fontsize=8,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
            framealpha=0.9,
        )
    return items


# ---------------------------------------------------------------------------
# VQ-VAE reconstruction panels: per-sample folder (0_orig / 1_recon / 2_error
# + combined) plus a full-sample grid.
# ---------------------------------------------------------------------------


def _to_np(x):
    """Convert a torch tensor or numpy array to float ndarray."""
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def plot_recon_grid(
    orig_fields, recon_fields, ious: Sequence[float], out_dir: str, n_samples: int | None = None
) -> list:
    """Save VQ-VAE reconstruction panels following the per-sample output rule.

    For each sample writes ``recon_{i}/`` with ``0_orig.png``, ``1_recon.png``,
    ``2_error.png`` and a ``combined.png`` (1×3); also writes a full-sample
    ``grid.png`` (N×2, orig | recon).  Returns the list of per-sample dirs.
    """
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = min(n_samples or len(orig_fields), len(orig_fields))
    dirs = []

    for i in range(n):
        folder = out / f"recon_{i}"
        folder.mkdir(parents=True, exist_ok=True)
        orig = _to_np(orig_fields[i, 0])
        rec = _to_np(recon_fields[i, 0])
        diff = np.abs(orig - rec)
        iou = float(ious[i]) if i < len(ious) else float("nan")

        # Individual panels
        panels = [
            ("0_orig.png", orig, "Original", "gray_r", (0, 1)),
            ("1_recon.png", rec, "Reconstructed", "gray_r", (0, 1)),
            ("2_error.png", diff, f"Error (IoU={iou:.3f})", "hot", (0, 0.5)),
        ]
        for fname, arr, title, cmap, vrange in panels:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(arr, cmap=cmap, vmin=vrange[0], vmax=vrange[1])
            ax.set_title(title, fontsize=10)
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(folder / fname, dpi=150, bbox_inches="tight")
            plt.close(fig)

        # Combined (1×3)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, (arr, title, cmap, vrange) in zip(
            axes,
            [
                (orig, "Original", "gray_r", (0, 1)),
                (rec, "Reconstructed", "gray_r", (0, 1)),
                (diff, f"Error (IoU={iou:.3f})", "hot", (0, 0.5)),
            ],
        ):
            ax.imshow(arr, cmap=cmap, vmin=vrange[0], vmax=vrange[1])
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(folder / "combined.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        dirs.append(folder)

    # Full-sample grid (N×2)
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    axes = np.atleast_2d(axes)
    for i in range(n):
        orig = _to_np(orig_fields[i, 0])
        rec = _to_np(recon_fields[i, 0])
        axes[i, 0].imshow(orig, cmap="gray_r", vmin=0, vmax=1)
        axes[i, 0].set_title(f"Original {i}", fontsize=9)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(rec, cmap="gray_r", vmin=0, vmax=1)
        axes[i, 1].set_title(f"Recon {i}", fontsize=9)
        axes[i, 1].axis("off")
    fig.tight_layout()
    fig.savefig(out / "grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dirs

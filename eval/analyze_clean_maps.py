#!/usr/bin/env python3
"""
Analyze graph extraction for each baseline, using each baseline's OWN method.

Graphs are located via ``eval/*.find_clean_map`` (the same per-baseline search
the export scripts use, targeting ``--target-nodes`` post-contraction nodes):
- HDMapGen : dict_to_nx (adjacency + degree-2 contraction)
- RoadGen  : generate_one_map (mask → skeleton → graph)
- MetaDrive: generate_metadrive_map (skeleton extraction — the only method
  that yields well-spread 10-80 node graphs, see eval/metadrive.py)

Shows a 3-panel figure per baseline:
  A — Road polylines (ground-truth)
  B — Graph overlaid on polylines (visually verify every node makes sense)
  C — Degree distribution

Usage:
    conda activate road-weaver
    python scripts/analyze_clean_maps.py [--target-nodes 40] [--only hdmapgen]
"""

from __future__ import annotations

import importlib
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from eval.polyline_graph import save_clean_map

# Lazy per-baseline import: eval.metadrive pulls in MetaDriveEnv (heavy).
FINDER_MODULES = {
    "hdmapgen": "eval.hdmapgen",
    "roadgen": "eval.roadgen",
    "metadrive": "eval.metadrive",
}


def _plot_analysis(polylines: list[np.ndarray], G, title: str, filepath: Path, dpi: int = 300):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    degs = [d for _, d in G.degree()]
    deg_counter = Counter(degs)
    n_deg1 = deg_counter.get(1, 0)
    n_deg2 = deg_counter.get(2, 0)
    n_deg3 = deg_counter.get(3, 0)
    n_deg4p = sum(v for k, v in deg_counter.items() if k >= 4)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5))

    # ── Panel A: polylines ──────────────────────────────────────────────
    ax = axes[0]
    for pts in polylines:
        pts = np.asarray(pts)
        if len(pts) >= 2:
            ax.plot(pts[:, 0], pts[:, 1], color="#222222", lw=1.2)
    ax.set_aspect("equal")
    ax.set_title(f"Road Polylines  |  {len(polylines)} segments", fontsize=11)
    ax.axis("off")

    # ── Panel B: graph on polylines ─────────────────────────────────────
    ax = axes[1]
    for pts in polylines:
        pts = np.asarray(pts)
        if len(pts) >= 2:
            ax.plot(pts[:, 0], pts[:, 1], color="#cccccc", lw=1.5, alpha=0.6, zorder=1)

    if n_nodes > 0:
        coords = np.array([G.nodes[n]["coords"] for n in G.nodes()])
        for u, v in G.edges():
            if u in G and v in G:
                c1, c2 = G.nodes[u]["coords"], G.nodes[v]["coords"]
                ax.plot(
                    [c1[0], c2[0]], [c1[1], c2[1]], color="#e74c3c", lw=2.5, alpha=0.85, zorder=3
                )

        colors = []
        for n in G.nodes():
            d = G.degree(n)
            if d == 1:
                colors.append("#2ecc71")
            elif d == 2:
                colors.append("#3498db")
            elif d == 3:
                colors.append("#f39c12")
            else:
                colors.append("#e74c3c")

        ax.scatter(
            coords[:, 0], coords[:, 1], c=colors, s=55, zorder=5, edgecolors="white", linewidths=0.7
        )

        leg = [
            Patch(facecolor="#2ecc71", label=f"Deg 1 (dead-end) ×{n_deg1}"),
            Patch(facecolor="#3498db", label=f"Deg 2 ×{n_deg2}"),
            Patch(facecolor="#f39c12", label=f"Deg 3 ×{n_deg3}"),
            Patch(facecolor="#e74c3c", label=f"Deg 4+ ×{n_deg4p}"),
            Line2D([0], [0], color="#e74c3c", lw=2.5, label=f"Edges ({n_edges})"),
        ]
        ax.legend(handles=leg, loc="upper right", fontsize=7, framealpha=0.9)
        info = f"{n_nodes}N / {n_edges}E  |  deg1={n_deg1}  deg2={n_deg2}  deg3={n_deg3}  deg4+={n_deg4p}"
    else:
        info = "Empty graph"

    ax.set_aspect("equal")
    ax.set_title(f"Graph — {info}", fontsize=10)
    ax.axis("off")

    # ── Panel C: degree histogram ──────────────────────────────────────
    ax = axes[2]
    max_deg = max(degs) if degs else 1
    bins = np.arange(0.5, max_deg + 1.5, 1)
    ax.hist(degs, bins=bins, color="#4C72B0", edgecolor="white", alpha=0.85, rwidth=0.8)
    ax.set_xlabel("Node Degree", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Degree Distribution", fontsize=12)
    ax.set_xticks(range(1, max_deg + 1))
    ax.grid(axis="y", alpha=0.3)
    for d, c in sorted(deg_counter.items()):
        ax.text(d, c + 0.3, str(c), ha="center", va="bottom", fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(filepath), dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [analysis] Saved {filepath}  |  {n_nodes}N {n_edges}E")


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-baseline analysis — locate a graph via eval/*.find_clean_map
# ═══════════════════════════════════════════════════════════════════════════════


def _analyze(
    out_dir: Path,
    name: str,
    title: str,
    n_target: int,
    max_aspect: float,
    dpi: int,
    clean_only: bool = False,
):
    """Find one graph for *name* and render the 3-panel analysis + clean map.

    With ``clean_only`` the analysis figure is skipped and only the clean map
    is written (the former ``export_clean_maps.py`` behaviour).
    """
    find = importlib.import_module(FINDER_MODULES[name]).find_clean_map
    r = find(n_target=n_target, max_aspect=max_aspect)
    G, polylines = r["graph"], r["polylines"]
    print(f"  {r['n_nodes']}N / {G.number_of_edges()}E")
    if not clean_only:
        _plot_analysis(polylines, G, title, out_dir / f"{name}_analysis.png", dpi)
    save_clean_map(polylines, str(out_dir / f"{name}_clean_map.png"), dpi=dpi)


def _analyze_hdmapgen(
    out_dir: Path, n_target: int, max_aspect: float, dpi: int, clean_only: bool = False
):
    print("\n━━━ HDMapGen ━━━")
    _analyze(
        out_dir,
        "hdmapgen",
        "HDMapGen — Node / Edge (legacy)",
        n_target,
        max_aspect,
        dpi,
        clean_only,
    )


def _analyze_roadgen(
    out_dir: Path, n_target: int, max_aspect: float, dpi: int, clean_only: bool = False
):
    print("\n━━━ RoadGen ━━━")
    _analyze(
        out_dir,
        "roadgen",
        "RoadGen — Node / Edge (skeleton+graph)",
        n_target,
        max_aspect,
        dpi,
        clean_only,
    )


def _analyze_metadrive(
    out_dir: Path, n_target: int, max_aspect: float, dpi: int, clean_only: bool = False
):
    print("\n━━━ MetaDrive ━━━")
    _analyze(
        out_dir,
        "metadrive",
        "MetaDrive — Node / Edge (skeleton)",
        n_target,
        max_aspect,
        dpi,
        clean_only,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--output", type=str, default=str(REPO / "analysis" / "clean_maps"))
    parser.add_argument(
        "--target-nodes",
        type=int,
        default=40,
        help="target post-contraction node count (default 40; 0 = largest)",
    )
    parser.add_argument("--max-aspect", type=float, default=2.0)
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="comma-separated baselines to analyze (default: all three)",
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="export clean maps only (skip the 3-panel analysis figures)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    print(f"Output: {out_dir}/   DPI: {args.dpi}   Target: {args.target_nodes} nodes\n")

    ANALYSERS = {
        "hdmapgen": _analyze_hdmapgen,
        "roadgen": _analyze_roadgen,
        "metadrive": _analyze_metadrive,
    }
    t0 = time.time()
    for name, fn in ANALYSERS.items():
        if only and name not in only:
            continue
        try:
            fn(out_dir, args.target_nodes, args.max_aspect, args.dpi, args.clean_only)
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'=' * 50}")
    print(f"Done in {time.time() - t0:.0f}s")
    for p in sorted(out_dir.glob("*analysis*")):
        print(f"  {p.name}  ({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

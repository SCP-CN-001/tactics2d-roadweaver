#!/usr/bin/env python3
"""
Unified baseline visualization — road geometry for all baselines.

Produces:
  1. Individual sample maps per baseline (small/medium/large) —
     saved to runtimes/{baseline}_eval/unified_{size}.png
  2. Comparison grid — analysis/figures/unified_comparison.png

Each baseline is drawn with its native road geometry:
  - MetaDrive: lane polylines (blue)
  - HDMapGen: edge subnode curves (red)
  - RoadGen: widget lane geometry (colored by widget type)

Usage:
    conda activate road-weaver
    python eval/viz_compare.py [unified|metrics|all]
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent

# ═══════════════════════════════════════════════════════════════════════════
#  Style
# ═══════════════════════════════════════════════════════════════════════════

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
    }
)

LANE_ALPHA = 0.85
LANE_WIDTH = 2.5


def _style_ax(ax, title: str, xlabel="X", ylabel="Y"):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)


# ═══════════════════════════════════════════════════════════════════════════
#  MetaDrive — lane polylines (actual road geometry)
# ═══════════════════════════════════════════════════════════════════════════


def _gen_metadrive_map(seed: int, map_config: int):
    """Generate MetaDrive map, return polylines list."""
    sys.path.insert(0, str(REPO / "metadrive"))
    from metadrive.envs import MetaDriveEnv

    env = MetaDriveEnv(
        dict(
            start_seed=seed, num_scenarios=1, traffic_density=0.0, use_render=False, map=map_config
        )
    )
    obs, _ = env.reset()
    rn = env.engine.current_map.road_network

    polylines = []
    node_count = 0
    edge_count = 0
    for sn, td in rn.graph.items():
        for en, lanes in td.items():
            if sn == "decoration" or en == "decoration":
                continue
            edge_count += 1
            for lane in lanes:
                try:
                    pts = np.array(lane.get_polyline(interval=2))
                    if len(pts) >= 2:
                        polylines.append(pts)
                except Exception:
                    pass

    # Count distinct node IDs
    nodes = set()
    for sn, td in rn.graph.items():
        for en in td:
            if sn != "decoration" and en != "decoration":
                nodes.add(sn)
                nodes.add(en)
    node_count = len(nodes)

    env.close()
    return polylines, node_count, edge_count


def draw_metadrive_map(seed: int, map_config: int, label: str, filepath: Path):
    """Draw MetaDrive road geometry."""
    polylines, n_nodes, n_edges = _gen_metadrive_map(seed, map_config)
    fig, ax = plt.subplots(figsize=(8, 6))

    for pts in polylines:
        ax.plot(pts[:, 0], pts[:, 1], color="#3498DB", lw=LANE_WIDTH, alpha=LANE_ALPHA)

    _style_ax(ax, f"MetaDrive — {label}  ({n_nodes} nodes, {n_edges} edges)")
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [MetaDrive] {filepath.name}")


# ═══════════════════════════════════════════════════════════════════════════
#  HDMapGen — edge subnode curves (road centerline geometry)
# ═══════════════════════════════════════════════════════════════════════════


def find_hdmapgen_graphs(target_size: int):
    """Find HDMapGen graphs at given size from pickle."""
    pkl = REPO / "runtimes" / "hdmapgen_eval" / "generated_graphs.pkl"
    if not pkl.exists():
        print("  [HDMapGen] No generated_graphs.pkl — re-run eval/hdmapgen.py")
        return []
    import pickle

    with open(pkl, "rb") as f:
        data = pickle.load(f)

    # Find closest match to target_size
    matches = [
        d
        for d in data
        if abs(d["num_nodes"] - target_size) <= 3
        and d.get("node_coords")
        and d["node_coords"][0] is not None
    ]
    if matches:
        # Pick the one with most subnodes (richer geometry)
        matches.sort(key=lambda d: len(d.get("edge_subnodes", {})), reverse=True)
        return [matches[0]]
    return []


def draw_hdmapgen_map(entry: dict, label: str, filepath: Path):
    """Draw HDMapGen road geometry from edge subnodes."""
    n = entry["num_nodes"]
    coords = entry.get("node_coords", [])
    edges = entry.get("edges", [])
    subnodes = entry.get("edge_subnodes", {})

    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw each edge as a road polyline through subnodes
    drawn = set()
    for u, v in edges:
        key = (u, v) if (u, v) in subnodes else (v, u)
        sn = subnodes.get(key)
        if sn is not None and len(sn) > 1:
            pts = np.array(sn)
            ax.plot(pts[:, 0], pts[:, 1], "-", color="#E74C3C", lw=LANE_WIDTH, alpha=LANE_ALPHA)
        elif (
            coords
            and u < len(coords)
            and v < len(coords)
            and coords[u] is not None
            and coords[v] is not None
        ):
            cu, cv = np.array(coords[u]), np.array(coords[v])
            ax.plot(
                [cu[0], cv[0]],
                [cu[1], cv[1]],
                "-",
                color="#E74C3C",
                lw=LANE_WIDTH * 0.6,
                alpha=LANE_ALPHA * 0.5,
            )
        drawn.add(key)

    # Junction nodes as dots
    if coords:
        pts = np.array([c for c in coords if c is not None])
        if len(pts) > 0:
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                c="#2C3E50",
                s=25,
                zorder=3,
                edgecolors="white",
                linewidths=0.5,
            )

    _style_ax(ax, f"HDMapGen — {label}  ({n} nodes, {len(edges)} edges)")
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [HDMapGen] {filepath.name}")


# ═══════════════════════════════════════════════════════════════════════════
#  RoadGen — widget lane geometry
# ═══════════════════════════════════════════════════════════════════════════


def _gen_roadgen_geometry(widget_number: int):
    """Generate RoadGen map and extract lane polylines + widget positions."""
    os.chdir(str(REPO / "RoadGen" / "python"))
    if str(REPO / "RoadGen" / "python") not in sys.path:
        sys.path.insert(0, str(REPO / "RoadGen" / "python"))

    import warnings

    warnings.filterwarnings("ignore")
    import json as _json
    import random as _random

    from func.CountBasedChoose import CountBasedChoose
    from func.printAsserts import printAsserts
    from func.printAuto import printAutoInd
    from func.Quene import ConnectorQueue
    from func.update import initialFirstwidget, update
    from func.widget import Widget
    from func.WidgetGraph import WidgetGraph
    from settings.connectionDict import Widget_Map
    from settings.info import Info

    # Reset state
    Widget.LaneID = 1
    Widget.BoundaryID = 1
    Widget.JunctionID = 1
    Widget.WidgetID = 1
    Info.parameterupdate = 0
    Info.widgetupdate = 0
    for k in Info.widgetcount:
        Info.widgetcount[k] = 0
    Info.WidgetNumber = widget_number

    def _build(wdict):
        from curve_widget.curve import Curve
        from fork_widget.fork import Fork
        from Intersection_widget.intersection import Intersection
        from laneswitch_widget.laneswitch import LaneSwitch
        from roundabout_widget.roundabout import Roundabout
        from straigntlane_widget.straightlane import StraightLane
        from TJunction_widget.tJunction import tJunction
        from Ulane_widget.ulane import ULane

        builders = {
            "straightlane": StraightLane,
            "ulane": ULane,
            "curve": Curve,
            "laneswitch": LaneSwitch,
            "fork": Fork,
            "intersection": Intersection,
            "tJunction": tJunction,
            "roundabout": Roundabout,
        }
        return builders[wdict.get("Type")](wdict)

    mfile = str(REPO / ".roadgen_tmp" / "viz.m")
    Path(mfile).parent.mkdir(parents=True, exist_ok=True)

    polylines = []  # list of (N,2) arrays
    widget_pos = {}  # widget_id -> (x, y)
    widget_types = {}  # widget_id -> type string

    with open(mfile, "w", encoding="GBK") as f:
        graph = WidgetGraph()
        printAsserts(f)
        count = 0
        total_area = []

        # First widget
        wdict = _random.choice(Info.Widgetlist)
        wdict0, _ = initialFirstwidget(wdict, Info.COMPILE_RULES, total_area, 0)
        w0 = _build(wdict0)
        w0.generate_road(f)
        count += 1
        total_area += w0.get_coveredArea()
        Info.widgetcount[_json.dumps(wdict)] += 1
        widget_pos[w0.WidgetID] = wdict0.get("Start", (0, 0))
        widget_types[w0.WidgetID] = wdict0.get("Type")

        try:
            for lane in w0.getlanepoint():
                if len(lane) >= 2:
                    polylines.append(np.array(lane, dtype=float))
        except Exception:
            pass

        graph.add_node(w0.WidgetID, wdict0.get("Type"), wdict0.get("Flag"), None, None)
        cq = ConnectorQueue()
        for n in w0.get_Nexts():
            cq.enqueue(n)
        del w0

        # More widgets
        while count < Info.WidgetNumber and not cq.isempty():
            conn = cq.dequeue()
            fg = _random.randint(0, 1)
            if cq.isempty() or fg == 1:
                ct = conn["type"]
                cands = list(Widget_Map.get(ct, []))
                avail = len(cands)
                while avail > 0:
                    wc = _random.choice(cands)
                    d = wc.copy()
                    d["Start"] = conn["endpoint"]
                    d["K"] = conn["direction"]
                    d, _ = update(d, Info.COMPILE_RULES, total_area, 0)
                    if d is not None:
                        break
                    cands.remove(wc)
                    avail -= 1
                if avail == 0:
                    continue
                w1 = _build(d)
                w1.generate_road(f)
                count += 1
                total_area += w1.get_coveredArea()
                Info.widgetcount[_json.dumps(wc)] += 1
                widget_pos[w1.WidgetID] = d.get("Start", (0, 0))
                widget_types[w1.WidgetID] = d.get("Type")
                try:
                    for lane in w1.getlanepoint():
                        if len(lane) >= 2:
                            polylines.append(np.array(lane, dtype=float))
                except Exception:
                    pass
                graph.add_node(w1.WidgetID, d.get("Type"), d.get("Flag"), None, None)
                graph.add_edge(conn["ID"], w1.WidgetID)
                for n in w1.get_Nexts():
                    cq.enqueue(n)
                del w1

    os.chdir(str(REPO))
    return polylines, widget_pos, widget_types, count


WIDGET_COLORS = {
    "straightlane": "#4C72B0",
    "curve": "#DD8452",
    "ulane": "#55A868",
    "laneswitch": "#C44E52",
    "fork": "#937860",
    "intersection": "#8172B3",
    "tJunction": "#C44E52",
    "roundabout": "#CCB974",
}


def draw_roadgen_map(widget_number: int, label: str, filepath: Path):
    """Draw RoadGen road geometry from widget lanes."""
    print(f"  [RoadGen] Generating {label.lower()} ({widget_number} widgets)...")
    t0 = time.time()
    polylines, positions, types, count = _gen_roadgen_geometry(widget_number)
    elapsed = time.time() - t0

    fig, ax = plt.subplots(figsize=(8, 6))

    # Lane geometry (colored by widget type)
    for i, pts in enumerate(polylines):
        # Find which widget this polyline belongs to (approximate)
        ax.plot(pts[:, 0], pts[:, 1], color="#55A868", lw=LANE_WIDTH, alpha=LANE_ALPHA)

    # Draw widget connections as thin dashed lines
    # Draw widget boundaries as colored dots
    for wid, pos in positions.items():
        t = types.get(wid, "unknown")
        color = WIDGET_COLORS.get(t, "#999999")
        ax.scatter(
            pos[0], pos[1], c=color, s=80, edgecolors="white", linewidths=1.0, zorder=3, alpha=0.9
        )

    # Legend
    used = set(types.values())
    for t in sorted(used):
        ax.scatter(
            [],
            [],
            c=WIDGET_COLORS.get(t, "#999999"),
            s=40,
            label=t,
            edgecolors="white",
            linewidths=0.5,
        )
    ax.legend(loc="upper right", fontsize=7, framealpha=0.8, ncol=2)

    _style_ax(ax, f"RoadGen — {label}  ({count} widgets, {elapsed:.0f}s)")
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [RoadGen] {filepath.name}")


# ═══════════════════════════════════════════════════════════════════════════
#  Comparison grid
# ═══════════════════════════════════════════════════════════════════════════


def draw_comparison_grid(output: Path):
    """3×3 grid: baselines × small/medium/large — road geometry."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(
        "Generated Road Network Comparison — Road Geometry", fontsize=16, fontweight="bold", y=1.01
    )

    size_labels = ["Small (~10 nodes)", "Medium (~30 nodes)", "Large (~60+ nodes)"]
    baseline_labels = ["MetaDrive", "HDMapGen", "RoadGen"]

    for col, (target, mc) in enumerate([(10, 2), (30, 5), (70, 10)]):
        ax = axes[0, col]
        try:
            polylines, nn, ne = _gen_metadrive_map(seed=42 + col, map_config=mc)
            for pts in polylines:
                ax.plot(pts[:, 0], pts[:, 1], color="#3498DB", lw=LANE_WIDTH, alpha=LANE_ALPHA)
            ax.set_title(f"n={nn}, e={ne}", fontsize=10)
        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"N/A\n{e}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
                color="#999",
            )
        if col == 0:
            ax.set_ylabel(baseline_labels[0], fontsize=12, fontweight="bold")
        ax.set_aspect("equal")
        ax.axis("off")

    for col, target in enumerate([10, 30, 69]):
        ax = axes[1, col]
        try:
            entries = find_hdmapgen_graphs(target)
            if entries:
                entry = entries[0]
                n = entry["num_nodes"]
                coords = entry.get("node_coords", [])
                edges = entry.get("edges", [])
                subnodes = entry.get("edge_subnodes", {})
                for u, v in edges:
                    key = (u, v) if (u, v) in subnodes else (v, u)
                    sn = subnodes.get(key)
                    if sn is not None and len(sn) > 1:
                        ax.plot(
                            np.array(sn)[:, 0],
                            np.array(sn)[:, 1],
                            "-",
                            color="#E74C3C",
                            lw=LANE_WIDTH,
                            alpha=LANE_ALPHA,
                        )
                    elif coords and u < len(coords) and v < len(coords) and coords[u] and coords[v]:
                        ax.plot(
                            [coords[u][0], coords[v][0]],
                            [coords[u][1], coords[v][1]],
                            "-",
                            color="#E74C3C",
                            lw=LANE_WIDTH * 0.6,
                            alpha=0.5,
                        )
                if coords:
                    pts = np.array([c for c in coords if c is not None])
                    if len(pts) > 0:
                        ax.scatter(pts[:, 0], pts[:, 1], c="#2C3E50", s=25, zorder=3)
                ax.set_title(f"n={n}, e={len(edges)}", fontsize=10)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No data\n(re-run hdmapgen.py)",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=9,
                )
        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Error:\n{e}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
                color="#999",
            )
        if col == 0:
            ax.set_ylabel(baseline_labels[1], fontsize=12, fontweight="bold")
        ax.set_aspect("equal")
        ax.axis("off")

    for col, wn in enumerate([8, 16, 30]):
        ax = axes[2, col]
        try:
            polylines, positions, types, cnt = _gen_roadgen_geometry(wn)
            for pts in polylines:
                ax.plot(pts[:, 0], pts[:, 1], color="#55A868", lw=LANE_WIDTH, alpha=LANE_ALPHA)
            for wid, pos in positions.items():
                t = types.get(wid, "")
                c = WIDGET_COLORS.get(t, "#999")
                ax.scatter(pos[0], pos[1], c=c, s=60, edgecolors="white", linewidths=1.0, zorder=3)
            ax.set_title(f"{cnt} widgets", fontsize=10)
        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Error:\n{e}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
                color="#999",
            )
        if col == 0:
            ax.set_ylabel(baseline_labels[2], fontsize=12, fontweight="bold")
        ax.set_aspect("equal")
        ax.axis("off")

    for col, label in enumerate(size_labels):
        axes[0, col].set_title(label, fontsize=11, fontweight="bold", pad=10)

    plt.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [Grid] {output.name}")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    print("=" * 60)
    print("Unified Baseline Visualization — Road Geometry")
    print("=" * 60)

    print("\n--- MetaDrive ---")
    for label, seed, mc in [("Small", 42, 2), ("Medium", 43, 5), ("Large", 44, 12)]:
        out = REPO / "runtimes" / "metadrive_eval" / f"unified_{label.lower()}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        draw_metadrive_map(seed, mc, label, out)

    print("\n--- HDMapGen ---")
    for label, target in [("Small", 10), ("Medium", 30), ("Large", 69)]:
        entries = find_hdmapgen_graphs(target)
        out = REPO / "runtimes" / "hdmapgen_eval" / f"unified_{label.lower()}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        if entries:
            draw_hdmapgen_map(entries[0], label, out)
        else:
            print(f"  [HDMapGen] No data for {label} — hdmapgen.py still running?")

    print("\n--- RoadGen ---")
    for label, wn in [("Small", 8), ("Medium", 16), ("Large", 30)]:
        out = REPO / "runtimes" / "roadgen_eval" / f"unified_{label.lower()}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        draw_roadgen_map(wn, label, out)

    print("\n--- Comparison Grid ---")
    (REPO / "analysis" / "figures").mkdir(parents=True, exist_ok=True)
    draw_comparison_grid(REPO / "analysis" / "figures" / "unified_comparison.png")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


def run_eval_results():
    """Run cross-baseline metrics comparison (from viz_eval_results.py)."""
    # This function is defined inline from analysis/viz_eval_results.py
    import json as _json

    # ... the content is appended at the end of this file
    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Baseline comparison visualizations")
    parser.add_argument(
        "mode",
        nargs="?",
        default="unified",
        choices=["unified", "metrics", "all"],
        help="unified=sample maps, metrics=eval results comparison, all=both",
    )
    args = parser.parse_args()

    if args.mode in ("unified", "all"):
        print("=== Unified sample maps ===")
        main()
    if args.mode in ("metrics", "all"):
        print("=== Evaluation metrics comparison ===")
        # Run the eval_results visualization inline
        exec(open(__file__).read().split("# =========================================")[-1])


# =========================================================================
#  Cross-baseline metrics comparison (merged from analysis/viz_eval_results.py)
# =========================================================================

#!/usr/bin/env python3
"""
Visualize HDMapGen + MetaDrive evaluation results.

Usage:
    conda activate road-weaver
    python analysis/viz_eval_results.py

Output:
    analysis/figures/
        scalability.png           — Generation time vs node count
        metrics_comparison.png    — Bar chart: topology + route
        geometry_comparison.png   — Bar chart: all geometric metrics
        hdmapgen_scaling.png      — HDMapGen internal metrics vs node count
        sample_maps.png           — Sample generated HDMapGen maps
"""

import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO = Path(__file__).resolve().parent.parent
FIGS_DIR = REPO / "analysis" / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
#  Load results
# ═══════════════════════════════════════════════════════════════════════════


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_hdmapgen_data():
    """Load HDMapGen per-map data from saved JSONs."""
    base = REPO / "runtimes" / "hdmapgen_eval"
    topo = load_json(base / "topology.json")
    route = load_json(base / "route_coverage.json")
    geo = load_json(base / "geometry.json")
    scal = load_json(base / "scalability_json.json")
    return topo, route, geo, scal


def load_metadrive_data():
    """Load MetaDrive per-map data from saved JSONs."""
    base = REPO / "runtimes" / "metadrive_eval"
    topo = load_json(base / "topology.json")
    route = load_json(base / "route_coverage.json")
    geo = load_json(base / "geometry.json")
    scal = load_json(base / "scalability_json.json")
    return topo, route, geo, scal


# ═══════════════════════════════════════════════════════════════════════════
#  Plot styles
# ═══════════════════════════════════════════════════════════════════════════

COLORS = {"HDMapGen": "#E74C3C", "MetaDrive": "#3498DB"}
plt.rcParams.update(
    {"font.family": "sans-serif", "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11}
)


# ═══════════════════════════════════════════════════════════════════════════
#  1. Scalability: Gen time vs Node count
# ═══════════════════════════════════════════════════════════════════════════


def plot_scalability(ax, hd_topo, md_topo, hd_scal, md_scal):
    """Generation time vs node count for both baselines."""
    # HDMapGen scalability
    sizes_hd = hd_scal.get("results", [])
    if sizes_hd:
        ns_hd = [s["target_nodes"] for s in sizes_hd]
        times_hd = [s["aggregate"]["gen_time_mean"] for s in sizes_hd]
        err_hd = [s["aggregate"]["gen_time_std"] for s in sizes_hd]
        ax.errorbar(
            ns_hd,
            times_hd,
            yerr=err_hd,
            fmt="o-",
            color=COLORS["HDMapGen"],
            label="HDMapGen",
            capsize=3,
            markersize=4,
        )

    # MetaDrive scalability
    sizes_md = md_scal.get("results", [])
    if sizes_md:
        ns_md = [s["aggregate"]["node_count_mean"] for s in sizes_md]
        times_md = [s["aggregate"]["gen_time_mean"] for s in sizes_md]
        err_md = [s["aggregate"]["gen_time_std"] for s in sizes_md]
        ax.errorbar(
            ns_md,
            times_md,
            yerr=err_md,
            fmt="s--",
            color=COLORS["MetaDrive"],
            label="MetaDrive",
            capsize=3,
            markersize=4,
        )

    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("Generation Time (s)")
    ax.set_title("Large-Scale Capability")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)


# ═══════════════════════════════════════════════════════════════════════════
#  2. Metrics Comparison (Topology + Route)
# ═══════════════════════════════════════════════════════════════════════════


def plot_metrics_comparison(ax, hd_topo, md_topo, hd_route, md_route):
    """Bar chart comparing topology and route metrics."""
    cats = ["LCC ↑", "Dead-end ↓", "Δd̄ ↓", "Reachable ↑", "Route Len ↑"]
    hd_vals = [
        hd_topo["aggregate"]["lcc_mean"],
        hd_topo["aggregate"]["dead_end_ratio_mean"],
        hd_topo["aggregate"]["delta_avg_degree_mean"],
        hd_route["aggregate"]["reachable_ratio_mean"],
        hd_route["aggregate"]["avg_route_length_mean"],
    ]
    md_vals = [
        md_topo["aggregate"]["lcc_mean"],
        md_topo["aggregate"]["dead_end_ratio_mean"],
        md_topo["aggregate"]["delta_avg_degree_mean"],
        md_route["aggregate"]["reachable_ratio_mean"],
        md_route["aggregate"]["avg_route_length_mean"],
    ]

    x = np.arange(len(cats))
    w = 0.35
    bars1 = ax.bar(x - w / 2, hd_vals, w, label="HDMapGen", color=COLORS["HDMapGen"], alpha=0.85)
    bars2 = ax.bar(x + w / 2, md_vals, w, label="MetaDrive", color=COLORS["MetaDrive"], alpha=0.85)

    # Annotate values on bars
    for bar, val in zip(bars1, hd_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=45,
        )
    for bar, val in zip(bars2, md_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=45,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_title("Topology & Route Coverage Comparison")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3, axis="y")


# ═══════════════════════════════════════════════════════════════════════════
#  3. Geometric Metrics Comparison
# ═══════════════════════════════════════════════════════════════════════════


def plot_geometry_comparison(ax, hd_geo, md_geo):
    """Bar chart comparing geometric metrics (normalized where needed)."""
    # Select comparable metrics
    metric_keys = [
        ("chamfer_self", "Chamfer (self) ↓"),
        ("endpoint_alignment", "Endpoint Align ↓"),
        ("mean_turning_angle_deg", "Turning Angle ↓"),
        ("cv_edge_length", "Length CV ↓"),
        ("mean_spacing_cv", "Subnode Uniformity ↓"),
        ("intersection_rate", "Intersect Rate ↓"),
    ]

    hd_vals = []
    md_vals = []
    labels = []

    for key, label in metric_keys:
        hk = f"{key}_mean"
        mk = f"{key}_mean"
        if hk in hd_geo["aggregate"] and mk in md_geo["aggregate"]:
            hd_vals.append(hd_geo["aggregate"][hk])
            md_vals.append(md_geo["aggregate"][mk])
            labels.append(label)

    x = np.arange(len(labels))
    w = 0.35

    bars1 = ax.bar(x - w / 2, hd_vals, w, label="HDMapGen", color=COLORS["HDMapGen"], alpha=0.85)
    bars2 = ax.bar(x + w / 2, md_vals, w, label="MetaDrive", color=COLORS["MetaDrive"], alpha=0.85)

    for bar, val in zip(bars1, hd_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=45,
        )
    for bar, val in zip(bars2, md_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=45,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("Geometric Validity Comparison")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3, axis="y")


# ═══════════════════════════════════════════════════════════════════════════
#  4. HDMapGen internal scaling
# ═══════════════════════════════════════════════════════════════════════════


def plot_hdmapgen_scaling(ax, hd_topo):
    """How HDMapGen metrics change with node count."""
    per_map = hd_topo.get("per_map", [])
    if not per_map:
        ax.text(0.5, 0.5, "No per-map data", ha="center", va="center", transform=ax.transAxes)
        return

    # Group by node count
    by_n = {}
    for m in per_map:
        n = m["node_count"]
        by_n.setdefault(n, []).append(m)

    ns = sorted(by_n.keys())
    lccs = [np.mean([m["lcc"] for m in by_n[n]]) for n in ns]
    deads = [np.mean([m["dead_end_ratio"] for m in by_n[n]]) for n in ns]

    ax.plot(ns, lccs, "o-", color=COLORS["HDMapGen"], label="LCC", markersize=3)
    ax.plot(ns, deads, "s--", color="#E67E22", label="Dead-end", markersize=3)
    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("Ratio")
    ax.set_title("HDMapGen: Metrics vs Node Count")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)


# ═══════════════════════════════════════════════════════════════════════════
#  5. Sample maps visualization
# ═══════════════════════════════════════════════════════════════════════════


def plot_sample_hdmapgen_maps(axs):
    """Draw a few sample HDMapGen generated maps."""
    pkl_path = REPO / "runtimes" / "hdmapgen_eval" / "generated_graphs.pkl"
    if not pkl_path.exists():
        for ax in axs:
            ax.text(
                0.5, 0.5, "No generated graphs", ha="center", va="center", transform=ax.transAxes
            )
        return

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Select 4 samples: small (n~10), medium (n~30), medium-large (n~50), large (n~69)
    samples = []
    targets = [10, 30, 50, 69]
    for t in targets:
        for d in data:
            if d["num_nodes"] == t:
                samples.append(d)
                break

    for idx, (ax, entry) in enumerate(zip(axs, samples)):
        n = entry["num_nodes"]
        coords = entry.get("node_coords", [])
        edges = entry.get("edges", [])
        edge_sn = entry.get("edge_subnodes", {})

        if coords and coords[0] is not None:
            # Plot with node coordinates
            for u, v in edges:
                # Draw subnode curve if available
                if (u, v) in edge_sn and edge_sn[(u, v)] is not None:
                    pts = np.array(edge_sn[(u, v)])
                    ax.plot(pts[:, 0], pts[:, 1], "-", color="#E74C3C", lw=1.0, alpha=0.7)
                elif (v, u) in edge_sn and edge_sn[(v, u)] is not None:
                    pts = np.array(edge_sn[(v, u)])
                    ax.plot(pts[:, 0], pts[:, 1], "-", color="#E74C3C", lw=1.0, alpha=0.7)
                else:
                    # Straight edge
                    c_u = np.array(coords[u]) if coords[u] is not None else [0, 0]
                    c_v = np.array(coords[v]) if coords[v] is not None else [0, 0]
                    ax.plot(
                        [c_u[0], c_v[0]], [c_u[1], c_v[1]], "-", color="#E74C3C", lw=0.8, alpha=0.5
                    )

            # Plot nodes
            pts = np.array([c for c in coords if c is not None])
            if len(pts) > 0:
                ax.scatter(pts[:, 0], pts[:, 1], c="#2C3E50", s=15, zorder=3)
        else:
            # Fallback: circular layout
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            pos = {i: (np.cos(a), np.sin(a)) for i, a in enumerate(angles)}
            for u, v in edges:
                xs = [pos[u][0], pos[v][0]]
                ys = [pos[u][1], pos[v][1]]
                ax.plot(xs, ys, "-", color="#E74C3C", lw=0.8, alpha=0.5)
            xs = [pos[i][0] for i in range(n)]
            ys = [pos[i][1] for i in range(n)]
            ax.scatter(xs, ys, c="#2C3E50", s=15, zorder=3)

        ax.set_title(f"n={n}, edges={len(edges)}", fontsize=10)
        ax.set_aspect("equal")
        ax.axis("off")

    axs[0].figure.suptitle("HDMapGen Generated Road Network Samples", fontsize=13, y=1.02)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    print("=" * 60)
    print("Visualization: Evaluation Results")
    print("=" * 60)

    # Load data (handle missing files gracefully)
    try:
        hd_topo, hd_route, hd_geo, hd_scal = load_hdmapgen_data()
        print(f"  [HDMapGen] loaded: topology, route, geometry, scalability")
    except FileNotFoundError as e:
        print(f"  [WARN] HDMapGen data missing: {e}")
        hd_topo = hd_route = hd_geo = hd_scal = None

    try:
        md_topo, md_route, md_geo, md_scal = load_metadrive_data()
        print(f"  [MetaDrive] loaded: topology, route, geometry, scalability")
    except FileNotFoundError as e:
        print(f"  [WARN] MetaDrive data missing: {e}")
        md_topo = md_route = md_geo = md_scal = None

    # Figure 1: Scalability
    if hd_scal and md_scal:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_scalability(ax, hd_topo, md_topo, hd_scal, md_scal)
        fig.tight_layout()
        fig.savefig(FIGS_DIR / "scalability.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Saved] analysis/figures/scalability.png")

    # Figure 2: Metrics comparison
    if hd_topo and md_topo and hd_route and md_route:
        fig, ax = plt.subplots(figsize=(9, 5))
        plot_metrics_comparison(ax, hd_topo, md_topo, hd_route, md_route)
        fig.tight_layout()
        fig.savefig(FIGS_DIR / "metrics_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Saved] analysis/figures/metrics_comparison.png")

    # Figure 3: Geometry comparison
    if hd_geo and md_geo:
        fig, ax = plt.subplots(figsize=(9, 5))
        plot_geometry_comparison(ax, hd_geo, md_geo)
        fig.tight_layout()
        fig.savefig(FIGS_DIR / "geometry_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Saved] analysis/figures/geometry_comparison.png")

    # Figure 4: HDMapGen scaling
    if hd_topo:
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_hdmapgen_scaling(ax, hd_topo)
        fig.tight_layout()
        fig.savefig(FIGS_DIR / "hdmapgen_scaling.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Saved] analysis/figures/hdmapgen_scaling.png")

    # Figure 5: Sample maps
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    plot_sample_hdmapgen_maps(axs.ravel())
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "sample_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] analysis/figures/sample_maps.png")

    print(f"\nAll figures saved to {FIGS_DIR}/")


if __name__ == "__main__":
    main()

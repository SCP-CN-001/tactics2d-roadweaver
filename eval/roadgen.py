#!/usr/bin/env python3
"""
RoadGen baseline evaluation — compatible with eval/metadrive.py outputs.

RoadGen: rule-based road network generator using component (widget) assembly.
Each widget (straight lane, curve, intersection, fork, etc.) is placed by a
CountBased algorithm that prefers under-used component types.

Sections (matching the paper experiments):
  1. Topological Validity   (Table 1)  — LCC, Dead-end Ratio, Δd̄
  2. Route-Level Coverage   (Table 3)  — Reachable Ratio, Avg Route Length
  3. Large-Scale Capability (Figure)   — Generation time vs node (widget) count

Usage:
    conda activate road-weaver
    python eval/roadgen.py --all                    # run all sections
    python eval/roadgen.py --topology               # topological validity only
    python eval/roadgen.py --route                  # route coverage only
    python eval/roadgen.py --scalability            # scalability only
    python eval/roadgen.py --topology --num_maps 50

Output:
    runtimes/roadgen_eval/
        topology.json          — topological validity results
        route_coverage.json    — route coverage results
        scalability.csv        — CSV for plotting

Notes:
    - RoadGen runs from the RoadGen/python/ directory (its imports rely on it).
    - Global class-level counters (Widget.LaneID, Info.widgetcount, etc.)
      are reset between every map generation.
    - Graph extraction pipeline: lane polylines from each widget are extracted
      via ``getlanepoint()`` / ``getLaneInfoList()``, rendered as a binary
      road mask at 1024×1024, skeletonised, and converted to an intersection-
      level ``nx.Graph`` with coordinate attributes.
    - No pruning is applied — all skeleton branches are preserved so that
      evaluation metrics reflect the raw generated road network faithfully.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import io
import os
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
from tqdm import tqdm

from eval.metrics import (
    append_csv_row,
    classify_scale,
    compute_all_geometric_metrics,
    compute_cycle_ratio,
    compute_route_coverage,
    compute_topological_metrics,
    load_csv_keys,
    load_csv_rows,
    load_osm_reference_degree,
    monitor_resources,
    save_binned_summary,
)
from eval.polyline_graph import save_vis

# ═══════════════════════════════════════════════════════════════════════════
#  Paths
# ═══════════════════════════════════════════════════════════════════════════

ROOT = Path(__file__).resolve().parent.parent
ROADGEN_PY = ROOT / "baselines" / "RoadGen" / "python"
OUT_DIR = ROOT / "runtimes" / "roadgen_eval"

# ═══════════════════════════════════════════════════════════════════════════
#  Globals
# ═══════════════════════════════════════════════════════════════════════════

SILENT = False
"""When True, suppress RoadGen's verbose stdout during generation."""


@contextlib.contextmanager
def _silent_if_needed():
    """Suppress stdout when SILENT is True."""
    if SILENT:
        with contextlib.redirect_stdout(io.StringIO()):
            yield
    else:
        yield


# ═══════════════════════════════════════════════════════════════════════════
#  RoadGen interface
# ═══════════════════════════════════════════════════════════════════════════

_ROADGEN_LOADED = False
"""Whether RoadGen modules have been imported in this process."""


def _load_roadgen():
    """Import RoadGen modules (call once per process).

    NOTE: We do NOT import RoadGen/main.py because it has a syntax error
    (Chinese full-width exclamation mark at line 79 in RandomAlgorithm).
    Instead we import only the submodules we actually use.
    """
    global _ROADGEN_LOADED
    if _ROADGEN_LOADED:
        return

    # Change to RoadGen's python dir — its imports are relative to it
    if os.getcwd() != str(ROADGEN_PY):
        os.chdir(str(ROADGEN_PY))
    if str(ROADGEN_PY) not in sys.path:
        sys.path.insert(0, str(ROADGEN_PY))

    # Suppress matplotlib backend warnings from RoadGen's imports
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    import importlib

    # Clean slate for RoadGen modules
    _roadgen_prefixes = (
        "func.",
        "settings.",
        "straigntlane_",
        "Ulane_",
        "curve_",
        "laneswitch_",
        "fork_",
        "roundabout_",
        "Intersection_",
        "TJunction_",
        "ArcLane_",
    )
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith(_roadgen_prefixes):
            del sys.modules[mod_name]

    # Force-import RoadGen submodules we need.
    # These must be imported IN THIS ORDER because:
    #   - func.widget.Widget  is the base class (class-level counters)
    #   - settings.info.Info  holds widget libraries and compile rules
    #   - Widget types build on top of Widget
    #   - func.* helpers use Widget/Info
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from curve_widget.curve import Curve  # noqa: F401
        from fork_widget.fork import Fork  # noqa: F401
        from func.CountBasedChoose import CountBasedChoose  # noqa: F401
        from func.printAsserts import printAsserts  # noqa: F401
        from func.printAuto import printAutoInd  # noqa: F401
        from func.printConnection import printConnection  # noqa: F401
        from func.Quene import ConnectorQueue  # noqa: F401
        from func.update import initialFirstwidget, update  # noqa: F401
        from func.widget import Widget  # noqa: F401
        from func.WidgetGraph import Node, WidgetGraph  # noqa: F401
        from Intersection_widget.intersection import Intersection  # noqa: F401
        from laneswitch_widget.laneswitch import LaneSwitch  # noqa: F401
        from roundabout_widget.roundabout import Roundabout  # noqa: F401
        from settings.connectionDict import Widget_Map  # noqa: F401
        from settings.info import Info  # noqa: F401

        # Widget type modules (each subclass of Widget)
        from straigntlane_widget.straightlane import StraightLane  # noqa: F401
        from TJunction_widget.tJunction import tJunction  # noqa: F401
        from Ulane_widget.ulane import ULane  # noqa: F401

    _ROADGEN_LOADED = True


def _reset_roadgen_state():
    """Reset all global counters for a fresh generation."""
    from func.widget import Widget
    from settings.info import Info

    Widget.LaneID = 1
    Widget.BoundaryID = 1
    Widget.JunctionID = 1
    Widget.WidgetID = 1
    Info.parameterupdate = 0
    Info.widgetupdate = 0
    for key in Info.widgetcount:
        Info.widgetcount[key] = 0


def _collect_widget_polylines(widget) -> list[np.ndarray]:
    """Extract lane centerline polylines from any RoadGen widget.

    * **Road-type widgets** (straightlane, curve, ulane, laneswitch, fork,
      arclane): use ``getlanepoint()`` for full lane geometry.

    * **Junction widgets** (intersection, tJunction, roundabout): create
      connecting polylines from ``Start`` (entry point) to each exit endpoint
      returned by ``get_Nexts()``.  This fills the internal geometry gap
      through the junction.
    """
    # Road-type widgets: full lane polylines with detailed geometry
    if hasattr(widget, "getlanepoint"):
        try:
            raw = widget.getlanepoint()
            out = []
            for lane in raw:
                pts = np.array(lane, dtype=np.float64)
                if len(pts) >= 2:
                    out.append(pts)
            return out
        except Exception:
            return []

    # Junction widgets (no getlanepoint): create connecting lines
    # from the entry point (Start) to each exit (get_Nexts endpoint).
    polylines: list[np.ndarray] = []
    start = getattr(widget, "Start", None)
    if start is not None and hasattr(widget, "get_Nexts"):
        try:
            for nxt in widget.get_Nexts():
                ep = nxt.get("endpoint")
                if ep is not None and hasattr(ep, "__len__"):
                    polylines.append(np.array([start, ep], dtype=np.float64))
        except Exception:
            pass

    return polylines


def _build_widget(widgetdict: dict):
    """Factory: instantiate the correct widget class from a widget dict."""
    from curve_widget.curve import Curve
    from fork_widget.fork import Fork
    from Intersection_widget.intersection import Intersection
    from laneswitch_widget.laneswitch import LaneSwitch
    from roundabout_widget.roundabout import Roundabout
    from straigntlane_widget.straightlane import StraightLane
    from TJunction_widget.tJunction import tJunction
    from Ulane_widget.ulane import ULane

    t = widgetdict.get("Type")
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
    cls = builders.get(t)
    if cls is None:
        raise ValueError(f"Unknown widget type: {t}")
    return cls(widgetdict)


def generate_one_map(
    widget_number: int = 8, temp_dir: str | Path | None = None, mask_resolution: int = 1024
) -> tuple[nx.Graph | None, float, list[np.ndarray]]:
    """Run RoadGen and return ``(graph, gen_time_s, polylines)``.

    *polylines* is the list of ``(N_i, 2)`` lane centreline arrays used for
    graph extraction — kept so the caller can re-render for visualisation
    without re-running generation.
    """
    """Run RoadGen CountBasedAlgorithm once.

    Returns (networkx Graph, generation_time_s) or (None, NaN) on failure.

    The graph is extracted by rendering all widget lane polylines as a binary
    mask at ``mask_resolution × mask_resolution``, then skeletonising + graph
    extraction via ``field_to_graph()``.  No pruning is applied.

    Parameters
    ----------
    widget_number : int
        Target number of widgets (components) per map.  RoadGen default is 8.
    temp_dir : Path, optional
        Scratch directory for .m / .rrhd / .pkl files RoadGen insists on writing.
    mask_resolution : int
        Raster resolution for the binary road mask (default 1024).
    """
    if temp_dir is None:
        temp_dir = ROOT / ".roadgen_tmp"
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    _load_roadgen()
    _reset_roadgen_state()

    from func.widget import Widget as Wgt
    from settings.info import Info

    Info.WidgetNumber = widget_number

    # RoadGen CountBasedAlgorithm expects 4 positional args + the graph list
    count = 1
    mfile = str(temp_dir / f"rg_eval_{count}.m")
    graphlst: list = []

    # ── Inline the core of CountBasedAlgorithm ──────────────────────
    # We inline rather than import from main.CountBasedAlgorithm so we
    # have full control over state and can skip writing MATLAB files
    # when they're unwanted (though RoadGen still opens them).

    import json
    import random

    from func.CountBasedChoose import CountBasedChoose
    from func.printAsserts import printAsserts
    from func.printAuto import printAutoInd
    from func.printConnection import printConnection
    from func.Quene import ConnectorQueue
    from func.update import initialFirstwidget, update
    from func.WidgetGraph import WidgetGraph
    from settings.connectionDict import Widget_Map

    rules = Info.COMPILE_RULES
    widgetlist = Info.Widgetlist
    num = Info.WidgetNumber
    parametupdateercount = 0
    widgetupdatecount = 0
    encoding_format = Info.COMPILE_PREF.setdefault("M File Ecoding Format", "GBK")

    # ── Polyline collector ──────────────────────────────────────────
    all_polylines: list[np.ndarray] = []

    # Widget positions: widget_id → (x, y) Start point (for nx.Graph coords)
    widget_positions: dict = {}

    t0 = time.time()

    try:
        with _silent_if_needed():
            with open(mfile, "w", encoding=encoding_format) as f:
                graph = WidgetGraph()
                printAsserts(f)
                component_count = 0
                total_covered_area: list = []

                # ── First widget ──
                widgetdict = CountBasedChoose(widgetlist, Info.widgetcount)
                widgetdict0, parametupdateercount = initialFirstwidget(
                    widgetdict, rules, total_covered_area, parametupdateercount
                )
                widgetupdatecount += 1

                w0 = _build_widget(widgetdict0)
                w0.generate_road(f)
                component_count += 1
                total_covered_area += w0.get_coveredArea()
                Info.widgetcount[json.dumps(widgetdict)] += 1

                all_polylines.extend(_collect_widget_polylines(w0))
                widget_positions[w0.WidgetID] = widgetdict0.get("Start", [0.0, 0.0])

                graph.add_node(
                    w0.WidgetID,
                    widgetdict0.get("Type"),
                    widgetdict0.get("Flag"),
                    widgetdict0.setdefault("Function", None),
                    widgetdict0.setdefault("Direction", None),
                )

                connector_queue = ConnectorQueue()
                for nxt in w0.get_Nexts():
                    connector_queue.enqueue(nxt)
                del w0

                # ── Subsequent widgets ──
                while component_count < num and not connector_queue.isempty():
                    connector = connector_queue.dequeue()
                    flag = random.randint(0, 1)
                    if connector_queue.isempty() or flag == 1:
                        connector_type = connector["type"]
                        all_widgets = list(Widget_Map.get(connector_type, []))
                        available = len(all_widgets)

                        while available > 0:
                            wdict = CountBasedChoose(all_widgets, Info.widgetcount)
                            widgetupdatecount += 1
                            d = wdict.copy()
                            d["Start"] = connector["endpoint"]
                            d["K"] = connector.get("direction", 0.0)
                            d, parametupdateercount = update(
                                d, rules, total_covered_area, parametupdateercount
                            )
                            if d is not None:
                                break
                            all_widgets.remove(wdict)
                            available -= 1

                        if available == 0:
                            continue

                        w1 = _build_widget(d)
                        w1.generate_road(f)
                        component_count += 1
                        total_covered_area += w1.get_coveredArea()
                        Info.widgetcount[json.dumps(wdict)] += 1

                        # ── Collect lane polylines for graph extraction ──
                        all_polylines.extend(_collect_widget_polylines(w1))
                        widget_positions[w1.WidgetID] = d.get("Start", [0.0, 0.0])

                        graph.add_node(
                            w1.WidgetID,
                            d.get("Type"),
                            d.get("Flag"),
                            d.setdefault("Function", None),
                            d.setdefault("Direction", None),
                        )
                        graph.add_edge(connector["ID"], w1.WidgetID)
                        for nxt in w1.get_Nexts():
                            connector_queue.enqueue(nxt)
                        if len(connector["lanes"]) == len(w1.get_Currents()["CurrentLanes"]):
                            printConnection(
                                connector["lanes"],
                                w1.get_Currents()["CurrentLanes"],
                                connector_type,
                                f,
                            )
                        del w1

                graphlst.append(graph)

        gen_time = time.time() - t0

        # Reset widget-level IDs for next map (prevent overflow)
        Wgt.LaneID = 1
        Wgt.BoundaryID = 1
        Wgt.JunctionID = 1
        Wgt.WidgetID = 1

        if not graphlst or not all_polylines:
            return None, float("nan")

        # ── Build graph from RoadGen's own WidgetGraph ──────────────────
        # Each widget is one node, each widget connection is one edge.
        # This is the natural "component count" semantics of RoadGen and
        # avoids skeleton spurs / over-clustering from lane-based counting.
        widget_graph = graphlst[0]
        G = nx.Graph()
        for wid, node in widget_graph.nodes.items():
            pos = widget_positions.get(wid, [0.0, 0.0])
            G.add_node(wid, coords=np.array(pos, dtype=np.float64), widget_type=node.type)
        for nid, node in widget_graph.nodes.items():
            for nb, _direction in widget_graph.edges[node]:
                if nb.id in G:
                    G.add_edge(nid, nb.id)

        if G.number_of_nodes() == 0:
            return None, gen_time, all_polylines

        return G, gen_time, all_polylines

    except Exception as e:
        gen_time = time.time() - t0
        print(f"  [WARN] Map generation failed ({type(e).__name__}): {e}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        return None, gen_time, []


# ═══════════════════════════════════════════════════════════════════════════
#  Visualization: --plot-samples
# ═══════════════════════════════════════════════════════════════════════════

_WIDGET_COLORS = {
    "straightlane": "#4C72B0",
    "curve": "#DD8452",
    "ulane": "#55A868",
    "laneswitch": "#C44E52",
    "fork": "#937860",
    "intersection": "#8172B3",
    "tJunction": "#937860",
    "roundabout": "#CCB974",
}

_WIDGET_LABELS = {
    "straightlane": "Straight",
    "curve": "Curve",
    "ulane": "U-Lane",
    "laneswitch": "LaneSwitch",
    "fork": "Fork",
    "intersection": "Intersection",
    "tJunction": "T-Junction",
    "roundabout": "Roundabout",
}


def main():
    parser = argparse.ArgumentParser(
        description="RoadGen baseline evaluation — all-metrics binned by node count.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Examples:\n" "  python eval/roadgen.py\n" "  python eval/roadgen.py --all\n"),
    )

    # --all is the only evaluation mode (kept for backward compatibility).
    parser.add_argument("--all", action="store_true", help="Run all sections (default)")

    # Configuration
    parser.add_argument(
        "--output", type=str, default=str(OUT_DIR), help=f"Output directory (default: {OUT_DIR})"
    )
    parser.add_argument(
        "--silent", action="store_true", help="Suppress RoadGen's verbose stdout during generation"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Binary mask resolution for skeleton graph extraction (default: 1024)",
    )
    parser.add_argument(
        "--vis", type=str, default=None, help="Save mask+graph visualisation images to DIR"
    )

    args = parser.parse_args()
    output_dir = Path(args.output).resolve()
    vis_dir = Path(args.vis).resolve() if args.vis else None

    # Global settings
    global SILENT
    SILENT = args.silent

    # The all-metrics binned evaluation is the only mode.
    run_all = True
    if vis_dir is None:
        vis_dir = output_dir / "vis"  # absolute, survives _load_roadgen() chdir

    if run_all:
        output_dir.mkdir(parents=True, exist_ok=True)
        ref_deg = load_osm_reference_degree()
        target_bins = list(range(10, 61, 5))  # [10,15,...,60]
        bin_counts = {b: 0 for b in target_bins}

        def _bin(n):
            return min((n // 5) * 5, 60)  # bins [10,15),[15,20),...,[55,60]; clamp at 60

        # With WidgetGraph counting, node count == widget count.
        # widget_number spans 6 → 62 to cover bins 10-60 (floor-binned).
        params = [
            (10, "small"),  # 10 widgets  → bin 10
            (14, "medium"),  # 14 widgets  → bin 10
            (18, "medium"),  # 18 widgets  → bin 15
            (22, "medium"),  # 22 widgets  → bin 20
            (26, "large"),  # 26 widgets  → bin 25
            (30, "large"),  # 30 widgets  → bin 30
            (34, "large"),  # 34 widgets  → bin 30
            (38, "large"),  # 38 widgets  → bin 35
            (42, "large"),  # 42 widgets  → bin 40
            (46, "large"),  # 46 widgets  → bin 45
            (50, "large"),  # 50 widgets  → bin 50
            (54, "large"),  # 54 widgets  → bin 50
            (58, "large"),  # 58 widgets  → bin 55
            (62, "large"),  # 62 widgets  → bin 60
        ]

        # ── Resume: load existing CSV ─────────────────────────────────────
        csv_path = output_dir / "all_metrics.csv"
        seen_keys = load_csv_keys(csv_path, ["wn", "attempt"])
        if seen_keys:
            print(f"  Resuming: {len(seen_keys)} existing maps loaded from {csv_path.name}")

        existing_count = 0
        if csv_path.exists():
            with open(csv_path) as _f:
                existing_count = max(0, sum(1 for _ in _f) - 1)

        per_map = []
        t0 = time.time()
        for wn, _label in params:
            consecutive_stall = 0
            for i in range(30):  # attempts per widget_number
                key = f"{wn}|{i}"
                if key in seen_keys:
                    continue

                if all(c >= 10 for c in bin_counts.values()):
                    break
                try:
                    G, gt, polylines = generate_one_map(
                        widget_number=wn, mask_resolution=args.resolution
                    )
                except Exception:
                    consecutive_stall += 1
                    if consecutive_stall >= 15:
                        print(f"    wn={wn}: stall ({consecutive_stall} failures), skipping")
                        break
                    continue
                if G is None or G.number_of_nodes() < 2:
                    consecutive_stall += 1
                    if consecutive_stall >= 15:
                        print(f"    wn={wn}: stall ({consecutive_stall} failures), skipping")
                        break
                    continue
                nb = _bin(G.number_of_nodes())
                if nb < target_bins[0]:  # skip small maps (bin 5)
                    consecutive_stall += 1
                    continue
                if nb > target_bins[-1]:  # skip maps above 80-node target
                    consecutive_stall += 1
                    if consecutive_stall >= 15:
                        print(
                            f"    wn={wn}: stall ({consecutive_stall} attempts "
                            f"no progress), skipping"
                        )
                        break
                    continue
                if nb in bin_counts and bin_counts[nb] >= 10:
                    consecutive_stall += 1
                    if consecutive_stall >= 15:  # RoadGen is slow, abort sooner
                        print(
                            f"    wn={wn}: stall ({consecutive_stall} attempts "
                            f"no progress), skipping"
                        )
                        break
                    continue

                # ── This map contributes to a bin → reset stall ──
                consecutive_stall = 0

                # ── Resource monitoring during metric computation ───
                with monitor_resources(interval=0.3) as peaks:
                    topo = compute_topological_metrics(G, ref_avg_degree=ref_deg)
                    route = compute_route_coverage(G, n_pairs=100, seed=i)
                    cycle = compute_cycle_ratio(G)
                    geo = compute_all_geometric_metrics(polylines, G=G)

                scale = classify_scale(topo["node_count"])

                row = {
                    "map_id": existing_count + len(per_map),
                    "wn": wn,
                    "attempt": i,
                    "node_bin": nb,
                    "n_nodes": topo["node_count"],
                    "n_edges": topo["edge_count"],
                    **{k: topo[k] for k in ("lcc", "dead_end_ratio", "avg_degree")},
                    **{k: route[k] for k in ("reachable_ratio",)},
                    **{k: cycle[k] for k in ("cycle_ratio",)},
                    **{
                        k: geo.get(k, "")
                        for k in (
                            "chamfer_loo",
                            "chamfer_loo_std",
                            "endpoint_alignment",
                            "mean_turning_angle_deg",
                            "crossings_per_km",
                            "mean_edge_length",
                            "cv_edge_length",
                            "total_road_length",
                            "mean_spacing_cv",
                            "mean_angle_deg",
                        )
                    },
                    "gen_time_s": round(gt, 4),
                    "cpu_peak": round(peaks["cpu_peak"], 1),
                    "mem_peak_mb": round(peaks["mem_peak_mb"], 1),
                    "gpu_peak_mb": round(peaks["gpu_peak_mb"], 1),
                }
                per_map.append(row)
                append_csv_row(csv_path, list(row.keys()), row)
                seen_keys.add(key)

                if nb in bin_counts:
                    bin_counts[nb] += 1

                if len(per_map) % 5 == 0:
                    filled = sum(1 for c in bin_counts.values() if c >= 10)
                    print(
                        f"  [{existing_count + len(per_map)} maps] bins filled: {filled}/{len(target_bins)}"
                    )

                # ── Vis ──
                if vis_dir is not None:
                    cnt = len(list(Path(vis_dir).glob(f"rg_{scale}_*.png")))
                    if cnt < 5:
                        save_vis(
                            polylines,
                            G,
                            str(
                                Path(vis_dir)
                                / f"rg_{scale}_N{G.number_of_nodes()}E{G.number_of_edges()}_{cnt}.png"
                            ),
                        )

                if all(c >= 10 for c in bin_counts.values()):
                    break

        # ── Per-bin aggregate summary (all maps incl. resumed) ──
        save_binned_summary(output_dir, "all_metrics", load_csv_rows(csv_path))

        gen_time = time.time() - t0
        n_total = existing_count + len(per_map)
        print(f"  Bin counts: {dict(bin_counts)}")
        print(
            f"  Generated {len(per_map)} new maps (→{n_total} total) in "
            f"{gen_time:.1f}s, {gen_time/max(len(per_map),1):.1f}s/map"
        )

    # Cleanup temp directory
    temp_dir = ROOT / ".roadgen_tmp"
    if temp_dir.exists():
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"\n{'=' * 60}")
    print(f"RoadGen evaluation complete. Results saved to {output_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

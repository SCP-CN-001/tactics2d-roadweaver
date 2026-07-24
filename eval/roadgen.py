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
        paper_tables/          — LaTeX table rows for copy-paste

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
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

from eval.metrics import (
    classify_scale,
    compute_route_coverage,
    compute_topological_metrics,
    load_osm_reference_degree,
    print_results_table,
    save_results,
    save_tex_row,
)
from eval.polyline_graph import polylines_to_graph, save_vis

# ═══════════════════════════════════════════════════════════════════════════
#  Paths
# ═══════════════════════════════════════════════════════════════════════════

ROOT = Path(__file__).resolve().parent.parent
ROADGEN_PY = ROOT / "RoadGen" / "python"
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

    def _add_bridge(ep):
        """Add a short bridge polyline at *ep* to close gaps at junction connections.

        The bridge is a cross-shaped set of line segments spanning ~6 units
        (RoadGen space), which at 1024×1024 renders as ~10 pixels — enough for
        the skeleton dilation to merge nearby road segments across a junction
        widget that does not provide ``getlanepoint()`` data.
        """
        ep_a = np.array(ep, dtype=np.float64)
        L = 3.0
        all_polylines.append(np.array([ep_a + [-L, 0.0], ep_a, ep_a + [L, 0.0]]))
        all_polylines.append(np.array([ep_a + [0.0, -L], ep_a, ep_a + [0.0, L]]))

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

                # Bridge gaps at the first widget's connectors
                for nxt in w0.get_Nexts():
                    _add_bridge(nxt["endpoint"])

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
                            d["K"] = connector["direction"]
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

                        # Bridge the gap at the connection point (critical for
                        # junction-widget neighbours that lack getlanepoint()).
                        _add_bridge(connector["endpoint"])

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

        # ── Convert lane polylines → binary mask → skeleton graph ──
        G = polylines_to_graph(
            all_polylines, resolution=mask_resolution, cleanup=False, merge_distance=3.0
        )

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
#  Section 1: Topological Validity  (Table 1)
# ═══════════════════════════════════════════════════════════════════════════


def run_topology_eval(
    num_maps: int = 200,
    seed: int = 0,
    widget_number: int = 8,
    output_dir: Path = OUT_DIR,
    mask_resolution: int = 1024,
    vis_dir: Path | None = None,
) -> dict:
    """Generate maps and compute topological validity metrics."""
    print(f"\n{'=' * 60}")
    print("Section 1: Topological Validity  (RoadGen)")
    print(
        f"  Generating {num_maps} maps ({widget_number} widgets each, "
        f"mask={mask_resolution}²)..."
    )
    print(f"{'=' * 60}")

    ref_deg = load_osm_reference_degree()
    print(f"  OSM reference avg_degree = {ref_deg:.4f}")

    topo_metrics: dict[str, list] = defaultdict(list)
    per_map: list[dict] = []
    vis_counts: dict[str, int] = {}

    t_start = time.time()
    for i in range(num_maps):
        G, gen_time, _polylines = generate_one_map(
            widget_number=widget_number, mask_resolution=mask_resolution
        )

        if G is None or G.number_of_nodes() == 0:
            print(f"    [{i + 1}/{num_maps}] SKIP (empty graph)")
            continue

        topo = compute_topological_metrics(G, ref_avg_degree=ref_deg)

        for k, v in topo.items():
            topo_metrics[k].append(v)

        scale = classify_scale(topo["node_count"])

        # ── Save vis (up to 5 per scale) ──
        if vis_dir is not None:
            cnt = vis_counts.get(scale, 0)
            if cnt < 5:
                vd = Path(vis_dir)
                vd.mkdir(parents=True, exist_ok=True)
                nc, ec = G.number_of_nodes(), G.number_of_edges()
                save_vis(
                    _polylines,
                    G,
                    str(vd / f"rg_{scale}_N{nc}E{ec}_{cnt}.png"),
                    resolution=mask_resolution,
                    title=f"RoadGen {scale}",
                )
                vis_counts[scale] = cnt + 1

        per_map.append({"map_id": i, "generation_time": round(gen_time, 4), "scale": scale, **topo})

        if (i + 1) % max(1, num_maps // 20) == 0 or i == 0:
            elapsed = time.time() - t_start
            done = i + 1
            if done > 0:
                eta_sec = (elapsed / done) * (num_maps - done)
                eta = str(datetime.timedelta(seconds=int(eta_sec)))
            else:
                eta = "?"
            print(
                f"    [{i + 1}/{num_maps}] "
                f"LCC={np.mean(topo_metrics['lcc']):.3f}  "
                f"dead={np.mean(topo_metrics['dead_end_ratio']):.3f}  "
                f"Δd̄={np.mean(topo_metrics['delta_avg_degree']):.3f}  "
                f"ETA {eta}"
            )

    if not per_map:
        print("  [ERROR] No valid maps generated. Aborting topology eval.")
        return {}

    # Aggregate
    agg: dict[str, float] = {}
    for metric_name, values in topo_metrics.items():
        agg[f"{metric_name}_mean"] = float(np.mean(values))
        agg[f"{metric_name}_std"] = float(np.std(values))

    agg["n_maps"] = len(per_map)
    agg["widget_number"] = widget_number
    agg["avg_generation_time"] = float(np.mean([m["generation_time"] for m in per_map]))
    agg["osm_reference_avg_degree"] = ref_deg

    save_results(output_dir, "topology", agg, per_map)
    save_tex_row(
        output_dir,
        "topological-validity",
        "RoadGen",
        agg,
        fields=[("lcc", ".3f"), ("dead_end_ratio", ".3f"), ("delta_avg_degree", ".3f")],
    )
    print_results_table(
        "Topological Validity (RoadGen)",
        agg,
        [
            ("lcc", "LCC", ".4f"),
            ("dead_end_ratio", "Dead-end", ".4f"),
            ("delta_avg_degree", "Δd̄", ".4f"),
            ("node_count", "Avg nodes", ".0f"),
        ],
    )

    return agg


# ═══════════════════════════════════════════════════════════════════════════
#  Section 2: Route-Level Coverage  (Table 3)
# ═══════════════════════════════════════════════════════════════════════════


def run_route_eval(
    num_maps: int = 200,
    seed: int = 0,
    widget_number: int = 8,
    n_pairs: int = 100,
    output_dir: Path = OUT_DIR,
    mask_resolution: int = 1024,
) -> dict:
    """Generate maps and compute route coverage metrics."""
    print(f"\n{'=' * 60}")
    print("Section 2: Route-Level Coverage  (RoadGen)")
    print(
        f"  Generating {num_maps} maps ({widget_number} widgets each, "
        f"mask={mask_resolution}²), {n_pairs} OD pairs per map..."
    )
    print(f"{'=' * 60}")

    route_metrics: dict[str, list] = defaultdict(list)
    per_map: list[dict] = []

    t_start = time.time()
    for i in range(num_maps):
        G, gen_time, _ = generate_one_map(
            widget_number=widget_number, mask_resolution=mask_resolution
        )

        if G is None or G.number_of_nodes() < 2:
            print(f"    [{i + 1}/{num_maps}] SKIP (< 2 nodes)")
            continue

        route = compute_route_coverage(G, n_pairs=n_pairs, seed=seed + i)
        for k, v in route.items():
            route_metrics[k].append(v)

        per_map.append(
            {
                "map_id": i,
                "node_count": G.number_of_nodes(),
                "edge_count": G.number_of_edges(),
                **route,
            }
        )

        if (i + 1) % max(1, num_maps // 20) == 0:
            elapsed = time.time() - t_start
            done = i + 1
            eta_sec = (elapsed / done) * (num_maps - done) if done > 0 else 0
            eta = str(datetime.timedelta(seconds=int(eta_sec)))
            print(
                f"    [{i + 1}/{num_maps}]  "
                f"reachable={np.mean(route_metrics['reachable_ratio']):.3f}  "
                f"avg_len={np.mean(route_metrics['avg_route_length']):.1f}  "
                f"ETA {eta}"
            )

    if not per_map:
        print("  [ERROR] No valid maps generated. Aborting route eval.")
        return {}

    # Aggregate
    agg: dict[str, float] = {}
    for metric_name, values in route_metrics.items():
        agg[f"{metric_name}_mean"] = float(np.mean(values))
        agg[f"{metric_name}_std"] = float(np.std(values))

    agg["n_maps"] = len(per_map)
    agg["widget_number"] = widget_number

    save_results(output_dir, "route_coverage", agg, per_map)
    save_tex_row(
        output_dir,
        "route-coverage",
        "RoadGen",
        agg,
        fields=[("reachable_ratio", ".3f"), ("avg_route_length", ".1f")],
    )
    print_results_table(
        "Route Coverage (RoadGen)",
        agg,
        [("reachable_ratio", "Reachable", ".4f"), ("avg_route_length", "Avg Length", ".1f")],
    )

    return agg


# ═══════════════════════════════════════════════════════════════════════════
#  Section 3: Large-Scale Capability  (Scalability Figure)
# ═══════════════════════════════════════════════════════════════════════════


def run_scalability_eval(
    n_per_size: int = 10,
    route_pairs: int = 100,
    output_dir: Path = OUT_DIR,
    mask_resolution: int = 1024,
) -> dict:
    """Generate maps at controlled sizes and record generation time + metrics.

    Varies widget_number from 4 to 40 in steps of 4, roughly giving
    linear scaling in node count.
    """
    print(f"\n{'=' * 60}")
    print("Section 3: Large-Scale Capability  (RoadGen)")
    print(f"  Widget counts: 4 → 40, {n_per_size} maps per size " f"(mask={mask_resolution}²)")
    print(f"{'=' * 60}")

    target_widgets = list(range(4, 41, 4))
    all_results: list[dict] = []

    for wc in target_widgets:
        per_size: dict = {"target_widgets": wc, "maps": []}

        for i in range(n_per_size):
            G, gen_time, _ = generate_one_map(widget_number=wc, mask_resolution=mask_resolution)

            if G is None:
                continue

            topo = compute_topological_metrics(G)
            route = compute_route_coverage(G, n_pairs=route_pairs, seed=i)

            per_size["maps"].append(
                {
                    "node_count": topo["node_count"],
                    "edge_count": topo["edge_count"],
                    "generation_time": round(gen_time, 4),
                    **topo,
                    **route,
                }
            )

        if not per_size["maps"]:
            print(f"    widgets={wc:2d}: all {n_per_size} failed, skipping")
            continue

        # Aggregate
        nodes = [m["node_count"] for m in per_size["maps"]]
        times = [m["generation_time"] for m in per_size["maps"]]
        lccs = [m["lcc"] for m in per_size["maps"]]
        deads = [m["dead_end_ratio"] for m in per_size["maps"]]
        degs = [m["avg_degree"] for m in per_size["maps"]]
        reachs = [m["reachable_ratio"] for m in per_size["maps"]]
        lengths = [m["avg_route_length"] for m in per_size["maps"]]

        per_size["aggregate"] = {
            "node_count_mean": float(np.mean(nodes)),
            "node_count_std": float(np.std(nodes)),
            "gen_time_mean": float(np.mean(times)),
            "gen_time_std": float(np.std(times)),
            "lcc_mean": float(np.mean(lccs)),
            "lcc_std": float(np.std(lccs)),
            "dead_end_ratio_mean": float(np.mean(deads)),
            "dead_end_ratio_std": float(np.std(deads)),
            "avg_degree_mean": float(np.mean(degs)),
            "avg_degree_std": float(np.std(degs)),
            "reachable_ratio_mean": float(np.mean(reachs)),
            "reachable_ratio_std": float(np.std(reachs)),
            "avg_route_length_mean": float(np.mean(lengths)),
            "avg_route_length_std": float(np.std(lengths)),
        }
        all_results.append(per_size)

        print(
            f"    widgets={wc:2d} → actual nodes={per_size['aggregate']['node_count_mean']:5.0f}±"
            f"{per_size['aggregate']['node_count_std']:4.1f}  "
            f"time={per_size['aggregate']['gen_time_mean']:.4f}s"
        )

    if not all_results:
        print("  [ERROR] No valid scalability data. Aborting.")
        return {}

    # Save JSON
    out = {"results": all_results}
    save_results(output_dir, "scalability", out, per_map=None)

    # Save CSV for plotting
    csv_path = output_dir / "scalability.csv"
    with open(csv_path, "w") as f:
        f.write(
            "target_widgets,"
            "actual_nodes_mean,actual_nodes_std,"
            "gen_time_mean,gen_time_std,"
            "lcc_mean,lcc_std,dead_end_ratio_mean,dead_end_ratio_std,"
            "avg_degree_mean,avg_degree_std,"
            "reachable_ratio_mean,reachable_ratio_std,"
            "avg_route_length_mean,avg_route_length_std\n"
        )
        for r in all_results:
            a = r["aggregate"]
            f.write(
                f"{r['target_widgets']},"
                f"{a['node_count_mean']},{a['node_count_std']},"
                f"{a['gen_time_mean']},{a['gen_time_std']},"
                f"{a['lcc_mean']},{a['lcc_std']},"
                f"{a['dead_end_ratio_mean']},{a['dead_end_ratio_std']},"
                f"{a['avg_degree_mean']},{a['avg_degree_std']},"
                f"{a['reachable_ratio_mean']},{a['reachable_ratio_std']},"
                f"{a['avg_route_length_mean']},{a['avg_route_length_std']}\n"
            )

    print(f"\n  CSV saved to {csv_path}")
    return {"n_sizes": len(all_results), "n_per_size": n_per_size}


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════


def _print_timing_estimate(args, run_all):
    """Print a rough wall-clock estimate before starting."""
    maps_topo = args.num_maps if (run_all or args.topology) else 0
    maps_route = args.num_maps if (run_all or args.route) else 0
    maps_scal = args.n_per_size * 10 if (run_all or args.scalability) else 0  # 10 sizes by default
    total_maps = maps_topo + maps_route + maps_scal

    if total_maps == 0:
        return

    # RoadGen is slow (~8-40s per map depending on geometric constraint solving)
    # Use a conservative 25s/map for the estimate
    est_seconds = total_maps * 25
    if est_seconds < 60:
        est_str = f"{est_seconds}s"
    elif est_seconds < 3600:
        est_str = f"{est_seconds // 60}m {est_seconds % 60}s"
    else:
        hours = est_seconds // 3600
        mins = (est_seconds % 3600) // 60
        est_str = f"{hours}h {mins}m"

    print(f"  [Note] RoadGen is computationally intensive (geometric overlap checks).")
    print(f"         Estimated wall-clock time for {total_maps} maps: ~{est_str}")
    print(f"         (actual time varies widely, ~3-100s per map).")
    print(f"         Use --silent to suppress RoadGen's verbose output.\n")


def main():
    parser = argparse.ArgumentParser(
        description="RoadGen baseline evaluation — topological validity, "
        "route coverage, and large-scale capability.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python eval/roadgen.py --topology --num_maps 50\n"
            "  python eval/roadgen.py --route --num_maps 100\n"
            "  python eval/roadgen.py --scalability --n_per_size 5\n"
        ),
    )

    # Section selection
    parser.add_argument("--all", action="store_true", help="Run all sections")
    parser.add_argument("--topology", action="store_true", help="Topological validity (Table 1)")
    parser.add_argument("--route", action="store_true", help="Route coverage (Table 3)")
    parser.add_argument(
        "--scalability", action="store_true", help="Large-scale capability (Figure)"
    )

    # Configuration
    parser.add_argument(
        "--num_maps", type=int, default=200, help="Number of maps for topology/route (default: 200)"
    )
    parser.add_argument(
        "--n_per_size",
        type=int,
        default=10,
        help="Maps per widget-count for scalability (default: 10)",
    )
    parser.add_argument(
        "--widget_number",
        type=int,
        default=8,
        help="Widgets (components) per map (default: 8, RoadGen default)",
    )
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
    output_dir = Path(args.output)
    vis_dir = Path(args.vis) if args.vis else None

    # Global settings
    global SILENT
    SILENT = args.silent

    # Default: run all if no section specified
    run_all = args.all or not (args.topology or args.route or args.scalability)

    _print_timing_estimate(args, run_all)

    if run_all or args.topology:
        run_topology_eval(
            args.num_maps,
            widget_number=args.widget_number,
            output_dir=output_dir,
            mask_resolution=args.resolution,
            vis_dir=vis_dir,
        )

    if run_all or args.route:
        run_route_eval(
            args.num_maps,
            widget_number=args.widget_number,
            output_dir=output_dir,
            mask_resolution=args.resolution,
        )

    if run_all or args.scalability:
        run_scalability_eval(
            args.n_per_size, output_dir=output_dir, mask_resolution=args.resolution
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

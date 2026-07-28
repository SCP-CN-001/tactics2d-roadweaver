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
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
from tqdm import tqdm

from eval.metrics import (
    append_csv_row,
    classify_scale,
    compute_all_geometric_metrics,
    compute_route_coverage,
    compute_topological_metrics,
    get_resource_stats,
    load_csv_keys,
    load_osm_reference_degree,
    monitor_resources,
    print_results_table,
    save_results,
    save_system_info,
)
from eval.polyline_graph import polylines_to_graph, save_vis

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
    bridge_polylines: list[np.ndarray] = []  # for mask rendering only, excluded from metrics

    def _add_bridge(ep):
        """Add a short bridge polyline at *ep* to close gaps at junction connections.

        The bridge is a cross-shaped set of line segments spanning ~6 units
        (RoadGen space), which at 1024×1024 renders as ~10 pixels — enough for
        the skeleton dilation to merge nearby road segments across a junction
        widget that does not provide ``getlanepoint()`` data.

        Bridges are kept separate from ``all_polylines`` so they are used for
        mask rendering and graph extraction but NOT for geometric metric
        evaluation.
        """
        ep_a = np.array(ep, dtype=np.float64)
        L = 3.0
        bridge_polylines.append(np.array([ep_a + [-L, 0.0], ep_a, ep_a + [L, 0.0]]))
        bridge_polylines.append(np.array([ep_a + [0.0, -L], ep_a, ep_a + [0.0, L]]))

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
        # NOTE: bridge_polylines are merged here for mask rendering so that
        #       skeletonised graph captures junction connectivity, but they
        #       are NOT returned to callers and therefore NOT passed to
        #       geometric metric functions.
        graph_polylines = all_polylines + bridge_polylines
        G = polylines_to_graph(
            graph_polylines, resolution=mask_resolution, cleanup=False, merge_distance=3.0
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

    for wc in tqdm(target_widgets, desc="Scaling"):
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
#  Section 4: Geometric Validity  (Table 2)  — RoadGen
# ═══════════════════════════════════════════════════════════════════════════


def run_geometry_eval(
    num_maps: int = 200,
    seed: int = 0,
    widget_number: int = 8,
    output_dir: Path = OUT_DIR,
    mask_resolution: int = 1024,
) -> dict:
    """Generate maps and compute all geometric metrics from lane polylines."""
    print(f"\n{'=' * 60}")
    print("Section 4: Geometric Validity  (RoadGen)")
    print(
        f"  Generating {num_maps} maps ({widget_number} widgets each, "
        f"mask={mask_resolution}²)..."
    )
    print(f"{'=' * 60}")

    geom_metrics: dict[str, list] = defaultdict(list)
    all_results: list[dict] = []

    for i in range(num_maps):
        G, gen_time, polylines = generate_one_map(
            widget_number=widget_number, mask_resolution=mask_resolution
        )
        if G is None or G.number_of_nodes() < 2:
            continue

        geo = compute_all_geometric_metrics(polylines, G=G)

        entry = {
            "map_id": i,
            "generation_time": round(gen_time, 4),
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            **geo,
        }
        for k, v in geo.items():
            if isinstance(v, (int, float, np.floating, np.integer)):
                geom_metrics[k].append(v)
        all_results.append(entry)

        if (i + 1) % max(1, num_maps // 10) == 0 or i == 0:
            print(
                f"    [{i + 1}/{num_maps}] "
                f"chamfer_loo={np.mean(geom_metrics['chamfer_loo']):.4f}"
            )

    if not all_results:
        print("  [ERROR] No valid maps generated. Aborting geometry eval.")
        return {}

    agg = {}
    for k, v in geom_metrics.items():
        agg[f"{k}_mean"] = float(np.mean(v))
        agg[f"{k}_std"] = float(np.std(v))
    agg["n_maps"] = len(all_results)

    save_results(output_dir, "geometry", agg, all_results)
    print_results_table(
        "Geometric Validity (RoadGen)",
        agg,
        [
            ("chamfer_loo", "Chamfer (LOO)", ".4f"),
            ("endpoint_alignment", "Endpoint Align", ".4f"),
            ("mean_turning_angle_deg", "Turning Angle", ".2f"),
            ("mean_edge_length", "Edge Length", ".4f"),
            ("cv_edge_length", "Length CV", ".3f"),
            ("intersection_rate", "Intersect Rate", ".6f"),
            ("mean_spacing_cv", "Subnode Uniformity", ".3f"),
        ],
    )

    return agg


# ═══════════════════════════════════════════════════════════════════════════
#  Visualization: --plot-summary
# ═══════════════════════════════════════════════════════════════════════════


def plot_summary(output_dir: Path = OUT_DIR):
    """Read existing evaluation results and produce a 6-panel summary figure
    and LaTeX table.

    Requires ``topology.json`` and ``route_coverage.json`` in *output_dir*
    (generated by ``--topology`` and ``--route`` runs respectively).

    Outputs
    -------
    output_dir/roadgen_eval_overview.png   --- 6-panel summary figure
    output_dir/roadgen_results.tex         --- LaTeX summary table
    """
    import json

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Load data
    topo_path = output_dir / "topology.json"
    route_path = output_dir / "route_coverage.json"
    if not topo_path.exists():
        print(f"[ERROR] {topo_path} not found. Run --topology first.")
        return
    if not route_path.exists():
        print(f"[ERROR] {route_path} not found. Run --route first.")
        return

    with open(topo_path) as f:
        topo = json.load(f)
    with open(route_path) as f:
        route = json.load(f)

    topo_maps = topo["per_map"]
    route_maps = route["per_map"]
    agg = topo["aggregate"]

    # Style
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.dpi": 300,
        }
    )

    C1, C2, C3, C4 = "#4C72B0", "#DD8452", "#55A868", "#C44E52"

    # --- 6-panel figure ---
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        "RoadGen Baseline Evaluation  (200 maps, 8 widgets each)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    # 1. Dead-end ratio histogram
    ax = axes[0, 0]
    d = [m["dead_end_ratio"] for m in topo_maps]
    ax.hist(d, bins=20, color=C1, edgecolor="white", alpha=0.85)
    ax.axvline(
        agg["dead_end_ratio_mean"],
        color="red",
        ls="--",
        label=f"Mean = {agg['dead_end_ratio_mean']:.3f}",
    )
    ax.set_xlabel("Dead-end Ratio")
    ax.set_ylabel("Number of Maps")
    ax.set_title("Dead-end Ratio Distribution")
    ax.legend(fontsize=8)

    # 2. Node count histogram
    ax = axes[0, 1]
    n_vals = [m["node_count"] for m in topo_maps]
    ax.hist(n_vals, bins=range(1, 16), color=C2, edgecolor="white", alpha=0.85, align="left")
    ax.axvline(
        agg["node_count_mean"], color="red", ls="--", label=f"Mean = {agg['node_count_mean']:.1f}"
    )
    ax.set_xlabel("Node Count (widgets)")
    ax.set_ylabel("Number of Maps")
    ax.set_title("Graph Size Distribution")
    ax.legend(fontsize=8)
    ax.set_xticks(range(1, 16))

    # 3. Delta avg degree histogram
    ax = axes[0, 2]
    dd = [m["delta_avg_degree"] for m in topo_maps]
    ax.hist(dd, bins=20, color=C3, edgecolor="white", alpha=0.85)
    ax.axvline(
        agg["delta_avg_degree_mean"],
        color="red",
        ls="--",
        label=f"Mean = {agg['delta_avg_degree_mean']:.3f}",
    )
    ax.set_xlabel("Delta d_bar (Deviation from OSM Avg Degree)")
    ax.set_ylabel("Number of Maps")
    ax.set_title("Topological Deviation Distribution")
    ax.legend(fontsize=8)

    # 4. Generation time vs node count (scatter)
    ax = axes[1, 0]
    times = [m["generation_time"] for m in topo_maps]
    ax.scatter(n_vals, times, alpha=0.5, s=15, color=C1)
    z = np.polyfit(n_vals, times, 1)
    p = np.poly1d(z)
    xs = sorted(set(n_vals))
    ax.plot(xs, p(xs), "r--", alpha=0.7, label=f"Trend: {z[0]:.1f}s/node")
    ax.set_xlabel("Node Count")
    ax.set_ylabel("Generation Time (s)")
    ax.set_title("Generation Time vs Graph Size")
    ax.legend(fontsize=8)

    # 5. Route length distribution
    ax = axes[1, 1]
    al = [m["avg_route_length"] for m in route_maps]
    ax.hist(al, bins=15, color=C4, edgecolor="white", alpha=0.85)
    ax.axvline(
        route["aggregate"]["avg_route_length_mean"],
        color="red",
        ls="--",
        label=f"Mean = {route['aggregate']['avg_route_length_mean']:.2f}",
    )
    ax.set_xlabel("Avg Route Length (widget hops)")
    ax.set_ylabel("Number of Maps")
    ax.set_title("Route Length Distribution")
    ax.legend(fontsize=8)

    # 6. Generation time histogram
    ax = axes[1, 2]
    ax.hist(times, bins=25, color=C1, edgecolor="white", alpha=0.85)
    ax.axvline(
        agg["avg_generation_time"],
        color="red",
        ls="--",
        label=f"Mean = {agg['avg_generation_time']:.1f}s",
    )
    ax.set_xlabel("Generation Time (s)")
    ax.set_ylabel("Number of Maps")
    ax.set_title("Generation Time Distribution")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / "roadgen_eval_overview.png", bbox_inches="tight")
    print(f"Saved plot to {output_dir / 'roadgen_eval_overview.png'}")

    # --- LaTeX table ---
    tex_path = output_dir / "roadgen_results.tex"
    with open(tex_path, "w") as f:
        f.write("% RoadGen evaluation results --- auto-generated\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write("Metric & Mean & Std \\\\\n")
        f.write("\\midrule\n")
        for key, label, fmt in [
            ("lcc", "LCC", ".4f"),
            ("dead_end_ratio", "Dead-end Ratio", ".4f"),
            ("delta_avg_degree", "$\\Delta\\bar{d}$", ".4f"),
            ("node_count", "Node Count", ".1f"),
            ("avg_generation_time", "Gen. Time (s)", ".1f"),
        ]:
            if key in agg:
                mean = agg.get(f"{key}_mean", agg.get(key, 0))
                std = agg.get(f"{key}_std", 0)
                f.write(f"{label} & {mean:{fmt}} & {std:{fmt}} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    print(f"Saved LaTeX table to {tex_path}")

    # --- Print stats ---
    print("\n--- RoadGen 200-map Summary ---")
    print(f"  LCC:              {agg['lcc_mean']:.4f} +/- {agg['lcc_std']:.4f}")
    print(
        f"  Dead-end Ratio:   {agg['dead_end_ratio_mean']:.4f} +/- {agg['dead_end_ratio_std']:.4f}"
    )
    print(
        f"  Delta d_bar:      {agg['delta_avg_degree_mean']:.4f} +/- {agg['delta_avg_degree_std']:.4f}"
    )
    print(f"  Avg Node Count:   {agg['node_count_mean']:.1f} +/- {agg['node_count_std']:.1f}")
    print(f"  Avg Gen Time:     {agg['avg_generation_time']:.1f}s")
    r_agg = route["aggregate"]
    print(
        f"  Reachable Ratio:  {r_agg['reachable_ratio_mean']:.4f} +/- {r_agg['reachable_ratio_std']:.4f}"
    )
    print(
        f"  Avg Route Length: {r_agg['avg_route_length_mean']:.2f} +/- {r_agg['avg_route_length_std']:.2f}"
    )


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


def _generate_with_geometry(widget_number=8, temp_dir=None):
    """Run RoadGen and return ``(widget_graph, positions, lane_data, gen_time, n_widgets)``.

    Like ``generate_one_map`` but returns additional metadata for
    visualisation: widget start positions and lane polylines, using
    the RoadGen WidgetGraph directly rather than a skeletonised networkx Graph.

    Parameters
    ----------
    widget_number : int
        Target number of widgets.
    temp_dir : Path, optional
        Scratch directory.

    Returns
    -------
    widget_graph : WidgetGraph or None
    positions : dict  node_id -> (x, y)
    lane_data : list[dict]  keys: widget_id, type, pts
    gen_time : float
    n_widgets : int
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

    mfile = str(temp_dir / "viz_temp.m")

    # Storage for visualisation data
    positions = {}
    lane_data = []

    t0 = time.time()

    try:
        with _silent_if_needed():
            with open(mfile, "w", encoding=encoding_format) as f:
                graph = WidgetGraph()
                printAsserts(f)
                component_count = 0
                total_covered_area = []

                # First widget
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

                # Collect viz data
                start = widgetdict0.get("Start", (0, 0))
                positions[w0.WidgetID] = (float(start[0]), float(start[1]))
                try:
                    lane_pts = w0.getlanepoint()
                    for lane in lane_pts:
                        lane_data.append(
                            {"widget_id": w0.WidgetID, "type": widgetdict0.get("Type"), "pts": lane}
                        )
                except Exception:
                    pass

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

                # Subsequent widgets
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

                        # Collect viz data
                        pos = d.get("Start", (0, 0))
                        positions[w1.WidgetID] = (float(pos[0]), float(pos[1]))
                        try:
                            lane_pts = w1.getlanepoint()
                            for lane in lane_pts:
                                lane_data.append(
                                    {"widget_id": w1.WidgetID, "type": d.get("Type"), "pts": lane}
                                )
                        except Exception:
                            pass

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

        gen_time = time.time() - t0

        # Reset widget-level IDs
        Wgt.LaneID = 1
        Wgt.BoundaryID = 1
        Wgt.JunctionID = 1
        Wgt.WidgetID = 1

        if not graph.nodes or not positions:
            return None, {}, [], gen_time, 0

        return graph, positions, lane_data, gen_time, len(graph.nodes)

    except Exception as e:
        gen_time = time.time() - t0
        print(f"  [WARN] Sample generation failed ({type(e).__name__}): {e}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        return None, {}, [], gen_time, 0


def _draw_map(graph, positions, lane_data, title, filepath):
    """Draw road network as a schematic map (widget types + lane geometry).

    Parameters
    ----------
    graph : WidgetGraph
    positions : dict  node_id -> (x, y)
    lane_data : list[dict]
    title : str
    filepath : Path
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_aspect("equal")

    # Draw lane geometry
    for ld in lane_data:
        pts = ld["pts"]
        if len(pts) >= 2:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            color = _WIDGET_COLORS.get(ld["type"], "#999999")
            ax.plot(xs, ys, color=color, linewidth=2.5, alpha=0.7, zorder=2)

    # Draw connections between widget centres as dashed lines
    for src_node, edge_list in graph.edges.items():
        if src_node.id not in positions:
            continue
        xu, yu = positions[src_node.id]
        for dst_node, _ in edge_list:
            if dst_node.id in positions:
                xv, yv = positions[dst_node.id]
                ax.plot(
                    [xu, xv],
                    [yu, yv],
                    color="#666666",
                    linewidth=1,
                    linestyle="--",
                    alpha=0.5,
                    zorder=1,
                )

    # Draw widget nodes
    for nid, node in graph.nodes.items():
        if nid in positions:
            x, y = positions[nid]
            ntype = node.type
            color = _WIDGET_COLORS.get(ntype, "#999999")
            ax.scatter(
                x, y, s=200, c=color, edgecolors="white", linewidths=1.5, zorder=3, alpha=0.9
            )

    # Legend
    used_types = set()
    for nid, node in graph.nodes.items():
        ntype = node.type
        if ntype in _WIDGET_COLORS:
            used_types.add(ntype)
    for ntype in sorted(used_types):
        ax.scatter([], [], c=_WIDGET_COLORS[ntype], s=80, label=_WIDGET_LABELS.get(ntype, ntype))

    ax.legend(loc="upper right", fontsize=8, framealpha=0.8, title="Widget Types", title_fontsize=9)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filepath}")


def plot_samples(output_dir: Path = OUT_DIR):
    """Generate sample RoadGen maps at 3 sizes and visualise them.

    Outputs
    -------
    output_dir/roadgen_map_small.png     (~4 widgets)
    output_dir/roadgen_map_medium.png    (~8 widgets)
    output_dir/roadgen_map_large.png     (~14 widgets)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("small", 4, "RoadGen - Small Map (~4 widgets)"),
        ("medium", 8, "RoadGen - Medium Map (~8 widgets)"),
        ("large", 14, "RoadGen - Large Map (~14 widgets)"),
    ]

    for label, wn, title in configs:
        print(f"Generating {label} map ({wn} widgets)...")
        graph, positions, lane_data, elapsed, actual_nodes = _generate_with_geometry(
            widget_number=wn
        )

        if graph is None:
            print(f"  [WARN] Failed to generate {label} map, skipping.")
            continue

        actual_edges = sum(len(v) for v in graph.edges.values())
        print(f"  Done: {actual_nodes} nodes, {actual_edges} edges, {elapsed:.1f}s")

        filepath = output_dir / f"roadgen_map_{label}.png"
        _draw_map(
            graph,
            positions,
            lane_data,
            f"{title}\n({actual_nodes} widgets, {elapsed:.1f}s generation)",
            filepath,
        )

    print(f"\nAll maps saved to {output_dir}/")


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
            "  python eval/roadgen.py --plot-summary\n"
            "  python eval/roadgen.py --plot-samples\n"
        ),
    )

    # Section selection
    parser.add_argument("--all", action="store_true", help="Run all sections")
    parser.add_argument("--topology", action="store_true", help="Topological validity (Table 1)")
    parser.add_argument("--route", action="store_true", help="Route coverage (Table 3)")
    parser.add_argument("--geometry", action="store_true", help="Geometric validity (Table 2)")
    parser.add_argument(
        "--scalability", action="store_true", help="Large-scale capability (Figure)"
    )
    parser.add_argument(
        "--plot-summary",
        action="store_true",
        help="Plot summary figures and LaTeX table from existing eval results",
    )
    parser.add_argument(
        "--plot-samples",
        action="store_true",
        help="Generate and visualize sample RoadGen maps at 3 sizes",
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
    vis_dir = Path(args.vis).resolve() if args.vis else None

    # Global settings
    global SILENT
    SILENT = args.silent

    init_stats = get_resource_stats()
    peak_stats = dict(init_stats)

    run_all = args.all or not (
        args.topology
        or args.route
        or args.geometry
        or args.scalability
        or args.plot_summary
        or args.plot_samples
    )
    # In --all mode, always save visualizations
    if run_all and vis_dir is None:
        vis_dir = Path(args.output) / "vis"
    _print_timing_estimate(args, run_all)

    if run_all:
        output_dir.mkdir(parents=True, exist_ok=True)
        ref_deg = load_osm_reference_degree()
        target_bins = list(range(10, 81, 5))
        bin_counts = {b: 0 for b in target_bins}

        def _bin(n):
            return ((n - 1) // 5) * 5 + 5

        params = [
            (6, "small"),  # ~10-20 nodes
            (10, "medium"),  # ~15-30 nodes
            (14, "medium"),  # ~20-40 nodes
            (18, "large"),  # ~25-50 nodes
            (22, "large"),  # ~30-55 nodes
            (26, "large"),  # ~35-60 nodes
            (30, "large"),  # ~40-65 nodes
            (34, "large"),  # ~45-70 nodes
            (38, "large"),  # ~50-80 nodes
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

                if all(c >= 5 for c in bin_counts.values()):
                    break
                try:
                    G, gt, polylines = generate_one_map(
                        widget_number=wn, mask_resolution=args.resolution
                    )
                except Exception:
                    continue
                if G is None or G.number_of_nodes() < 2:
                    continue
                nb = _bin(G.number_of_nodes())
                if nb < target_bins[0]:  # skip small maps (bin 5)
                    consecutive_stall += 1
                    continue
                if nb in bin_counts and bin_counts[nb] >= 5:
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
                    geo = compute_all_geometric_metrics(polylines, G=G)

                scale = classify_scale(topo["node_count"])

                row = {
                    "map_id": existing_count + len(per_map),
                    "wn": wn,
                    "attempt": i,
                    "scale": scale,
                    "n_nodes": topo["node_count"],
                    "n_edges": topo["edge_count"],
                    **{
                        k: topo[k]
                        for k in ("lcc", "dead_end_ratio", "avg_degree", "delta_avg_degree")
                    },
                    **{k: route[k] for k in ("reachable_ratio", "avg_route_length")},
                    **{
                        k: geo.get(k, "")
                        for k in (
                            "chamfer_loo",
                            "chamfer_loo_std",
                            "endpoint_alignment",
                            "mean_turning_angle_deg",
                            "intersection_rate",
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
                    filled = sum(1 for c in bin_counts.values() if c >= 5)
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

                if all(c >= 5 for c in bin_counts.values()):
                    break

        gen_time = time.time() - t0
        n_total = existing_count + len(per_map)
        print(f"  Bin counts: {dict(bin_counts)}")
        print(
            f"  Generated {len(per_map)} new maps (→{n_total} total) in "
            f"{gen_time:.1f}s, {gen_time/max(len(per_map),1):.1f}s/map"
        )
        final_stats = get_resource_stats()
        for k in peak_stats:
            peak_stats[k] = max(peak_stats[k], final_stats[k])
        save_system_info(output_dir, "RoadGen", init_stats, peak_stats, n_total)
    else:
        if args.topology:
            run_topology_eval(
                args.num_maps,
                widget_number=args.widget_number,
                output_dir=output_dir,
                mask_resolution=args.resolution,
                vis_dir=vis_dir,
            )
        if args.route:
            run_route_eval(
                args.num_maps,
                widget_number=args.widget_number,
                output_dir=output_dir,
                mask_resolution=args.resolution,
            )
        if args.geometry:
            run_geometry_eval(
                args.num_maps,
                widget_number=args.widget_number,
                output_dir=output_dir,
                mask_resolution=args.resolution,
            )
        if args.scalability:
            run_scalability_eval(
                args.n_per_size, output_dir=output_dir, mask_resolution=args.resolution
            )
        if args.plot_summary:
            plot_summary(output_dir)
        if args.plot_samples:
            plot_samples(output_dir)
        final_stats = get_resource_stats()
        for k in peak_stats:
            peak_stats[k] = max(peak_stats[k], final_stats[k])
        save_system_info(output_dir, "RoadGen", init_stats, peak_stats, args.num_maps)

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

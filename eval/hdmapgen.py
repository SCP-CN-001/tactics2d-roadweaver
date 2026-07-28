#!/usr/bin/env python3
"""
HDMapGen baseline evaluation — same format as eval/metadrive.py.

Sections:
  1. Topological Validity   (Table 1)  — LCC, Dead-end Ratio, Δd̄
  2. Route-Level Coverage   (Table 3)  — Reachable Ratio, Avg Route Length
  3. Geometric Validity     (Table 2)  — Chamfer, Smoothness, Self-intersect, etc.
  4. Large-Scale Capability (Figure)   — Generation time vs node count

Usage:
    conda activate road-weaver
    python eval/hdmapgen.py --all                       # run all sections
    python eval/hdmapgen.py --topology                  # topological only
    python eval/hdmapgen.py --regenerate                # regenerate 610 maps first

Output:
    runtimes/hdmapgen_eval/
        topology.json          — topological validity results
        route_coverage.json    — route coverage results
        geometry.json          — geometric validity results (all metrics)
        scalability.csv        — CSV for plotting
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from scipy.interpolate import CubicSpline
from tqdm import tqdm

from eval.metrics import (
    append_csv_row,
    classify_scale,
    compute_all_geometric_metrics,
    compute_route_coverage,
    compute_subnode_uniformity,
    compute_topological_metrics,
    contract_degree2_nodes,
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

REPO = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO / "runtimes" / "hdmapgen_eval"
GRAPH_PKL = OUTPUT_DIR / "generated_graphs.pkl"

# HDMapGen's GRAN model preprocesses nuPlan coordinates by dividing by 32.
# Multiply generated coordinates by 32 to recover real-world meters.
_HDMAPGEN_METER_SCALE = 32

# ═══════════════════════════════════════════════════════════════════════════
#  Graph loading / regeneration
# ═══════════════════════════════════════════════════════════════════════════


def load_generated_graphs() -> list[dict]:
    """Load pre-generated graphs from pickle."""
    with open(GRAPH_PKL, "rb") as f:
        return pickle.load(f)


def dict_to_nx(entry: dict, graph_method: str = "legacy", scale_to_meters: float = 1.0) -> nx.Graph:
    """Convert saved dict back to NetworkX graph.

    Two extraction methods:

    * ``"legacy"`` (default)
        Build a raw graph from adjacency, assign node coordinates from
        ``node_coords``, then contract degree-2 waypoints so that the
        resulting graph is at intersection-level granularity.

    * ``"skeleton"``
        Extract road polylines from ``edge_subnodes``, render as a binary
        mask at 1024×1024, skeletonise, and extract the graph.  No pruning
        is applied (keeps all branches for fair evaluation).

    Args:
        entry: Saved graph dict.
        graph_method: ``"legacy"`` or ``"skeleton"``.
        scale_to_meters: Multiply coordinates by this factor to convert to
            physical meters.  HDMapGen's GRAN model outputs normalised
            coordinates (preprocessed as ``nuplan_coord / 32``), so pass
            ``scale_to_meters=32`` to recover real-world metre values.
    """
    # ── Skeleton method ──────────────────────────────────────────────
    if graph_method == "skeleton":
        polylines = extract_polylines_from_subnodes(entry, scale_to_meters=scale_to_meters)
        # Adaptive merge distance = 3% of extent
        if polylines:
            all_pts = np.vstack(polylines)
            extent = max(all_pts.max(axis=0) - all_pts.min(axis=0))
            merge_d = max(0.02, extent * 0.03)
        else:
            merge_d = 0.05
        return polylines_to_graph(polylines, resolution=1024, cleanup=False, merge_distance=merge_d)

    # ── Legacy method ────────────────────────────────────────────────
    G = nx.Graph()
    n = entry["num_nodes"]
    G.add_nodes_from(range(n))

    coords = entry.get("node_coords")
    if coords:
        for i in range(n):
            if coords[i] is not None:
                G.nodes[i]["coords"] = np.array(coords[i]) * scale_to_meters

    for u, v in entry["edges"]:
        G.add_edge(u, v)

    G = contract_degree2_nodes(G)
    return G


def extract_polylines_from_subnodes(
    entry: dict, interpolate: bool = True, n_interp: int = 100, scale_to_meters: float = 1.0
) -> list[np.ndarray]:
    """Extract road geometry polylines from edge subnodes (20 pts/edge normally).

    When *interpolate* is ``True`` (default), sparse subnode sequences are
    up-sampled to *n_interp* evenly-spaced points via cubic interpolation so
    that the rendered binary mask shows smooth road curves rather than jagged
    stick figures.

    Args:
        entry: Saved graph dict with ``edge_subnodes``.
        interpolate: Whether to cubic-interpolate sparse subnode sequences.
        n_interp: Target number of points per polyline when interpolating.
        scale_to_meters: Multiply coordinates by this factor to convert to
            physical meters.  Pass ``32`` for HDMapGen's GRAN model.
    """
    polylines = []
    for (_u, _v), raw_sn in entry.get("edge_subnodes", {}).items():
        if raw_sn is None or len(raw_sn) < 2:
            continue
        pts = np.array(raw_sn, dtype=np.float64)
        if pts.shape[1] != 2:
            continue

        if interpolate and len(pts) < n_interp:
            # Cubic interpolation over the arc-length parameter
            t_orig = np.linspace(0, 1, len(pts))
            t_new = np.linspace(0, 1, n_interp)
            cs_x = CubicSpline(t_orig, pts[:, 0], bc_type="natural")
            cs_y = CubicSpline(t_orig, pts[:, 1], bc_type="natural")
            pts = np.column_stack([cs_x(t_new), cs_y(t_new)])

        if scale_to_meters != 1.0:
            pts = pts * scale_to_meters

        polylines.append(pts)
    return polylines


def _checkpoint_model_dir() -> Path:
    """Find the nuplan checkpoint directory."""
    base = REPO / "baselines" / "HDMapGen" / "exp" / "GRAN"
    candidates = list(base.glob("GRANMixtureBernoulli_nuplan_*"))
    if candidates:
        return sorted(candidates)[-1]
    return base / "GRANMixtureBernoulli_nuplan_2024-Mar-02-16-11-53_2285727"


def regenerate_graphs() -> list[dict]:
    """Regenerate 610 maps (9-69 nodes, 10 each) using the HDMapGen model.
    NOTE: Model is architected for max 70 nodes, so 70-80 bins rely on
    the existing 610-map dataset."""
    print("  Regenerating graphs with HDMapGen model...")
    HDMAPGEN = REPO / "baselines" / "HDMapGen"
    sys.path.insert(0, str(HDMAPGEN))

    import yaml
    from easydict import EasyDict as edict
    from model import GRANMixtureBernoulli

    from utils.train_helper import load_model

    ckpt_dir = _checkpoint_model_dir()
    CKPT_FILE = ckpt_dir / "model_snapshot_0000425.pth"
    CONFIG_FILE = HDMAPGEN / "config" / "gran_nuplan.yaml"

    with open(CONFIG_FILE) as f:
        cfg = edict(yaml.safe_load(f))
    cfg.device = "cuda:0"
    cfg.use_gpu = True
    cfg.gpus = [0]
    cfg.model.max_num_nodes = 70
    cfg.dataset.has_stop_node = False
    cfg.dataset.is_noisy = False
    cfg.dataset.has_node_feat = True
    cfg.dataset.has_sub_nodes = True
    cfg.dataset.num_sub_nodes = 20
    cfg.test.is_vis = False
    cfg.test.animated_vis = False

    model = GRANMixtureBernoulli(cfg)
    load_model(model, str(CKPT_FILE), cfg.device)
    model = model.to(cfg.device)
    model.eval()

    all_graphs = []
    batch_size = 10

    for n in range(9, 70):
        pmf = np.zeros(cfg.model.max_num_nodes)
        pmf[n - 1] = 1.0

        with torch.no_grad():
            t0 = time.time()
            input_dict = {"is_sampling": True, "batch_size": batch_size, "num_nodes_pmf": pmf}
            A_list, node_embed_list, subnode_list, _ = model(input_dict)
            elapsed = time.time() - t0

        for bidx in range(len(A_list)):
            A_np = A_list[bidx].cpu().numpy() if hasattr(A_list[bidx], "cpu") else A_list[bidx]
            ne_np = node_embed_list[bidx]
            if hasattr(ne_np, "cpu"):
                ne_np = ne_np.cpu().numpy()
            if ne_np.ndim == 3:
                ne_np = ne_np[0]

            sn_np = subnode_list[bidx] if subnode_list is not None else None
            if hasattr(sn_np, "cpu"):
                sn_np = sn_np.cpu().numpy()

            n_actual = A_np.shape[0]
            edges = []
            edge_subnodes = {}
            for u in range(n_actual):
                for v in range(u + 1, n_actual):
                    if A_np[u, v] > 0.5:
                        edges.append((u, v))
                        if sn_np is not None and u < sn_np.shape[0] and v < sn_np.shape[1]:
                            # Subnode data stored at (larger, smaller) index
                            raw = sn_np[v, u] if sn_np.ndim == 3 else sn_np
                            if isinstance(raw, np.ndarray) and raw.size >= 40:
                                pts = raw.reshape(-1, 2).tolist()
                                edge_subnodes[(u, v)] = pts

            entry = {
                "num_nodes": n_actual,
                "gen_time_s": elapsed / batch_size,
                "edges": edges,
                "node_coords": [
                    ne_np[i].tolist() if i < len(ne_np) else None for i in range(n_actual)
                ],
                "edge_subnodes": edge_subnodes,
            }
            all_graphs.append(entry)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(GRAPH_PKL, "wb") as f:
        pickle.dump(all_graphs, f)
    print(f"    Generated {len(all_graphs)} graphs, saved to {GRAPH_PKL}")
    return all_graphs


# ═══════════════════════════════════════════════════════════════════════════
#  Section 1: Topological Validity  (Table 1)
# ═══════════════════════════════════════════════════════════════════════════


def run_topology_eval(
    graph_method: str = "legacy",
    output_dir: Path = OUTPUT_DIR,
    vis_dir: Path | None = None,
    max_maps: int = 0,
) -> dict:
    """Load graphs and compute topological validity."""
    print(f"\n{'=' * 60}")
    print("Section 1: Topological Validity  (HDMapGen)")
    print(f"  Processing {len(load_generated_graphs())} graphs, " f"graph_method={graph_method}")
    print(f"{'=' * 60}")

    ref_deg = load_osm_reference_degree()
    print(f"  OSM reference avg_degree = {ref_deg:.4f}")

    data = load_generated_graphs()
    if max_maps > 0:
        data = data[:max_maps]
    print(f"  Loaded {len(data)} graphs")

    topo_metrics = defaultdict(list)
    all_results = []
    vis_counts: dict[str, int] = {}

    for idx, entry in enumerate(data):
        G = dict_to_nx(entry, graph_method=graph_method, scale_to_meters=_HDMAPGEN_METER_SCALE)
        if G.number_of_nodes() < 2:
            continue
        topo = compute_topological_metrics(G, ref_avg_degree=ref_deg)
        for k, v in topo.items():
            topo_metrics[k].append(v)

        scale = classify_scale(topo["node_count"])

        # ── Save vis (up to 5 per scale) ──
        if vis_dir is not None:
            cnt = vis_counts.get(scale, 0)
            if cnt < 5:
                polylines = extract_polylines_from_subnodes(
                    entry, scale_to_meters=_HDMAPGEN_METER_SCALE
                )
                nc, ec = G.number_of_nodes(), G.number_of_edges()
                vis_path = Path(vis_dir) / f"hd_{scale}_N{nc}E{ec}_{cnt}.png"
                vis_path.parent.mkdir(parents=True, exist_ok=True)
                save_vis(polylines, G, str(vis_path), resolution=1024, title=f"HDMapGen {scale}")
                vis_counts[scale] = cnt + 1

        all_results.append(
            {
                "num_nodes": entry["num_nodes"],
                "gen_time_s": entry["gen_time_s"],
                "scale": scale,
                **topo,
            }
        )

    agg = {}
    for k, v in topo_metrics.items():
        agg[f"{k}_mean"] = float(np.mean(v))
        agg[f"{k}_std"] = float(np.std(v))
    agg["n_maps"] = len(all_results)
    agg["osm_reference_avg_degree"] = ref_deg

    save_results(output_dir, "topology", agg, all_results)
    print_results_table(
        "Topological Validity",
        agg,
        [
            ("lcc", "LCC", ".4f"),
            ("dead_end_ratio", "Dead-end", ".4f"),
            ("delta_avg_degree", "Δd̄", ".4f"),
        ],
    )

    return agg


# ═══════════════════════════════════════════════════════════════════════════
#  Section 2: Route-Level Coverage  (Table 3)
# ═══════════════════════════════════════════════════════════════════════════


def run_route_eval(
    n_pairs: int = 100,
    graph_method: str = "legacy",
    output_dir: Path = OUTPUT_DIR,
    max_maps: int = 0,
) -> dict:
    """Compute route coverage."""
    print(f"\n{'=' * 60}")
    print("Section 2: Route-Level Coverage  (HDMapGen)")
    print(f"  {n_pairs} OD pairs per graph (graph_method={graph_method})")
    print(f"{'=' * 60}")

    data = load_generated_graphs()
    if max_maps > 0:
        data = data[:max_maps]
    print(f"  Loaded {len(data)} graphs")

    route_metrics = defaultdict(list)
    all_results = []

    for idx, entry in enumerate(data):
        G = dict_to_nx(entry, graph_method=graph_method, scale_to_meters=_HDMAPGEN_METER_SCALE)
        if G.number_of_nodes() < 2:
            continue
        route = compute_route_coverage(G, n_pairs=n_pairs, seed=idx)
        for k, v in route.items():
            route_metrics[k].append(v)
        all_results.append({"num_nodes": entry["num_nodes"], **route})

    agg = {}
    for k, v in route_metrics.items():
        agg[f"{k}_mean"] = float(np.mean(v))
        agg[f"{k}_std"] = float(np.std(v))
    agg["n_maps"] = len(all_results)

    save_results(output_dir, "route_coverage", agg, all_results)
    print_results_table(
        "Route Coverage",
        agg,
        [("reachable_ratio", "Reachable", ".4f"), ("avg_route_length", "Avg Length", ".1f")],
    )

    return agg


# ═══════════════════════════════════════════════════════════════════════════
#  Section 3: Geometric Validity  (Table 2)
# ═══════════════════════════════════════════════════════════════════════════


def run_geometry_eval(
    graph_method: str = "legacy", output_dir: Path = OUTPUT_DIR, max_maps: int = 0
) -> dict:
    """Compute all geometric metrics from generated edge subnodes."""
    print(f"\n{'=' * 60}")
    print("Section 3: Geometric Validity  (HDMapGen)")
    print(
        f"  (Internal: edge smoothness, self-intersection, length CV, "
        f"uniformity, angles)  graph_method={graph_method}"
    )
    print(f"{'=' * 60}")

    data = load_generated_graphs()
    if max_maps > 0:
        data = data[:max_maps]
    print(f"  Loaded {len(data)} graphs")

    geom_metrics = defaultdict(list)
    all_results = []

    for entry in data:
        G = dict_to_nx(entry, graph_method=graph_method, scale_to_meters=_HDMAPGEN_METER_SCALE)
        polylines = extract_polylines_from_subnodes(entry, scale_to_meters=_HDMAPGEN_METER_SCALE)
        # Raw subnodes (pre-interpolation) for uniformity — cubic spline masks spacing
        raw_polylines = extract_polylines_from_subnodes(
            entry, interpolate=False, scale_to_meters=_HDMAPGEN_METER_SCALE
        )

        if polylines:
            geo = compute_all_geometric_metrics(polylines, G=G)
            geo.update(compute_subnode_uniformity(raw_polylines))  # raw, not interpolated
        else:
            geo = {"chamfer_loo": 0.0, "endpoint_alignment": 0.0}

        entry_out = {"num_nodes": entry["num_nodes"], "num_edges": G.number_of_edges(), **geo}
        for k, v in geo.items():
            if isinstance(v, (int, float, np.floating, np.integer)):
                geom_metrics[k].append(v)
        all_results.append(entry_out)

    agg = {}
    for k, v in geom_metrics.items():
        agg[f"{k}_mean"] = float(np.mean(v))
        agg[f"{k}_std"] = float(np.std(v))
    agg["n_maps"] = len(all_results)

    save_results(output_dir, "geometry", agg, all_results)
    print_results_table(
        "Geometric Validity",
        agg,
        [
            ("chamfer_loo", "Chamfer (LOO)", ".4f"),
            ("endpoint_alignment", "Endpoint Align", ".4f"),
            ("mean_turning_angle_deg", "Turning Angle", ".2f"),
            ("mean_edge_length", "Edge Length", ".4f"),
            ("cv_edge_length", "Length CV", ".3f"),
            ("intersection_rate", "Intersect Rate", ".6f"),
            ("mean_spacing_cv", "Subnode Uniformity", ".3f"),
            ("mean_angle_deg", "Junction Angle", ".2f"),
        ],
    )

    return agg


# ═══════════════════════════════════════════════════════════════════════════
#  Section 4: Large-Scale Capability  (Figure)
# ═══════════════════════════════════════════════════════════════════════════


def run_scalability_eval(
    graph_method: str = "legacy", output_dir: Path = OUTPUT_DIR, max_maps: int = 0
) -> dict:
    """Aggregate generation time vs node count."""
    print(f"\n{'=' * 60}")
    print("Section 4: Large-Scale Capability  (HDMapGen)")
    print(f"  graph_method={graph_method}")
    print(f"{'=' * 60}")

    data = load_generated_graphs()
    if max_maps > 0:
        data = data[:max_maps]
    print(f"  Loaded {len(data)} graphs")

    by_size = defaultdict(list)
    for entry in data:
        n = entry["num_nodes"]
        G = dict_to_nx(entry, graph_method=graph_method, scale_to_meters=_HDMAPGEN_METER_SCALE)
        topo = compute_topological_metrics(G)
        by_size[n].append(
            {
                "node_count": n,
                "edge_count": G.number_of_edges(),
                "generation_time": entry["gen_time_s"],
                **{k: topo.get(k, 0) for k in ("lcc", "dead_end_ratio", "avg_degree")},
            }
        )

    all_sizes = []
    for n in tqdm(sorted(by_size.keys()), desc="Scaling"):
        maps = by_size[n]
        nodes = [m["node_count"] for m in maps]
        times = [m["generation_time"] for m in maps]
        lccs = [m["lcc"] for m in maps]
        edges = [m["edge_count"] for m in maps]

        all_sizes.append(
            {
                "target_nodes": n,
                "aggregate": {
                    "node_count_mean": float(np.mean(nodes)),
                    "node_count_std": float(np.std(nodes)),
                    "edge_count_mean": float(np.mean(edges)),
                    "gen_time_mean": float(np.mean(times)),
                    "gen_time_std": float(np.std(times)),
                    "lcc_mean": float(np.mean(lccs)),
                    "lcc_std": float(np.std(lccs)),
                    "dead_end_ratio_mean": float(np.mean([m["dead_end_ratio"] for m in maps])),
                    "dead_end_ratio_std": float(np.std([m["dead_end_ratio"] for m in maps])),
                },
            }
        )
        print(
            f"    n={n:2d}: time={np.mean(times):.4f}s  "
            f"edges={np.mean(edges):.0f}  LCC={np.mean(lccs):.3f}"
        )

    csv_path = output_dir / "scalability.csv"
    with open(csv_path, "w") as f:
        f.write(
            "target_nodes,node_count_mean,node_count_std,edge_count_mean,"
            "gen_time_mean,gen_time_std,lcc_mean,lcc_std\n"
        )
        for r in all_sizes:
            a = r["aggregate"]
            f.write(
                f"{r['target_nodes']},{a['node_count_mean']},{a['node_count_std']},"
                f"{a['edge_count_mean']},{a['gen_time_mean']},{a['gen_time_std']},"
                f"{a['lcc_mean']},{a['lcc_std']}\n"
            )
    print(f"\n  CSV saved to {csv_path}")
    return {"n_sizes": len(all_sizes)}


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="HDMapGen baseline evaluation")
    parser.add_argument("--all", action="store_true", help="Run all sections")
    parser.add_argument("--topology", action="store_true", help="Topological validity")
    parser.add_argument("--route", action="store_true", help="Route coverage")
    parser.add_argument("--geometry", action="store_true", help="Geometric validity")
    parser.add_argument("--scalability", action="store_true", help="Large-scale capability")
    parser.add_argument("--regenerate", action="store_true", help="Regenerate graphs first")
    parser.add_argument(
        "--graph-method",
        type=str,
        default="legacy",
        choices=["legacy", "skeleton"],
        help="Graph extraction: 'legacy' (adjacency + degree-2 contract) "
        "or 'skeleton' (render edge subnodes → skeletonise → graph)",
    )
    parser.add_argument(
        "--vis", type=str, default=None, help="Save mask+graph visualisation images to DIR"
    )
    parser.add_argument(
        "--max-maps",
        type=int,
        default=100,
        help="Limit number of graphs processed (default 100, 0 = all 610)",
    )
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output)
    vis_dir = Path(args.vis) if args.vis else None
    max_maps = args.max_maps if args.max_maps > 0 else 0  # 0 = all
    gm = args.graph_method
    run_all = args.all or not (args.topology or args.route or args.geometry or args.scalability)

    # In --all mode, always save visualizations
    if run_all and vis_dir is None:
        vis_dir = output_dir / "vis"

    init_stats = get_resource_stats()
    peak_stats = dict(init_stats)

    if args.regenerate or not GRAPH_PKL.exists():
        regenerate_graphs()
    else:
        print(f"Using existing graphs from {GRAPH_PKL}")

    if run_all:
        output_dir.mkdir(parents=True, exist_ok=True)
        ref_deg = load_osm_reference_degree()
        all_data = pickle.load(open(GRAPH_PKL, "rb"))
        if max_maps > 0:
            all_data = all_data[:max_maps]

        target_bins = list(range(10, 81, 5))
        bin_counts = {b: 0 for b in target_bins}

        def _bin(n):
            return ((n - 1) // 5) * 5 + 5

        # ── Resume: load existing CSV ─────────────────────────────────────
        csv_path = output_dir / "all_metrics.csv"
        seen_keys = load_csv_keys(csv_path, ["pickle_idx"])
        if seen_keys:
            print(f"  Resuming: {len(seen_keys)} existing entries loaded from {csv_path.name}")

        existing_count = 0
        if csv_path.exists():
            with open(csv_path) as _f:
                existing_count = max(0, sum(1 for _ in _f) - 1)

        per_map = []
        t0 = time.time()
        for idx, entry in enumerate(all_data):
            key = str(idx)
            if key in seen_keys:
                continue

            if all(c >= 5 for c in bin_counts.values()):
                break
            try:
                G = dict_to_nx(entry, graph_method=gm, scale_to_meters=_HDMAPGEN_METER_SCALE)
            except Exception:
                continue
            if G.number_of_nodes() < 2:
                continue
            nb = _bin(G.number_of_nodes())
            if nb < target_bins[0]:  # skip small maps (bin 5)
                continue
            if nb in bin_counts and bin_counts[nb] >= 5:
                continue

            # Single resource monitoring block for all metric computation
            with monitor_resources(interval=0.3) as peaks:
                topo = compute_topological_metrics(G, ref_avg_degree=ref_deg)
                route = compute_route_coverage(G, n_pairs=100, seed=idx)
                polylines = extract_polylines_from_subnodes(
                    entry, scale_to_meters=_HDMAPGEN_METER_SCALE
                )
                raw_polylines = extract_polylines_from_subnodes(
                    entry, interpolate=False, scale_to_meters=_HDMAPGEN_METER_SCALE
                )
                geo = compute_all_geometric_metrics(polylines, G=G)
                geo.update(compute_subnode_uniformity(raw_polylines))

            scale = classify_scale(topo["node_count"])

            row = {
                "map_id": existing_count + len(per_map),
                "pickle_idx": idx,
                "scale": scale,
                "n_nodes": topo["node_count"],
                "n_edges": topo["edge_count"],
                **{k: topo[k] for k in ("lcc", "dead_end_ratio", "avg_degree", "delta_avg_degree")},
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
                "gen_time_s": round(entry["gen_time_s"], 4),
                "cpu_peak": round(peaks["cpu_peak"], 1),
                "mem_peak_mb": round(peaks["mem_peak_mb"], 1),
                "gpu_peak_mb": round(peaks["gpu_peak_mb"], 1),
            }
            per_map.append(row)
            append_csv_row(csv_path, list(row.keys()), row)
            seen_keys.add(key)

            if nb in bin_counts:
                bin_counts[nb] += 1

            if len(per_map) % 20 == 0:
                filled = sum(1 for c in bin_counts.values() if c >= 5)
                print(
                    f"  [{existing_count + len(per_map)} maps] bins filled: {filled}/{len(target_bins)}"
                )

            # ── Vis ──
            if vis_dir is not None:
                cnt = len(list(Path(vis_dir).glob(f"hd_{scale}_*.png")))
                if cnt < 5:
                    save_vis(
                        polylines,
                        G,
                        str(
                            Path(vis_dir)
                            / f"hd_{scale}_N{G.number_of_nodes()}E{G.number_of_edges()}_{cnt}.png"
                        ),
                    )

        gen_time = time.time() - t0
        n_total = existing_count + len(per_map)
        print(f"  Bin counts: {dict(bin_counts)}")
        print(f"  Processed {len(per_map)} new maps (→{n_total} total) in " f"{gen_time:.1f}s")
        final_stats = get_resource_stats()
        for k in peak_stats:
            peak_stats[k] = max(peak_stats[k], final_stats[k])
        save_system_info(output_dir, "HDMapGen", init_stats, peak_stats, n_total)
    else:
        if args.topology:
            run_topology_eval(gm, output_dir, vis_dir=vis_dir, max_maps=max_maps)
        if args.route:
            run_route_eval(graph_method=gm, output_dir=output_dir, max_maps=max_maps)
        if args.geometry:
            run_geometry_eval(gm, output_dir, max_maps=max_maps)
        if args.scalability:
            run_scalability_eval(gm, output_dir, max_maps=max_maps)
        final_stats = get_resource_stats()
        for k in peak_stats:
            peak_stats[k] = max(peak_stats[k], final_stats[k])
        save_system_info(output_dir, "HDMapGen", init_stats, peak_stats, len(data))

    print(f"\n{'=' * 60}")
    print(f"HDMapGen evaluation complete — results in {output_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
MetaDrive baseline evaluation — organized by paper experiment sections.

Sections:
  1. Topological Validity   (Table 1)  — LCC, Dead-end Ratio, Δd̄
  2. Route-Level Coverage   (Table 3)  — Reachable Ratio, Avg Route Length
  3. Geometric Validity     (Table 2)  — Chamfer Distance, Endpoint Alignment, etc.
  4. Large-Scale Capability (Figure)   — Generation time vs node count

Usage:
    conda activate road-weaver
    python eval/metadrive.py --all                    # run all sections
    python eval/metadrive.py --topology               # topological validity only
    python eval/metadrive.py --route                  # route coverage only
    python eval/metadrive.py --geometry               # geometric only
    python eval/metadrive.py --topology --num_maps 200

Output:
    runtimes/metadrive_eval/
        topology.json          — topological validity results
        route_coverage.json    — route coverage results
        geometry.json          — geometric validity results
        scalability.csv        — scalability data
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
from tqdm import tqdm

from eval.metrics import (
    classify_scale,
    compute_all_geometric_metrics,
    compute_route_coverage,
    compute_topological_metrics,
    extract_intersection_graph,
    get_resource_stats,
    is_metadrive_road_node,
    load_osm_reference_degree,
    print_results_table,
    save_results,
    save_system_info,
)
from eval.polyline_graph import polylines_to_graph, save_vis

# Ensure the metadrive submodule is importable (shadows pip package)
_metadrive_path = Path(__file__).resolve().parent.parent / "baselines" / "MetaDrive"
if _metadrive_path.exists() and str(_metadrive_path) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_metadrive_path))

from metadrive.envs import MetaDriveEnv

# ═══════════════════════════════════════════════════════════════════════════
#  Paths
# ═══════════════════════════════════════════════════════════════════════════

REPO = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO / "runtimes" / "metadrive_eval"

# ═══════════════════════════════════════════════════════════════════════════
#  Map generation
# ═══════════════════════════════════════════════════════════════════════════


def _extract_graph_legacy(env) -> nx.Graph:
    """Extract intersection-based road graph from MetaDrive environment.

    Uses spatial clustering (30m radius) to merge lane-level nodes into
    real-world intersections, then collapses degree-2 waypoints.
    """
    return extract_intersection_graph(env)


def generate_metadrive_map(seed: int, map_config: int = 7, graph_method: str = "legacy") -> dict:
    """Generate a MetaDrive map and extract graph + lane geometry.

    Args:
        seed: Random seed.
        map_config: MetaDrive map configuration index.
        graph_method: ``"legacy"`` (spatial clustering + degree-2 contraction)
                      or ``"skeleton"`` (render lane polylines → skeleton graph).

    Returns:
        dict with keys: graph, lanes, generation_time, node_count, edge_count, seed
    """
    config = dict(
        start_seed=seed, num_scenarios=1, traffic_density=0.0, use_render=False, map=map_config
    )
    env = MetaDriveEnv(config)
    t0 = time.time()
    _obs, _ = env.reset()
    gen_time = time.time() - t0

    # Extract lane geometry polylines (used by both methods)
    # Filter out decoration/edge nodes (">", "->", etc.) using naming pattern
    rn = env.engine.current_map.road_network
    polylines = []
    for sn, td in rn.graph.items():
        for en, lanes in td.items():
            if not is_metadrive_road_node(sn) or not is_metadrive_road_node(en):
                continue
            for lane in lanes:
                try:
                    pts = np.array(lane.get_polyline(interval=2))
                    if len(pts) >= 2:
                        polylines.append(pts)
                except Exception:
                    pass

    # Graph extraction
    if graph_method == "skeleton":
        # Merge distance in meters: enough to combine skeleton junction
        # pixels at the same intersection (typically 3-8 px at 1024 res)
        # without merging separate intersections (typically >30m apart).
        merge_m = 8.0
        G = polylines_to_graph(polylines, resolution=1024, cleanup=False, merge_distance=merge_m)
    else:
        G = _extract_graph_legacy(env)

    env.close()

    return {
        "graph": G,
        "polylines": polylines,
        "generation_time": gen_time,
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "seed": seed,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Section 1: Topological Validity  (Table 1)
# ═══════════════════════════════════════════════════════════════════════════


def run_topology_eval(
    num_maps: int = 200,
    seed: int = 0,
    map_config: int = 7,
    graph_method: str = "legacy",
    output_dir: Path = OUTPUT_DIR,
    vis_dir: Path | None = None,
) -> dict:
    """Generate maps and compute topological validity metrics."""
    print(f"\n{'=' * 60}")
    print("Section 1: Topological Validity  (MetaDrive)")
    print(
        f"  Generating {num_maps} maps (map={map_config}, seed={seed}, " f"graph={graph_method})..."
    )
    print(f"{'=' * 60}")

    ref_deg = load_osm_reference_degree()
    print(f"  OSM reference avg_degree = {ref_deg:.4f}")

    topo_metrics = defaultdict(list)
    all_results = []

    # ── Per-scale vis counters (up to 5 per scale) ──
    vis_counts: dict[str, int] = {}

    for i in range(num_maps):
        result = generate_metadrive_map(seed + i, map_config=map_config, graph_method=graph_method)

        G = result["graph"]
        if G.number_of_nodes() == 0:
            continue

        topo = compute_topological_metrics(G, ref_avg_degree=ref_deg)

        # ── Save vis (up to 5 per scale) ──
        scale = classify_scale(topo["node_count"])
        if vis_dir is not None:
            cnt = vis_counts.get(scale, 0)
            if cnt < 5:
                nc, ec = G.number_of_nodes(), G.number_of_edges()
                vis_path = Path(vis_dir) / f"md_{scale}_N{nc}E{ec}_{cnt}.png"
                vis_path.parent.mkdir(parents=True, exist_ok=True)
                save_vis(
                    result["polylines"],
                    G,
                    str(vis_path),
                    resolution=1024,
                    title=f"MetaDrive {graph_method} {scale}",
                )
                vis_counts[scale] = cnt + 1

        for k, v in topo.items():
            topo_metrics[k].append(v)

        all_results.append(
            {
                "seed": result["seed"],
                "node_count": result["node_count"],
                "edge_count": result["edge_count"],
                "generation_time": round(result["generation_time"], 3),
                "scale": classify_scale(topo["node_count"]),
                **topo,
            }
        )

        if (i + 1) % 50 == 0 or i == 0:
            print(
                f"    [{i + 1}/{num_maps}] "
                f"LCC={np.mean(topo_metrics['lcc']):.3f} "
                f"dead={np.mean(topo_metrics['dead_end_ratio']):.3f}"
            )

    agg = {}
    for k, v in topo_metrics.items():
        agg[f"{k}_mean"] = float(np.mean(v))
        agg[f"{k}_std"] = float(np.std(v))
    agg["n_maps"] = len(all_results)
    agg["avg_generation_time"] = float(np.mean([r["generation_time"] for r in all_results]))
    agg["osm_reference_avg_degree"] = ref_deg

    save_results(output_dir, "topology", agg, all_results)
    print_results_table(
        "Topological Validity",
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
    map_config: int = 7,
    n_pairs: int = 100,
    graph_method: str = "legacy",
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    """Generate maps and compute route coverage metrics."""
    print(f"\n{'=' * 60}")
    print("Section 2: Route-Level Coverage  (MetaDrive)")
    print(f"  Generating {num_maps} maps, {n_pairs} OD pairs each " f"(graph={graph_method})...")
    print(f"{'=' * 60}")

    route_metrics = defaultdict(list)
    all_results = []

    for i in range(num_maps):
        result = generate_metadrive_map(seed + i, map_config=map_config, graph_method=graph_method)
        G = result["graph"]
        if G.number_of_nodes() < 2:
            continue

        route = compute_route_coverage(G, n_pairs=n_pairs, seed=seed + i)
        for k, v in route.items():
            route_metrics[k].append(v)

        all_results.append(
            {
                "seed": result["seed"],
                "node_count": result["node_count"],
                "edge_count": result["edge_count"],
                **route,
            }
        )

        if (i + 1) % 50 == 0:
            print(
                f"    [{i + 1}/{num_maps}] "
                f"reachable={np.mean(route_metrics['reachable_ratio']):.3f}"
            )

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
    num_maps: int = 200,
    seed: int = 0,
    map_config: int = 7,
    graph_method: str = "legacy",
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    """Generate maps and compute all geometric metrics from lane polylines."""
    print(f"\n{'=' * 60}")
    print("Section 3: Geometric Validity  (MetaDrive)")
    print(f"  Generating {num_maps} maps (graph={graph_method})...")
    print(f"{'=' * 60}")

    geom_metrics = defaultdict(list)
    all_results = []

    for i in range(num_maps):
        result = generate_metadrive_map(seed + i, map_config=map_config, graph_method=graph_method)
        G = result["graph"]
        if G.number_of_nodes() < 2:
            continue

        # MetaDrive nodes now have coordinates from lane polyline midpoints
        # (tagged during _extract_graph → _tag_node_positions)
        geo = compute_all_geometric_metrics(result["polylines"], G=G)

        entry = {
            "seed": result["seed"],
            "node_count": result["node_count"],
            "edge_count": result["edge_count"],
            **geo,
        }
        for k, v in geo.items():
            if isinstance(v, (int, float, np.floating, np.integer)):
                geom_metrics[k].append(v)
        all_results.append(entry)

        if (i + 1) % 50 == 0:
            print(
                f"    [{i + 1}/{num_maps}] "
                f"chamfer_loo={np.mean(geom_metrics['chamfer_loo']):.4f}"
            )

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
#  Section 4: Large-Scale Capability  (Scalability Figure)
# ═══════════════════════════════════════════════════════════════════════════


def run_scalability_eval(
    n_per_size: int = 10,
    route_pairs: int = 100,
    graph_method: str = "legacy",
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    """Generate maps at controlled sizes and record metrics."""
    print(f"\n{'=' * 60}")
    print("Section 4: Large-Scale Capability  (MetaDrive)")
    print(f"  Maps: 10 → 200 nodes, {n_per_size} per size")
    print(f"{'=' * 60}")

    target_nodes = list(range(10, 201, 10))
    map_values = [max(1, int(t / 8)) for t in target_nodes]

    all_results = []

    for target, map_val in tqdm(list(zip(target_nodes, map_values)), desc="Scaling"):
        per_size = {"target_nodes": target, "map_config": map_val, "maps": []}

        for i in range(n_per_size):
            s = target * 1000 + i
            result = generate_metadrive_map(s, map_config=map_val, graph_method=graph_method)
            G = result["graph"]

            topo = compute_topological_metrics(G)
            route = compute_route_coverage(G, n_pairs=route_pairs, seed=s)

            per_size["maps"].append(
                {
                    "node_count": result["node_count"],
                    "edge_count": result["edge_count"],
                    "generation_time": result["generation_time"],
                    **topo,
                    **route,
                }
            )

        nodes = [m["node_count"] for m in per_size["maps"]]
        times = [m["generation_time"] for m in per_size["maps"]]
        lccs = [m["lcc"] for m in per_size["maps"]]

        per_size["aggregate"] = {
            "node_count_mean": float(np.mean(nodes)),
            "node_count_std": float(np.std(nodes)),
            "gen_time_mean": float(np.mean(times)),
            "gen_time_std": float(np.std(times)),
            "lcc_mean": float(np.mean(lccs)),
            "lcc_std": float(np.std(lccs)),
        }
        all_results.append(per_size)

        print(
            f"    target={target:3d} → actual={per_size['aggregate']['node_count_mean']:5.0f}±"
            f"{per_size['aggregate']['node_count_std']:4.1f}  "
            f"time={per_size['aggregate']['gen_time_mean']:.3f}s"
        )

    # CSV
    csv_path = output_dir / "scalability.csv"
    with open(csv_path, "w") as f:
        f.write(
            "target_nodes,map_config,actual_nodes_mean,actual_nodes_std,"
            "gen_time_mean,gen_time_std,lcc_mean,lcc_std\n"
        )
        for r in all_results:
            a = r["aggregate"]
            f.write(
                f"{r['target_nodes']},{r['map_config']},"
                f"{a['node_count_mean']},{a['node_count_std']},"
                f"{a['gen_time_mean']},{a['gen_time_std']},"
                f"{a['lcc_mean']},{a['lcc_std']}\n"
            )
    print(f"\n  CSV saved to {csv_path}")

    return {"n_sizes": len(all_results), "n_per_size": n_per_size}


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="MetaDrive baseline evaluation")
    parser.add_argument("--all", action="store_true", help="Run all sections")
    parser.add_argument("--topology", action="store_true", help="Topological validity")
    parser.add_argument("--route", action="store_true", help="Route coverage")
    parser.add_argument("--geometry", action="store_true", help="Geometric validity")
    parser.add_argument("--scalability", action="store_true", help="Large-scale capability")
    parser.add_argument("--num_maps", type=int, default=200, help="Maps for topology/route")
    parser.add_argument("--n_per_size", type=int, default=10, help="Maps per size for scalability")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--map_config", type=int, default=7, help="MetaDrive map config")
    parser.add_argument(
        "--graph-method",
        type=str,
        default="legacy",
        choices=["legacy", "skeleton"],
        help="Graph extraction: 'legacy' (spatial cluster + degree-2 contract) "
        "or 'skeleton' (render polylines → skeletonise → graph)",
    )
    parser.add_argument(
        "--vis", type=str, default=None, help="Save mask+graph visualisation images to DIR"
    )
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output)
    vis_dir = Path(args.vis) if args.vis else None
    gm = args.graph_method
    run_all = args.all or not (args.topology or args.route or args.geometry or args.scalability)

    init_stats = get_resource_stats()
    peak_stats = dict(init_stats)

    if run_all:
        # Multi-config sweep: ensure each 5-node bin (5-60) gets ≥5 maps
        ref_deg = load_osm_reference_degree()
        from eval.metrics import compute_all_geometric_metrics

        target_bins = list(range(5, 61, 5))  # [5,10,15,...,60]
        bin_counts = {b: 0 for b in target_bins}

        def _bin(n):
            return ((n - 1) // 5) * 5 + 5  # round up to nearest 5

        params = [
            (3, "small"),
            (7, "medium"),
            (10, "medium"),
            (15, "large"),
            (20, "large"),
            (25, "large"),
        ]
        per_map = []
        t0 = time.time()

        for mc, _label in params:
            for seed in range(200):
                # Stop when all target bins have ≥5
                if all(c >= 5 for c in bin_counts.values()):
                    break
                try:
                    r = generate_metadrive_map(seed, map_config=mc, graph_method=gm)
                    G = r["graph"]
                except Exception:
                    continue
                if G.number_of_nodes() < 2:
                    continue
                nb = _bin(G.number_of_nodes())
                if nb in bin_counts and bin_counts[nb] >= 5:
                    continue  # this bin already full
                resource = get_resource_stats()
                topo = compute_topological_metrics(G, ref_avg_degree=ref_deg)
                route = compute_route_coverage(G, n_pairs=100, seed=seed)
                geo = compute_all_geometric_metrics(r["polylines"], G=G)
                scale = classify_scale(topo["node_count"])
                if vis_dir is not None:
                    cnt = len(list(Path(vis_dir).glob(f"md_{scale}_*.png")))
                    if cnt < 5:
                        save_vis(
                            r["polylines"],
                            G,
                            str(
                                Path(vis_dir)
                                / f"md_{scale}_N{G.number_of_nodes()}E{G.number_of_edges()}_{cnt}.png"
                            ),
                        )
                per_map.append(
                    {
                        "map_id": len(per_map),
                        "seed": seed,
                        "scale": scale,
                        "gen_time_s": round(r["generation_time"], 4),
                        **resource,
                        **topo,
                        **route,
                        **geo,
                    }
                )
                if nb in bin_counts:
                    bin_counts[nb] += 1
                if len(per_map) % 10 == 0:
                    filled = sum(1 for c in bin_counts.values() if c >= 5)
                    print(f"  [{len(per_map)} maps] bins filled: {filled}/{len(target_bins)}")
            if all(c >= 5 for c in bin_counts.values()):
                break

        gen_time = time.time() - t0
        print(f"  Bin counts: {dict(bin_counts)}")
        # Save combined CSV
        import csv

        cols = [
            "map_id",
            "scale",
            "n_nodes",
            "n_edges",
            "lcc",
            "dead_end_ratio",
            "avg_degree",
            "delta_avg_degree",
            "reachable_ratio",
            "avg_route_length",
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
            "gen_time_s",
            "cpu_percent",
            "mem_mb",
            "gpu_mem_mb",
        ]
        key_map = {"n_nodes": "node_count", "n_edges": "edge_count"}
        with open(output_dir / "all_metrics.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for m in per_map:
                w.writerow([m.get(key_map.get(c, c), "") for c in cols])
        print(f"  Saved {len(per_map)} maps to {output_dir / 'all_metrics.csv'}")
        print(f"  Generation: {gen_time:.1f}s total, {gen_time/max(len(per_map),1):.1f}s/map")
        final_stats = get_resource_stats()
        for k in peak_stats:
            peak_stats[k] = max(peak_stats[k], final_stats[k])
        save_system_info(output_dir, "MetaDrive", init_stats, peak_stats, len(per_map))
    else:
        if args.topology:
            run_topology_eval(
                args.num_maps, args.seed, args.map_config, gm, output_dir, vis_dir=vis_dir
            )
        if args.route:
            run_route_eval(
                args.num_maps, args.seed, args.map_config, graph_method=gm, output_dir=output_dir
            )
        if args.geometry:
            run_geometry_eval(args.num_maps, args.seed, args.map_config, gm, output_dir)
        if args.scalability:
            run_scalability_eval(args.n_per_size, graph_method=gm, output_dir=output_dir)
        final_stats = get_resource_stats()
        for k in peak_stats:
            peak_stats[k] = max(peak_stats[k], final_stats[k])
        save_system_info(output_dir, "MetaDrive", init_stats, peak_stats, args.num_maps)

    print(f"\n{'=' * 60}")
    print(f"MetaDrive evaluation complete — results in {output_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

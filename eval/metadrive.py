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
    extract_intersection_graph,
    is_metadrive_road_node,
    load_csv_keys,
    load_csv_rows,
    load_osm_reference_degree,
    monitor_resources,
    save_binned_summary,
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
        # Merge distance in pixels at 1024×1024 resolution.
        # Tuned: merge_d=15 gives mc=15→~10 nodes, mc=25→~58 nodes.
        merge_m = 15.0
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
#  CLI
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="MetaDrive baseline evaluation")
    # --all is the only evaluation mode (kept for backward compatibility).
    parser.add_argument("--all", action="store_true", help="Run all sections (default)")
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

    output_dir = Path(args.output).resolve()
    vis_dir = Path(args.vis) if args.vis else None
    gm = args.graph_method

    # The all-metrics binned evaluation is the only mode; always save visualizations.
    # NOTE: legacy extract_intersection_graph() collapses to only 4-6 nodes for
    # ANY map_config (degree-2 contraction too aggressive), so the 10-80 node
    # target can never be reached.  Force skeleton extraction, which produces
    # well-spread 10-80 node graphs.  (export_40node_maps.py has the same
    # legacy limitation and cannot find 40-node maps with it either.)
    run_all = True
    if run_all:
        gm = "skeleton"
    if vis_dir is None:
        vis_dir = output_dir / "vis"

    if run_all:
        output_dir.mkdir(parents=True, exist_ok=True)
        # ── Multi-config sweep: ensure each 5-node bin (10-80) gets ≥5 maps ──
        ref_deg = load_osm_reference_degree()

        target_bins = list(range(10, 61, 5))  # [10,15,...,60]
        bin_counts = {b: 0 for b in target_bins}

        def _bin(n):
            return min((n // 5) * 5, 60)  # bins [10,15),[15,20),...,[55,60]; clamp at 60

        # NOTE: The params sweep below was tuned for skeleton extraction
        # (merge_d=15: mc=7→~5-10, mc=15→~10-20, mc=25→~40-60, mc=40→~80+).
        # With the default "legacy" graph method (30m cluster + degree-2
        # contract, matching scripts/export_40node_maps.py) the per-config
        # node counts are smaller — re-tune if bins don't fill.
        params = [
            (7, "small"),
            (10, "small"),
            (15, "medium"),
            (20, "medium"),
            (25, "large"),
            (30, "large"),
            (35, "large"),
            (40, "large"),
            (45, "large"),
            (50, "large"),
        ]

        # ── Resume: load existing CSV ─────────────────────────────────────
        csv_path = output_dir / "all_metrics.csv"
        seen_keys = load_csv_keys(csv_path, ["map_config", "seed"])
        if seen_keys:
            print(f"  Resuming: {len(seen_keys)} existing maps loaded from {csv_path.name}")

        # Count existing rows to continue map_id
        existing_count = 0
        if csv_path.exists():
            with open(csv_path) as _f:
                existing_count = max(0, sum(1 for _ in _f) - 1)

        per_map = []
        t0 = time.time()

        for mc, _label in params:
            consecutive_stall = 0
            for seed in range(200):
                key = f"{mc}|{seed}"
                if key in seen_keys:
                    continue  # already done

                # Stop when all target bins have ≥5
                if all(c >= 10 for c in bin_counts.values()):
                    break
                try:
                    with monitor_resources(interval=0.3) as peaks:
                        r = generate_metadrive_map(seed, map_config=mc, graph_method=gm)
                        G = r["graph"]
                    if G.number_of_nodes() < 2:
                        continue
                except Exception:
                    # Invalid map_config → skip to next config
                    consecutive_stall = 999
                    break

                nb = _bin(G.number_of_nodes())
                if nb < target_bins[0]:  # skip small maps (bin 5)
                    consecutive_stall += 1
                    if consecutive_stall >= 30:
                        print(
                            f"    map_config={mc}: stall ({consecutive_stall} seeds "
                            f"no progress), skipping"
                        )
                        break
                    continue
                if nb > target_bins[-1]:  # skip maps above 80-node target
                    consecutive_stall += 1
                    if consecutive_stall >= 30:
                        print(
                            f"    map_config={mc}: stall ({consecutive_stall} seeds "
                            f"no progress), skipping"
                        )
                        break
                    continue
                if nb in bin_counts and bin_counts[nb] >= 10:
                    consecutive_stall += 1  # bin already full, no progress
                    if consecutive_stall >= 30:
                        print(
                            f"    map_config={mc}: stall ({consecutive_stall} seeds "
                            f"no progress), skipping"
                        )
                        break
                    continue

                # ── This map contributes to a bin → reset stall ──
                consecutive_stall = 0

                topo = compute_topological_metrics(G, ref_avg_degree=ref_deg)
                route = compute_route_coverage(G, n_pairs=100, seed=seed)
                cycle = compute_cycle_ratio(G)
                geo = compute_all_geometric_metrics(r["polylines"], G=G)
                scale = classify_scale(topo["node_count"])

                row = {
                    "map_id": existing_count + len(per_map),
                    "seed": seed,
                    "map_config": mc,
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
                    "gen_time_s": round(r["generation_time"], 4),
                    "cpu_peak": round(peaks["cpu_peak"], 1),
                    "mem_peak_mb": round(peaks["mem_peak_mb"], 1),
                    "gpu_peak_mb": round(peaks["gpu_peak_mb"], 1),
                }
                per_map.append(row)
                append_csv_row(csv_path, list(row.keys()), row)
                seen_keys.add(key)

                if nb in bin_counts:
                    bin_counts[nb] += 1

                if len(per_map) % 10 == 0:
                    filled = sum(1 for c in bin_counts.values() if c >= 10)
                    print(
                        f"  [{existing_count + len(per_map)} maps] "
                        f"bins filled: {filled}/{len(target_bins)}"
                    )

                # ── Vis (up to 5 per scale) ──
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

    print(f"\n{'=' * 60}")
    print(f"MetaDrive evaluation complete — results in {output_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

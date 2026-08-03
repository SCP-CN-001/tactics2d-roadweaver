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
    compute_cycle_ratio,
    compute_route_coverage,
    compute_subnode_uniformity,
    compute_topological_metrics,
    contract_degree2_nodes,
    load_csv_keys,
    load_csv_rows,
    load_osm_reference_degree,
    monitor_resources,
    save_binned_summary,
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
#  CLI
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="HDMapGen baseline evaluation")
    # --all is the only evaluation mode (kept for backward compatibility).
    parser.add_argument("--all", action="store_true", help="Run all sections (default)")
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

    output_dir = Path(args.output).resolve()
    vis_dir = Path(args.vis) if args.vis else None
    max_maps = args.max_maps if args.max_maps > 0 else 0  # 0 = all
    gm = args.graph_method

    # The all-metrics binned evaluation is the only mode; always save visualizations.
    run_all = True
    if vis_dir is None:
        vis_dir = output_dir / "vis"

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

        target_bins = list(range(10, 61, 5))  # [10,15,...,60]
        bin_counts = {b: 0 for b in target_bins}

        def _bin(n):
            return min((n // 5) * 5, 60)  # bins [10,15),[15,20),...,[55,60]; clamp at 60

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

            if all(c >= 10 for c in bin_counts.values()):
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
            if nb in bin_counts and bin_counts[nb] >= 10:
                continue

            # Single resource monitoring block for all metric computation
            with monitor_resources(interval=0.3) as peaks:
                topo = compute_topological_metrics(G, ref_avg_degree=ref_deg)
                route = compute_route_coverage(G, n_pairs=100, seed=idx)
                cycle = compute_cycle_ratio(G)
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
                filled = sum(1 for c in bin_counts.values() if c >= 10)
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

        # ── Per-bin aggregate summary (all maps incl. resumed) ──
        save_binned_summary(output_dir, "all_metrics", load_csv_rows(csv_path))

        gen_time = time.time() - t0
        n_total = existing_count + len(per_map)
        print(f"  Bin counts: {dict(bin_counts)}")
        print(f"  Processed {len(per_map)} new maps (→{n_total} total) in " f"{gen_time:.1f}s")

    print(f"\n{'=' * 60}")
    print(f"HDMapGen evaluation complete — results in {output_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

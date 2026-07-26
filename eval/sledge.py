#!/usr/bin/env python3
"""
SLEDGE baseline evaluation — compatible with other eval/*.py outputs.

SLEDGE: Synthesizing Simulation Environments for Driving Agents with
        Generative Models (ECCV 2024).  Uses an RVAE + Diffusion Transformer
        to generate lane-level road graphs from latent noise.

Sections (matching the paper experiments):
  1. Topological Validity   (Table 1)  — LCC, Dead-end Ratio, Δd̄
  2. Route-Level Coverage   (Table 3)  — Reachable Ratio, Avg Route Length
  3. Geometric Validity     (Table 2)  — Chamfer, Smoothness, Self-intersect, etc.

  (Large-scale is N/A — SLEDGE generates fixed-size local scene patches.)

Usage:
    conda activate road-weaver
    python eval/sledge.py --all                           # run all sections
    python eval/sledge.py --topology                      # topological only
    python eval/sledge.py --route                         # route coverage only
    python eval/sledge.py --geometry                      # geometric only
    python eval/sledge.py --regenerate                    # generate maps from model
    python eval/sledge.py --visualize                     # draw sample maps

    # With a pretrained checkpoint:
    python eval/sledge.py --regenerate \\
        --sledge_root /path/to/SLEDGE --checkpoint /path/to/ldm_checkpoint

Output:
    runtimes/sledge_eval/
        generated_lanes.pkl      — raw lane data from SLEDGE (for fast reload)
        topology.json             — topological validity results
        route_coverage.json       — route coverage results
        geometry.json             — geometric validity results
        sledge_map_{small,medium,large}.png  — visualization

Notes:
    - SLEDGE's lane graph is lane-level: each lane is a graph node, and edges
      represent lane-to-lane connections.  This is analogous to RoadGen's
      widget-level graph — both operate at a finer granularity than
      MetaDrive's intersection-level graph.
    - The diffusion model requires a GPU with sufficient VRAM (>= 8 GB).
    - The SLEDGE submodule (baselines/SLEDGE/) must be importable for --regenerate.
      For metric evaluation on pre-generated data, no SLEDGE imports are needed.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

from eval.metrics import (
    compute_all_geometric_metrics,
    compute_route_coverage,
    compute_topological_metrics,
    load_osm_reference_degree,
    print_results_table,
    save_results,
)

# ═══════════════════════════════════════════════════════════════════════════
#  Paths
# ═══════════════════════════════════════════════════════════════════════════

REPO = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO / "runtimes" / "sledge_eval"
LANE_PKL = OUTPUT_DIR / "generated_lanes.pkl"

# ═══════════════════════════════════════════════════════════════════════════
#  SLEDGE import guard
# ═══════════════════════════════════════════════════════════════════════════

_SLEDGE_AVAILABLE = False
"""Whether the sledge package could be imported."""


def _check_sledge_imports() -> bool:
    """Attempt to import SLEDGE modules; return True on success."""
    global _SLEDGE_AVAILABLE

    sledge_root = REPO / "baselines" / "SLEDGE"
    if not sledge_root.exists():
        print("  [WARN] SLEDGE submodule not found at baselines/SLEDGE/")
        return False

    # Clean any stale sledge package references from path/modules
    sys.path = [p for p in sys.path if "sledge" not in p]
    for mod in list(sys.modules.keys()):
        if mod.startswith("sledge") or mod.startswith("nuplan"):
            del sys.modules[mod]

    sledge_pkg = sledge_root / "sledge"  # the actual Python package
    sys.path.insert(0, str(sledge_root))
    sys.path.insert(0, str(sledge_pkg))

    try:
        from sledge.autoencoder.modeling.models.rvae.rvae_config import RVAEConfig  # noqa: F401

        _SLEDGE_AVAILABLE = True
        return True
    except ImportError as e:
        print(f"  [WARN] Cannot import SLEDGE autoencoder: {e}")
        print(f"         --regenerate will be unavailable; use pre-generated")
        print(f"         data from a pickle file instead.")
        return False


# ═══════════════════════════════════════════════════════════════════════════
#  Lane data helpers  (no SLEDGE imports needed)
# ═══════════════════════════════════════════════════════════════════════════

# Each entry in the pickle stores:
#   "num_lanes": int
#   "lines_states": np.ndarray (N, 20, 2)  — raw (x, y) per lane subnode
#   "lines_mask":    np.ndarray (N,)         — lane existence probability
#   "gen_time_s":  float


def _lane_data_to_undirected_graph(
    states: np.ndarray, mask: np.ndarray, threshold: float = 0.3, distance_thresh: float = 1.5
) -> nx.Graph:
    """Build an undirected lane graph from SledgeVector line data.

    Each valid lane (mask > threshold) becomes a graph node.  Two lanes
    are connected by an edge when the polyline endpoint of one is within
    *distance_thresh* of the other's start point.

    Returns
    -------
    nx.Graph
        Nodes are lane indices (int).  Edges represent lane-to-lane
        connectivity in the generated road network.
    """
    valid_idx = np.where(mask > threshold)[0]
    G = nx.Graph()
    if len(valid_idx) == 0:
        return G

    for idx in valid_idx:
        G.add_node(int(idx))

    starts = states[valid_idx, 0, :]  # (M, 2)
    ends = states[valid_idx, -1, :]  # (M, 2)

    from scipy.spatial.distance import cdist

    end_to_start = cdist(ends, starts)

    M = len(valid_idx)
    for i in range(M):
        for j in range(M):
            if i != j and end_to_start[i, j] < distance_thresh:
                G.add_edge(int(valid_idx[i]), int(valid_idx[j]))

    return G


def _lane_data_to_polylines(
    states: np.ndarray, mask: np.ndarray, threshold: float = 0.3
) -> list[np.ndarray]:
    """Extract lane polylines from SledgeVector line data.

    Returns a list of (N_i, 2) arrays for lanes with mask > threshold.
    """
    valid_idx = np.where(mask > threshold)[0]
    polylines = []
    for idx in valid_idx:
        pts = states[idx]
        if len(pts) >= 2:
            polylines.append(pts)
    return polylines


def load_lane_data() -> list[dict]:
    """Load pre-generated lane data from pickle."""
    import pickle

    with open(LANE_PKL, "rb") as f:
        return pickle.load(f)


def save_lane_data(data: list[dict]):
    """Save generated lane data to pickle."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    import pickle

    with open(LANE_PKL, "wb") as f:
        pickle.dump(data, f)
    print(f"    Saved {len(data)} lane graphs to {LANE_PKL}")


# ═══════════════════════════════════════════════════════════════════════════
#  Map generation via SLEDGE pipeline   (requires --regenerate)
# ═══════════════════════════════════════════════════════════════════════════


def _decode_one(pipeline, class_label: int, seed: int, threshold: float = 0.3) -> dict | None:
    """Run the SLEDGE diffusion pipeline once and return raw lane data.

    Returns dict with keys: num_lanes, lines_states, lines_mask, gen_time_s.
    Returns None on failure.
    """
    import torch

    t0 = time.time()
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    try:
        sledge_vectors = pipeline(class_labels=[class_label], generator=generator)
    except Exception as e:
        print(f"      [WARN] SLEDGE pipeline failed: {e}")
        return None

    gen_time = time.time() - t0
    sv = sledge_vectors[0].torch_to_numpy()

    lines_states = sv.lines.states  # (N, 20, 2)
    lines_mask = sv.lines.mask  # (N,)

    return {
        "num_lanes": lines_states.shape[0],
        "lines_states": lines_states,
        "lines_mask": lines_mask,
        "gen_time_s": round(gen_time, 4),
    }


def _build_pipeline(checkpoint_dir: str | Path, device: str = "cuda:0"):
    """Load the LDMPipeline from a diffusers-style checkpoint directory.

    The checkpoint must contain subfolders ``decoder/``, ``transformer/``,
    ``scheduler/`` and a ``model_index.json``.

    Requires the full SLEDGE package (+ nuplan-devkit) to be importable.
    """
    from sledge.diffusion.modelling.ldm_pipeline import LDMPipeline

    pipeline = LDMPipeline.from_pretrained(str(checkpoint_dir), use_safetensors=True)
    pipeline = pipeline.to(device)
    pipeline.eval()
    return pipeline


def regenerate_maps(
    num_maps: int = 200,
    checkpoint: str | None = None,
    device: str = "cuda:0",
    seed: int = 0,
    batch_size: int = 10,
) -> list[dict]:
    """Generate lane data by sampling the SLEDGE diffusion pipeline.

    Parameters
    ----------
    num_maps : int
        Number of maps to generate.
    checkpoint : str or None
        Path to a diffusers-style LDMPipeline checkpoint.  If None, tries
        ``baselines/SLEDGE/pretrained/ldm/``.
    device : str
        Torch device.
    seed : int
        Random seed for reproducibility.
    batch_size : int
        Number of maps generated per pipeline call (used for multi-class
        generation internally; we take the first output per seed).

    Returns
    -------
    list[dict]
        Raw lane data, one dict per map.
    """
    if not _SLEDGE_AVAILABLE:
        _check_sledge_imports()
    if not _SLEDGE_AVAILABLE:
        print("  [ERROR] SLEDGE package is required for --regenerate.")
        print("          Install the sledge submodule first, or use")
        print("          pre-generated lane data (load from pickle).")
        return []

    # Resolve checkpoint path
    if checkpoint is None:
        default_ckpt = REPO / "baselines" / "SLEDGE" / "pretrained" / "ldm"
        if default_ckpt.exists():
            checkpoint = str(default_ckpt)
        else:
            print("  [ERROR] No checkpoint specified and default path")
            print(f"          {default_ckpt} does not exist.")
            print("          Use --checkpoint /path/to/checkpoint")
            return []

    print(f"  Loading pipeline from {checkpoint} ...")
    pipeline = _build_pipeline(checkpoint, device=device)

    print(f"  Generating {num_maps} maps (seed={seed}, device={device})...")
    all_maps = []
    rng = np.random.default_rng(seed)

    t_start = time.time()
    for i in range(num_maps):
        # Alternate between available class labels for diversity
        class_label = int(rng.integers(0, 4))  # SLEDGE has 4 scene types
        result = _decode_one(pipeline, class_label, seed + i)
        if result is None:
            continue

        all_maps.append(result)

        if (i + 1) % max(1, num_maps // 20) == 0 or i == 0:
            elapsed = time.time() - t_start
            done = i + 1
            n_lanes = result["num_lanes"]
            eta_sec = (elapsed / done) * (num_maps - done) if done > 0 else 0
            import datetime

            eta = str(datetime.timedelta(seconds=int(eta_sec)))
            print(f"    [{i + 1}/{num_maps}]  lanes={n_lanes}  " f"ETA {eta}")

    print(f"  Generated {len(all_maps)} / {num_maps} maps " f"({time.time() - t_start:.1f}s total)")
    save_lane_data(all_maps)
    return all_maps


# ═══════════════════════════════════════════════════════════════════════════
#  Section 1: Topological Validity  (Table 1)
# ═══════════════════════════════════════════════════════════════════════════


def run_topology_eval(
    num_maps: int | None = None, seed: int = 0, output_dir: Path = OUTPUT_DIR
) -> dict:
    """Load lane data and compute topological validity."""
    print(f"\n{'=' * 60}")
    print("Section 1: Topological Validity  (SLEDGE)")
    print(f"{'=' * 60}")

    data = load_lane_data()
    if num_maps is not None:
        data = data[:num_maps]
    print(f"  Loaded {len(data)} lane graphs")

    ref_deg = load_osm_reference_degree()
    print(f"  OSM reference avg_degree = {ref_deg:.4f}")

    topo_metrics = defaultdict(list)
    all_results = []

    for idx, entry in enumerate(data):
        G = _lane_data_to_undirected_graph(entry["lines_states"], entry["lines_mask"])
        if G.number_of_nodes() < 2:
            continue

        topo = compute_topological_metrics(G, ref_avg_degree=ref_deg)
        for k, v in topo.items():
            topo_metrics[k].append(v)

        all_results.append(
            {
                "map_id": idx,
                "num_lanes": entry["num_lanes"],
                "valid_lanes": G.number_of_nodes(),
                "gen_time_s": entry["gen_time_s"],
                **topo,
            }
        )

    if not all_results:
        print("  [ERROR] No valid lane graphs found. Aborting.")
        return {}

    agg = {}
    for k, v in topo_metrics.items():
        agg[f"{k}_mean"] = float(np.mean(v))
        agg[f"{k}_std"] = float(np.std(v))
    agg["n_maps"] = len(all_results)
    agg["osm_reference_avg_degree"] = ref_deg

    save_results(output_dir, "topology", agg, all_results)
    print_results_table(
        "Topological Validity (SLEDGE)",
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
    num_maps: int | None = None, n_pairs: int = 100, seed: int = 0, output_dir: Path = OUTPUT_DIR
) -> dict:
    """Load lane data and compute route coverage."""
    print(f"\n{'=' * 60}")
    print("Section 2: Route-Level Coverage  (SLEDGE)")
    print(f"  {n_pairs} OD pairs per graph")
    print(f"{'=' * 60}")

    data = load_lane_data()
    if num_maps is not None:
        data = data[:num_maps]
    print(f"  Loaded {len(data)} lane graphs")

    route_metrics = defaultdict(list)
    all_results = []

    for idx, entry in enumerate(data):
        G = _lane_data_to_undirected_graph(entry["lines_states"], entry["lines_mask"])
        if G.number_of_nodes() < 2:
            continue

        route = compute_route_coverage(G, n_pairs=n_pairs, seed=seed + idx)
        for k, v in route.items():
            route_metrics[k].append(v)

        all_results.append(
            {
                "map_id": idx,
                "num_lanes": entry["num_lanes"],
                "valid_lanes": G.number_of_nodes(),
                **route,
            }
        )

    if not all_results:
        print("  [ERROR] No valid lane graphs found. Aborting.")
        return {}

    agg = {}
    for k, v in route_metrics.items():
        agg[f"{k}_mean"] = float(np.mean(v))
        agg[f"{k}_std"] = float(np.std(v))
    agg["n_maps"] = len(all_results)

    save_results(output_dir, "route_coverage", agg, all_results)
    print_results_table(
        "Route Coverage (SLEDGE)",
        agg,
        [("reachable_ratio", "Reachable", ".4f"), ("avg_route_length", "Avg Length", ".1f")],
    )

    return agg


# ═══════════════════════════════════════════════════════════════════════════
#  Section 3: Geometric Validity  (Table 2)
# ═══════════════════════════════════════════════════════════════════════════


def run_geometry_eval(num_maps: int | None = None, output_dir: Path = OUTPUT_DIR) -> dict:
    """Load lane data and compute all geometric metrics."""
    print(f"\n{'=' * 60}")
    print("Section 3: Geometric Validity  (SLEDGE)")
    print(f"  (Edge smoothness, self-intersection, length CV, uniformity)")
    print(f"{'=' * 60}")

    data = load_lane_data()
    if num_maps is not None:
        data = data[:num_maps]
    print(f"  Loaded {len(data)} lane graphs")

    geom_metrics = defaultdict(list)
    all_results = []

    for entry in data:
        polylines = _lane_data_to_polylines(entry["lines_states"], entry["lines_mask"])
        if not polylines:
            continue

        G = _lane_data_to_undirected_graph(entry["lines_states"], entry["lines_mask"])
        geo = compute_all_geometric_metrics(polylines, G=G)

        entry_out = {"num_lanes": entry["num_lanes"], "num_valid_lanes": len(polylines), **geo}
        for k, v in geo.items():
            if isinstance(v, (int, float, np.floating, np.integer)):
                geom_metrics[k].append(v)
        all_results.append(entry_out)

    if not all_results:
        print("  [ERROR] No valid lane geometry found. Aborting.")
        return {}

    agg = {}
    for k, v in geom_metrics.items():
        agg[f"{k}_mean"] = float(np.mean(v))
        agg[f"{k}_std"] = float(np.std(v))
    agg["n_maps"] = len(all_results)

    save_results(output_dir, "geometry", agg, all_results)
    print_results_table(
        "Geometric Validity (SLEDGE)",
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
#  Visualization  (consistent with other baselines)
# ═══════════════════════════════════════════════════════════════════════════


def draw_sledge_maps(output_dir: Path = OUTPUT_DIR):
    """Draw 3 sample SLEDGE maps (small / medium / large lane counts) as PNG.

    Uses the same matplotlib style as *analysis/viz_unified.py* so that
    the visual appearance is consistent across baselines.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load_lane_data()
    if not data:
        print("  [WARN] No lane data to visualize. Run --regenerate first.")
        return

    # Sort by number of valid lanes and pick small / medium / large
    def _valid_count(d):
        return int(np.sum(d["lines_mask"] > 0.3))

    sorted_data = sorted(data, key=_valid_count)
    n = len(sorted_data)
    picks = {
        "small": sorted_data[max(0, n // 6)],
        "medium": sorted_data[n // 2],
        "large": sorted_data[min(n - 1, 5 * n // 6)],
    }

    # Consistent style with viz_unified.py
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

    SLEDGE_COLOR = "#2ECC71"  # green — distinct from MetaDrive (blue),
    # HDMapGen (red), RoadGen (green-brown)

    for label, entry in picks.items():
        polylines = _lane_data_to_polylines(entry["lines_states"], entry["lines_mask"])
        G = _lane_data_to_undirected_graph(entry["lines_states"], entry["lines_mask"])

        fig, ax = plt.subplots(figsize=(8, 6))

        # ── Lane geometry ──
        for pts in polylines:
            ax.plot(pts[:, 0], pts[:, 1], color=SLEDGE_COLOR, lw=2.5, alpha=0.85)

        # ── Lane graph nodes as dots ──
        # For each lane, plot its midpoint
        for pts in polylines:
            mid = pts[len(pts) // 2]
            ax.scatter(
                mid[0], mid[1], c="#1A5C34", s=15, zorder=3, edgecolors="white", linewidths=0.5
            )

        ax.set_title(
            f"SLEDGE — {label.capitalize()}  "
            f"({G.number_of_nodes()} lanes, "
            f"{G.number_of_edges()} connections)"
        )
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

        filepath = output_dir / f"sledge_map_{label}.png"
        fig.tight_layout()
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {filepath}")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════


def _print_timing_estimate(args, run_all):
    """Print a rough wall-clock estimate before starting."""
    maps_topo = args.num_maps if (run_all or args.topology) else 0
    maps_route = args.num_maps if (run_all or args.route) else 0
    maps_geom = args.num_maps if (run_all or args.geometry) else 0

    total_sections = (
        (1 if args.topology else 0) + (1 if args.route else 0) + (1 if args.geometry else 0)
    )
    if run_all:
        total_sections = 3

    if total_sections == 0 and not args.regenerate and not args.visualize:
        return

    if args.regenerate:
        print(f"  [Note] SLEDGE generation requires a GPU and a pretrained")
        print(f"         checkpoint.  Estimated ~0.5-2s per map.")
        print(f"         Use --checkpoint to specify the model path.\n")


def main():
    parser = argparse.ArgumentParser(
        description="SLEDGE baseline evaluation — topological, route, and "
        "geometric validity metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python eval/sledge.py --all\n"
            "  python eval/sledge.py --topology --num_maps 200\n"
            "  python eval/sledge.py --regenerate --checkpoint /path/ckpt\n"
            "  python eval/sledge.py --visualize\n"
        ),
    )

    # Section selection
    parser.add_argument(
        "--all", action="store_true", help="Run all sections (topology + route + geometry)"
    )
    parser.add_argument("--topology", action="store_true", help="Topological validity (Table 1)")
    parser.add_argument("--route", action="store_true", help="Route coverage (Table 3)")
    parser.add_argument("--geometry", action="store_true", help="Geometric validity (Table 2)")
    parser.add_argument(
        "--regenerate", action="store_true", help="Generate maps from SLEDGE model checkpoint"
    )
    parser.add_argument("--visualize", action="store_true", help="Draw sample maps as PNG")

    # Configuration
    parser.add_argument(
        "--num_maps", type=int, default=200, help="Number of maps for evaluation (default: 200)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to SLEDGE LDMPipeline checkpoint"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Torch device (default: cuda:0)"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()
    output_dir = Path(args.output)
    run_all = args.all or not (
        args.topology or args.route or args.geometry or args.regenerate or args.visualize
    )

    _print_timing_estimate(args, run_all)

    # ── Optional: probe SLEDGE imports early if regeneration is requested ──
    if args.regenerate:
        _check_sledge_imports()

    # ── Regenerate if requested (or if no pickle exists yet) ──
    if args.regenerate or (run_all and not LANE_PKL.exists()):
        if not args.regenerate:
            print(f"\n  No pre-generated lane data at {LANE_PKL}.")
            print("  Attempting generation from model checkpoint...")
        data = regenerate_maps(
            num_maps=args.num_maps, checkpoint=args.checkpoint, device=args.device, seed=args.seed
        )
        if not data:
            print("  [ERROR] Lane data generation failed.  Aborting.")
            return

    # ── Run evaluation sections ──
    if run_all or args.topology:
        if not LANE_PKL.exists():
            print(f"  [ERROR] No lane data at {LANE_PKL}. " f"Use --regenerate first.")
            return
        run_topology_eval(num_maps=args.num_maps, seed=args.seed, output_dir=output_dir)

    if run_all or args.route:
        if not LANE_PKL.exists():
            print(f"  [ERROR] No lane data at {LANE_PKL}. " f"Use --regenerate first.")
            return
        run_route_eval(num_maps=args.num_maps, seed=args.seed, output_dir=output_dir)

    if run_all or args.geometry:
        if not LANE_PKL.exists():
            print(f"  [ERROR] No lane data at {LANE_PKL}. " f"Use --regenerate first.")
            return
        run_geometry_eval(num_maps=args.num_maps, output_dir=output_dir)

    # ── Visualization ──
    if args.visualize or run_all:
        if LANE_PKL.exists():
            draw_sledge_maps(output_dir)
        else:
            print("  [SKIP] --visualize requires lane data; run --regenerate")

    if run_all or any([args.topology, args.route, args.geometry]):
        print(f"\n{'=' * 60}")
        print(f"SLEDGE evaluation complete. Results saved to {output_dir}/")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

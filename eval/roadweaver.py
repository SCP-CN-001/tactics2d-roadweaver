#!/usr/bin/env python3
# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""RoadWeaver baseline evaluation."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import (  # noqa: E402
    append_csv_row,
    classify_scale,
    compute_all_geometric_metrics,
    compute_cycle_ratio,
    compute_route_coverage,
    compute_topological_metrics,
    get_resource_stats,
    load_csv_keys,
    load_csv_rows,
    load_osm_reference_degree,
    monitor_resources,
    save_binned_summary,
    save_system_info,
)
from eval.polyline_graph import save_vis  # noqa: E402
from network_generator import run_pipeline  # noqa: E402
from network_generator.backbone.config import CONFIG  # noqa: E402
from network_generator.backbone.dataset import make_field_dataloader  # noqa: E402
from network_generator.backbone.generator import make_generator  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════
#  Paths & defaults
# ═══════════════════════════════════════════════════════════════════════════

REPO = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO / "runtimes" / "roadweaver_eval"

VQ_CKPT = "runtimes/vq_vae_2km_phase_a/checkpoints/best.pth"
TFM_CKPT = "runtimes/transformer_2km_phase_a/checkpoints/best.pth"
CACHE = "cache/masked_code_maps_2km_phase_a/train.npz"
VQ_MAP_M = 2000.0  # VQ native map size (m)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Organic 6-dim style vector (soft one-hot over the 6 patterns).
STYLE_ORGANIC = [0.05, 0.05, 0.0, 0.8, 0.05, 0.05]

# Metrics shown in the per-bin report (subset of full row, order matches
# docs/baseline-eval.md tables).
ROW_FIELDS = [
    ("lcc", "LCC", ".4f"),
    ("dead_end_ratio", "Dead-end", ".4f"),
    ("cycle_ratio", "Cycle", ".4f"),
    ("avg_degree", "Avg Deg", ".3f"),
    ("reachable_ratio", "Reachable", ".4f"),
    ("endpoint_alignment", "Endpoint Align", ".4f"),
    ("chamfer_loo", "Chamfer LOO", ".4f"),
    ("mean_edge_length", "Edge Len", ".2f"),
    ("mean_turning_angle_deg", "Turn Angle", ".2f"),
]


# ═══════════════════════════════════════════════════════════════════════════
#  Generation helpers
# ═══════════════════════════════════════════════════════════════════════════


def load_models(vq_ckpt: str, tfm_ckpt: str, cache: str, device: str = DEVICE):
    """Load the Phase A generator and return it."""
    CONFIG.resolution = 128
    CONFIG.code_map_size = 32
    CONFIG.val_split_path = "data/urban_prior/2km/splits_style/val.parquet"
    return make_generator(vq_ckpt, tfm_ckpt, cache_path=cache, device=device)


def build_condition(
    density: float, gridness=0.25, radialness=0.10, organic=0.75, be=0.70
) -> torch.Tensor:
    """11-dim condition = [style_vector(6) | structural_priors(5)]."""
    cond = torch.zeros(11)
    cond[:6] = torch.tensor(STYLE_ORGANIC)
    cond[6] = density  # road_density_km_per_km2
    cond[7] = gridness
    cond[8] = radialness
    cond[9] = organic
    cond[10] = be  # bearing_entropy
    return cond


def generate_one_map(
    gen,
    condition: torch.Tensor,
    struct: torch.Tensor,
    seed: int,
    map_w: float,
    map_h: float,
    name: str = "rw_eval",
) -> dict:
    """Run the full pipeline once, return the branch dict + timing.

    The generator's ``torch.multinomial`` sampling uses the *global* torch RNG,
    so ``torch.manual_seed`` must be called before the pipeline call.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    t0 = time.time()
    result = run_pipeline(
        gen,
        condition,
        struct,
        map_w=map_w,
        map_h=map_h,
        vq_map_size_m=VQ_MAP_M,
        seed=seed,
        name=name,
        scenario_type="urban",
        assemble_hdmap=False,
    )
    gen_time = time.time() - t0
    return {"branch": result["branch"], "gen_time": gen_time}


def branch_to_graph(branch: dict, map_w: float, map_h: float) -> tuple[nx.Graph, list[np.ndarray]]:
    """Convert branch output to an intersection graph + polylines (metres).

    ``coords_int`` / ``geometries`` are normalised [0, 1] relative to the map;
    they are scaled by ``[map_w, map_h]`` so all geometric metrics operate in
    the same metre-scale space as the other baselines.
    """
    scale = np.array([map_w, map_h])
    coords_m = np.asarray(branch["coords_int"], dtype=np.float64) * scale
    ei = branch["edge_index_int"]

    G = nx.Graph()
    for i, xy in enumerate(coords_m):
        G.add_node(i, coords=[float(xy[0]), float(xy[1])])
    for u, v in ei:
        if int(u) == int(v):  # roundabout ring closure → self-loop; strip it
            continue
        G.add_edge(int(u), int(v))

    polys = [
        np.asarray(g, dtype=np.float64) * scale for g in branch.get("geometries", []) if len(g) >= 2
    ]
    return G, polys


def compute_row(
    branch: dict,
    gen_time: float,
    peaks: dict,
    ref_deg: float,
    *,
    mode: str,
    seed: int,
    density: float,
    map_w: float,
    map_h: float,
    map_id: int,
) -> dict:
    """Compute the full metric row for one generated map (reuses eval.metrics)."""
    G, polys = branch_to_graph(branch, map_w, map_h)
    n = G.number_of_nodes()
    nb = (n // 5) * 5

    topo = compute_topological_metrics(G, ref_avg_degree=ref_deg)
    cycle = compute_cycle_ratio(G)
    route = compute_route_coverage(G, n_pairs=100, seed=seed)
    geo = compute_all_geometric_metrics(polys, G=G)

    return {
        "map_id": map_id,
        "mode": mode,
        "seed": seed,
        "density": round(float(density), 3),
        "node_bin": nb,
        "n_nodes": topo["node_count"],
        "n_edges": topo["edge_count"],
        "lcc": topo["lcc"],
        "dead_end_ratio": topo["dead_end_ratio"],
        "cycle_ratio": cycle["cycle_ratio"],
        "avg_degree": topo["avg_degree"],
        "delta_avg_degree": topo.get("delta_avg_degree", ""),
        "reachable_ratio": route["reachable_ratio"],
        "avg_route_length": route["avg_route_length"],
        "endpoint_alignment": geo.get("endpoint_alignment", ""),
        "chamfer_loo": geo.get("chamfer_loo", ""),
        "mean_turning_angle_deg": geo.get("mean_turning_angle_deg", ""),
        "crossings_per_km": geo.get("crossings_per_km", ""),
        "mean_edge_length": geo.get("mean_edge_length", ""),
        "cv_edge_length": geo.get("cv_edge_length", ""),
        "total_road_length": geo.get("total_road_length", ""),
        "mean_spacing_cv": geo.get("mean_spacing_cv", ""),
        "mean_angle_deg": geo.get("mean_angle_deg", ""),
        "gen_time_s": round(gen_time, 4),
        "cpu_peak": round(peaks["cpu_peak"], 1),
        "mem_peak_mb": round(peaks["mem_peak_mb"], 1),
        "gpu_peak_mb": round(peaks["gpu_peak_mb"], 1),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Mode A — density-sweep scalability bins (matches docs/baseline-eval.md)
# ═══════════════════════════════════════════════════════════════════════════


def run_bins(
    gen,
    *,
    output_dir: Path,
    vis_dir: Path | None,
    map_w: float,
    map_h: float,
    target_per_bin: int,
    seeds_per_density: int,
    base_seed: int,
) -> None:
    """Fill 10–80 node bins with fixed organic style + density sweep."""
    ref_deg = load_osm_reference_degree()
    target_bins = list(range(10, 81, 5))  # [10,15,...,80]
    bin_counts = {b: 0 for b in target_bins}
    densities = list(range(2, 23))  # density 2..22 → roughly 10..150 nodes

    def _bin(n):
        return (n // 5) * 5

    csv_path = output_dir / "all_metrics.csv"
    seen_keys = load_csv_keys(csv_path, ["density", "seed"])
    if seen_keys:
        print(f"  Resuming: {len(seen_keys)} existing maps loaded from {csv_path.name}")

    existing_count = 0
    if csv_path.exists():
        with open(csv_path) as _f:
            existing_count = max(0, sum(1 for _ in _f) - 1)

    # Initialise bin counts from resumed rows so already-full bins are skipped.
    for row in load_csv_rows(csv_path):
        b = int(float(row.get("node_bin", 0)))
        if b in bin_counts:
            bin_counts[b] += 1

    n_new = 0
    t0 = time.time()
    init_stats = get_resource_stats()
    peak_stats = {"cpu_percent": 0.0, "mem_mb": 0.0, "gpu_mem_mb": 0.0}

    for density in densities:
        condition = build_condition(float(density)).to(DEVICE)
        struct = condition[6:11]
        consecutive_stall = 0
        for k in range(seeds_per_density):
            seed = base_seed + k
            key = f"{density}|{seed}"
            if key in seen_keys:
                continue
            if all(c >= target_per_bin for c in bin_counts.values()):
                break

            try:
                with monitor_resources(interval=0.3) as peaks:
                    r = generate_one_map(gen, condition, struct, seed, map_w, map_h)
                    branch = r["branch"]
                n = len(branch["coords_int"])
                if n < 2:
                    continue
            except Exception as e:
                print(f"    density={density} seed={seed} FAILED: {e}")
                consecutive_stall += 1
                if consecutive_stall >= 10:
                    break
                continue

            nb = _bin(n)
            if nb < target_bins[0] or nb > target_bins[-1]:
                consecutive_stall += 1
            elif bin_counts.get(nb, 0) >= target_per_bin:
                consecutive_stall += 1
            else:
                consecutive_stall = 0
                row = compute_row(
                    branch,
                    r["gen_time"],
                    peaks,
                    ref_deg,
                    mode="bins",
                    seed=seed,
                    density=density,
                    map_w=map_w,
                    map_h=map_h,
                    map_id=existing_count + n_new,
                )
                n_new += 1
                peak_stats["cpu_percent"] = max(peak_stats["cpu_percent"], peaks["cpu_peak"])
                peak_stats["mem_mb"] = max(peak_stats["mem_mb"], peaks["mem_peak_mb"])
                peak_stats["gpu_mem_mb"] = max(peak_stats["gpu_mem_mb"], peaks["gpu_peak_mb"])
                append_csv_row(csv_path, list(row.keys()), row)
                seen_keys.add(key)
                bin_counts[nb] = bin_counts.get(nb, 0) + 1

                if vis_dir is not None:
                    scale = classify_scale(row["n_nodes"])
                    cnt = len(list(Path(vis_dir).glob(f"rw_{scale}_*.png")))
                    if cnt < 5:
                        G, polys = branch_to_graph(branch, map_w, map_h)
                        save_vis(
                            polys,
                            G,
                            str(
                                Path(vis_dir)
                                / f"rw_{scale}_N{row['n_nodes']}E{row['n_edges']}_{cnt}.png"
                            ),
                        )

                if n_new % 10 == 0:
                    filled = sum(1 for c in bin_counts.values() if c >= target_per_bin)
                    print(
                        f"  [{existing_count + n_new} maps] bins filled: {filled}/{len(target_bins)}"
                    )

            if consecutive_stall >= 30:
                print(
                    f"    density={density}: stall ({consecutive_stall} seeds no progress), skipping"
                )
                break

        if all(c >= target_per_bin for c in bin_counts.values()):
            break

    save_binned_summary(output_dir, "all_metrics", load_csv_rows(csv_path))
    save_system_info(output_dir, "roadweaver", init_stats, peak_stats, existing_count + n_new)

    print(f"  Bin counts: {dict(bin_counts)}")
    print(
        f"  Generated {n_new} new maps (→{existing_count + n_new} total) in "
        f"{time.time() - t0:.1f}s, {(time.time() - t0) / max(n_new, 1):.1f}s/map"
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Mode B — diverse-style generation (val dataloader conditions)
# ═══════════════════════════════════════════════════════════════════════════


def run_styles(
    gen,
    *,
    output_dir: Path,
    vis_dir: Path | None,
    num_styles: int,
    map_w: float,
    map_h: float,
    base_seed: int,
) -> None:
    """Generate maps from val-dataloader style conditions; report aggregates."""
    ref_deg = load_osm_reference_degree()
    dl = make_field_dataloader(
        "val", batch_size=num_styles, num_workers=0, limit_samples=num_styles, cache_fields=False
    )
    batch = next(iter(dl))
    style = batch["style_vector"]
    struct = batch["structural_priors"]
    cond = torch.cat([style, struct], dim=1).to(DEVICE)

    csv_path = output_dir / "styles_metrics.csv"
    seen_keys = load_csv_keys(csv_path, ["seed"])

    existing_count = 0
    if csv_path.exists():
        with open(csv_path) as _f:
            existing_count = max(0, sum(1 for _ in _f) - 1)

    per_map = []
    t0 = time.time()
    init_stats = get_resource_stats()
    peak_stats = {"cpu_percent": 0.0, "mem_mb": 0.0, "gpu_mem_mb": 0.0}
    for i in range(num_styles):
        seed = base_seed + i
        if str(seed) in seen_keys:
            continue
        try:
            with monitor_resources(interval=0.3) as peaks:
                r = generate_one_map(
                    gen, cond[i], struct[i], seed, map_w, map_h, name=f"rw_style_{i}"
                )
                branch = r["branch"]
            if len(branch["coords_int"]) < 2:
                continue
        except Exception as e:
            print(f"  style sample {i} (seed={seed}) FAILED: {e}")
            continue

        row = compute_row(
            branch,
            r["gen_time"],
            peaks,
            ref_deg,
            mode="styles",
            seed=seed,
            density=float(struct[i, 0]),
            map_w=map_w,
            map_h=map_h,
            map_id=existing_count + len(per_map),
        )
        per_map.append(row)
        peak_stats["cpu_percent"] = max(peak_stats["cpu_percent"], peaks["cpu_peak"])
        peak_stats["mem_mb"] = max(peak_stats["mem_mb"], peaks["mem_peak_mb"])
        peak_stats["gpu_mem_mb"] = max(peak_stats["gpu_mem_mb"], peaks["gpu_peak_mb"])
        append_csv_row(csv_path, list(row.keys()), row)
        seen_keys.add(str(seed))

        if vis_dir is not None:
            scale = classify_scale(row["n_nodes"])
            cnt = len(list(Path(vis_dir).glob(f"rw_style_{scale}_*.png")))
            if cnt < 5:
                G, polys = branch_to_graph(branch, map_w, map_h)
                save_vis(
                    polys,
                    G,
                    str(
                        Path(vis_dir)
                        / f"rw_style_{scale}_N{row['n_nodes']}E{row['n_edges']}_{cnt}.png"
                    ),
                )

    # Aggregate over all generated style maps.
    rows = load_csv_rows(csv_path)
    save_binned_summary(output_dir, "styles_metrics", rows)
    save_system_info(output_dir, "roadweaver_styles", init_stats, peak_stats, len(rows))
    print(f"\n  Styles: {len(rows)} maps in {time.time() - t0:.1f}s")
    if rows:

        def _mean(key):
            vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
            return float(np.mean(vals)) if vals else float("nan")

        print("  Aggregate (mean over all style maps):")
        for key, name, fmt in ROW_FIELDS:
            print(f"    {name:16s}  {_mean(key):{fmt}}")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="RoadWeaver baseline evaluation")
    parser.add_argument("--all", action="store_true", help="Run bins + styles modes")
    parser.add_argument("--bins", action="store_true", help="Density-sweep scalability bins")
    parser.add_argument("--styles", action="store_true", help="Diverse-style generation pass")
    parser.add_argument("--num-styles", type=int, default=36, help="Style samples (styles mode)")
    parser.add_argument("--target-per-bin", type=int, default=10, help="Maps per node bin")
    parser.add_argument("--seeds-per-density", type=int, default=60, help="Max seeds per density")
    parser.add_argument("--map-w", type=float, default=2000.0)
    parser.add_argument("--map-h", type=float, default=2000.0)
    parser.add_argument("--vq-ckpt", default=VQ_CKPT)
    parser.add_argument("--model-ckpt", default=TFM_CKPT)
    parser.add_argument("--cache", default=CACHE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vis", type=str, default=None)
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    do_bins = args.bins or args.all or not args.styles
    do_styles = args.styles or args.all

    output_dir = Path(args.output).resolve()
    vis_dir = Path(args.vis) if args.vis else output_dir / "vis"
    output_dir.mkdir(parents=True, exist_ok=True)
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    gen = load_models(args.vq_ckpt, args.model_ckpt, args.cache)
    print(f"  Device: {DEVICE}")

    if do_bins:
        print(f"\n{'=' * 60}\nMode A: density-sweep scalability bins\n{'=' * 60}")
        run_bins(
            gen,
            output_dir=output_dir,
            vis_dir=vis_dir,
            map_w=args.map_w,
            map_h=args.map_h,
            target_per_bin=args.target_per_bin,
            seeds_per_density=args.seeds_per_density,
            base_seed=args.seed,
        )

    if do_styles:
        print(f"\n{'=' * 60}\nMode B: diverse-style generation\n{'=' * 60}")
        run_styles(
            gen,
            output_dir=output_dir,
            vis_dir=vis_dir,
            num_styles=args.num_styles,
            map_w=args.map_w,
            map_h=args.map_h,
            base_seed=args.seed,
        )

    print(f"\n{'=' * 60}")
    print(f"RoadWeaver evaluation complete — results in {output_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Evaluate RoadWeaver (Phase A) using eval/metrics.py, compare against baselines.

Usage:
    conda activate road-weaver
    python scripts/eval_roadweaver.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import compute_route_coverage, compute_topological_metrics, contract_degree2_nodes
from eval.metrics import merge_close_nodes as metrics_merge

# Need to import the generator
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from network_generator.backbone.config import CONFIG
from network_generator.backbone.dataset import make_field_dataloader
from network_generator.backbone.generator import Generator
from network_generator.growth.config import GrowthConfig
from network_generator.growth.growth import grow
from network_generator.topology.connector import EndpointConnector
from network_generator.topology.graph_cleanup import (
    clean_sharp_angles,
    keep_lcc,
    prune_dead_ends,
    snap_endpoints,
)
from network_generator.topology.graph_ops import (
    detect_roundabouts,
    merge_close_nodes,
    simplify_chains,
)

# ── Config ──
CONFIG.resolution = 128
CONFIG.code_map_size = 32
CONFIG.val_split_path = "data/urban_prior/2km/splits_style/val.parquet"
VQ_CKPT = "runtimes/vq_vae_2km_phase_a/checkpoints/best.pth"
MODEL_CKPT = "runtimes/transformer_2km_phase_a/checkpoints/best.pth"
CACHE = "cache/masked_code_maps_2km_phase_a/train.npz"
DEVICE = "cuda"
OUT = Path("analysis/roadweaver_eval")
OUT.mkdir(parents=True, exist_ok=True)
N_SAMPLES = 50
MAP_SIZE_M = 2000.0

# ── Load ──
print("[Eval] Loading Phase A models...")
gen = Generator(
    vq_checkpoint=VQ_CKPT,
    model_checkpoint=MODEL_CKPT,
    cache_path=CACHE,
    device=DEVICE,
    num_codes=512,
    resolution=128,
    code_map_size=32,
    d_model=256,
    num_layers=6,
    nhead=4,
    use_adaln=True,
    cond_dim=11,
)

# ── Generate evaluation samples ──
results = []
t_start = time.time()

for seed in range(N_SAMPLES):
    # Use random conditions from a fixed set
    with torch.no_grad():
        # Create a diverse set of conditions
        sv = torch.zeros(1, 6, device=DEVICE)
        sv[0, seed % 6] = 1.0
        sp = torch.tensor(
            [[15.0 + (seed % 5) * 8.0, 0.5, 0.5, 0.5, 0.5]], dtype=torch.float, device=DEVICE
        )
        cond = torch.cat([sv, sp], dim=1)

        raw = gen.generate(cond, anchor_ratio=0.08, temperature=0.75, top_p=0.65, seed=seed)
    fld = raw["road_field"]
    c_n, ei = raw["coords"], raw["edge_index"]
    if len(c_n) < 5:
        continue

    # Pipeline
    try:
        conn = EndpointConnector(map_size_m=MAP_SIZE_M).run(
            raw,
            fld,
            max_connections=30,
            connect_remaining=True,
            max_remaining_m=600,
            simplify=False,
        )
        simp = simplify_chains(
            conn["coords"], conn["edge_index"], angle_threshold_deg=15, dp_epsilon_norm=0.002
        )
        c_n, ei = simp[0], simp[1]
    except:
        pass

    # Growth
    try:
        gc = GrowthConfig.from_condition(
            cond[0].cpu().numpy(), local_spacing_m=80.0, map_size_m=MAP_SIZE_M
        )
        gc.map_width_m = MAP_SIZE_M
        gc.map_height_m = MAP_SIZE_M
        grown = grow(c_n * MAP_SIZE_M, ei, np.zeros(len(c_n), dtype=np.int64), fld, gc)
        c_m = grown["coords"] / MAP_SIZE_M
        ei = grown["edge_index"]
    except:
        continue

    # Cleanup: prune → LCC → sharp angles → snap endpoints
    c_m, ei = prune_dead_ends(c_m, ei, 120.0, MAP_SIZE_M)
    c_m, ei = keep_lcc(c_m, ei)
    c_m, ei = clean_sharp_angles(c_m, ei, min_deg=15.0)
    c_m, ei = snap_endpoints(c_m, ei, MAP_SIZE_M, snap_dist_m=50.0)
    c_m, ei = keep_lcc(c_m, ei)  # one more LCC in case snap disconnected something

    if len(c_m) < 5:
        continue

    # ── Convert to nx.Graph for eval ──
    G_raw = nx.Graph()
    for i in range(len(c_m)):
        G_raw.add_node(i, pos=c_m[i].copy())
    for u, v in ei:
        G_raw.add_edge(u, v)

    # Contract degree-2, merge close nodes (eval standard pipeline)
    G_intersection = contract_degree2_nodes(G_raw)
    G_intersection = metrics_merge(G_intersection, distance_threshold=30.0)

    # Compute metrics
    topo = compute_topological_metrics(G_intersection)
    route = compute_route_coverage(G_intersection)

    # Count remaining dead-ends in the raw graph
    degs = dict(G_raw.degree())
    n_endpoints = sum(1 for d in degs.values() if d == 1)
    n_junctions = sum(1 for d in degs.values() if d >= 3)

    results.append(
        {
            **topo,
            **route,
            "n_raw_nodes": len(c_m),
            "n_int_nodes": G_intersection.number_of_nodes(),
            "n_endpoints_raw": n_endpoints,
            "n_junctions_raw": n_junctions,
        }
    )

    if (seed + 1) % 10 == 0:
        print(f"  [{seed+1}/{N_SAMPLES}] {len(results)} valid so far")

# ── Summary ──
elapsed = time.time() - t_start
print(f"\n=== RoadWeaver Eval ({len(results)}/{N_SAMPLES} valid, {elapsed:.0f}s) ===")

metrics_map = {
    "lcc": "LCC ↑",
    "dead_end_ratio": "Dead-end ↓",
    "avg_degree": "Avg Deg",
    "node_count": "Intersection Nodes",
}
for key, label in metrics_map.items():
    vals = [r[key] for r in results]
    print(
        f"  {label:20s} = {np.mean(vals):.4f} ± {np.std(vals):.4f}  [{np.min(vals):.4f}, {np.max(vals):.4f}]"
    )

# Route metrics
route_vals = [r.get("reachable", 0) for r in results if "reachable" in r]
if route_vals:
    print(f"  {'Reachable ↑':20s} = {np.mean(route_vals):.4f}")

# Raw graph stats
raw_nodes = [r["n_raw_nodes"] for r in results]
raw_ends = [r["n_endpoints_raw"] for r in results]
raw_juncs = [r["n_junctions_raw"] for r in results]
print(f"\n  Raw graph: {np.mean(raw_nodes):.0f}±{np.std(raw_nodes):.0f} nodes")
print(
    f"  Endpoints: {np.mean(raw_ends):.0f}±{np.std(raw_ends):.0f} ({np.mean(raw_ends)/max(np.mean(raw_nodes),1)*100:.1f}%)"
)
print(f"  Junctions: {np.mean(raw_juncs):.0f}±{np.std(raw_juncs):.0f}")

import csv

# ── Save as CSV ──
fieldnames = list(results[0].keys()) if results else []
csv_path = OUT / "metrics.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["seed"] + fieldnames)
    w.writeheader()
    for i, r in enumerate(results):
        w.writerow({"seed": i, **r})
print(f"\nSaved → {csv_path}  ({len(results)} rows)")

# ── Comparison with baselines ──
print("\n=== Comparison with baselines (from project_status.md) ===")
print(f"{'Metric':20s} {'MetaDrive':>10s} {'RoadGen':>10s} {'HDMapGen':>10s} {'RoadWeaver':>12s}")
print("-" * 64)
baselines = {
    "LCC": [(0.95, 77), (0.73, 67), ("N/A", 67)],
    "Dead-end": [(0.12, 77), (0.61, 67), ("N/A", 67)],
}
for metric, (md, rg, hd) in baselines.items():
    rw_mean = np.mean([r[{"LCC": "lcc", "Dead-end": "dead_end_ratio"}[metric]] for r in results])
    print(f"{metric:20s} {str(md):>10s} {str(rg):>10s} {str(hd):>10s} {rw_mean:>8.4f}")

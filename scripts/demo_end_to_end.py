#!/usr/bin/env python3
"""
End-to-end demo: generate road fields → graphs → refined skeletons.

Usage:
    conda activate road-weaver
    python scripts/demo_end_to_end.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
# ── Imports from unmigrated modules (src_old) ────────────────────────
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from graph_refiner.endpoint_connector import EndpointConnector
from graph_refiner.road_growth.config import GrowthConfig
from graph_refiner.road_growth.growth import grow

from network_generator.backbone.config import CONFIG
from network_generator.backbone.dataset import make_field_dataloader
from network_generator.backbone.transformer import MaskedCodeModel
from network_generator.backbone.vq_vae import VQVAE
from network_generator.topology.raster_to_graph import field_to_graph
from utils.graph_ops import (
    NT_CURVE,
    NT_ENDPOINT,
    NT_ROUNDABOUT,
    NT_WAYPOINT,
    detect_roundabouts,
    merge_close_nodes,
    simplify_chains,
)

sys.path.insert(0, "src_old")

# ── Config ───────────────────────────────────────────────────────────
VQ_CKPT = "runtimes/vq_vae_5km_new/checkpoints/best.pth"
TRANSFORMER_CKPT = "runtimes/transformer_5km_style_new/checkpoints/best.pth"
RES = 256
CODE_MAP_SIZE = 64
NUM_CODES = 1024
MAP_SIZE = 5000.0
N_SEEDS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "analysis/demo_e2e"
os.makedirs(OUT, exist_ok=True)

# ── Load models ──────────────────────────────────────────────────────
print("[Demo] Loading VQ-VAE + Transformer...")
vq = VQVAE(resolution=RES, num_codes=NUM_CODES, code_map_size=CODE_MAP_SIZE).to(DEVICE)
vq.eval()
state = torch.load(VQ_CKPT, map_location=DEVICE, weights_only=True)
vq.load_state_dict(state["model_state_dict"], strict=False)

seq_len = CODE_MAP_SIZE * CODE_MAP_SIZE
model = MaskedCodeModel(vocab_size=NUM_CODES + 1, max_seq_len=seq_len, d_model=256, nhead=4).to(
    DEVICE
)
model.eval()
state = torch.load(TRANSFORMER_CKPT, map_location=DEVICE, weights_only=True)
model.load_state_dict(state["model_state_dict"], strict=False)
print(f"  Models loaded (code_map={CODE_MAP_SIZE}x{CODE_MAP_SIZE})")

# ── Load validation conditions ───────────────────────────────────────
CONFIG.val_split_path = "data/urban_prior/5km/splits_keras/val.parquet"
ds = make_field_dataloader(
    "val", batch_size=N_SEEDS, shuffle=True, num_workers=2, limit_samples=200, cache_fields=False
)
batch = next(iter(ds))
style = batch["style_vector"]
struct = batch["structural_priors"]
cond = torch.cat([style, struct], dim=1).to(DEVICE)

# ── Generate ─────────────────────────────────────────────────────────
print(f"[Demo] Generating {N_SEEDS} samples...")
all_fields = []
all_graphs = []

for seed in range(N_SEEDS):
    with torch.no_grad():
        code_map = model.sample(cond, num_steps=8, temperature=0.75)
        field_6ch = torch.sigmoid(vq.decode_from_code(code_map))
    road = field_6ch[0, 0].cpu().numpy()

    # Field → graph
    raw_graph = field_to_graph(
        road,
        road_threshold=0.10,
        resolution=RES,
        prune_short_branches=True,
        cleanup=True,
        opening_radius=2,
        closing_radius=2,
        min_edge_len=0.008,
    )

    # Graph refinement
    connector = EndpointConnector(map_size_m=MAP_SIZE, max_pair_dist_m=800.0)
    conn = connector.run(
        raw_graph,
        road,
        max_connections=20,
        connect_remaining=True,
        max_remaining_m=800.0,
        simplify=False,
    )

    conn_types = detect_roundabouts(
        conn["coords"],
        conn["edge_index"],
        conn.get("node_types", np.zeros(len(conn["coords"]), dtype=np.int64)),
        map_size_m=MAP_SIZE,
        max_cycle_size=80,
    )
    ra_keep = {i for i, t in enumerate(conn_types) if t == NT_ROUNDABOUT}

    simp = simplify_chains(
        conn["coords"],
        conn["edge_index"],
        force_keep=ra_keep or None,
        angle_threshold_deg=15.0,
        dp_epsilon_norm=0.002,
        max_seg_len_norm=0.04,
    )
    sc, se, st = simp[:3]
    st = detect_roundabouts(sc, se, st, map_size_m=MAP_SIZE, max_cycle_size=80)

    merged = merge_close_nodes(
        sc, se, st, merge_dist=0.008, map_size_m=MAP_SIZE, dp_epsilon_norm=0.002
    )
    mc, me, mt = merged[:3]

    try:
        gc = GrowthConfig.from_condition(
            cond[seed].cpu().numpy(), local_spacing_m=80.0, map_size_m=MAP_SIZE
        )
        grown = grow(mc * MAP_SIZE, me, mt, road, gc)
        result = {
            "coords": grown["coords"] / MAP_SIZE,
            "edge_index": grown["edge_index"],
            "node_types": grown["node_types"],
        }
        result["node_types"] = detect_roundabouts(
            result["coords"],
            result["edge_index"],
            result["node_types"],
            map_size_m=MAP_SIZE,
            max_cycle_size=80,
        )
        merged2 = merge_close_nodes(
            result["coords"],
            result["edge_index"],
            result["node_types"],
            merge_dist=0.006,
            map_size_m=MAP_SIZE,
            dp_epsilon_norm=0.002,
        )
        refined = {"coords": merged2[0], "edge_index": merged2[1], "node_types": merged2[2]}
    except Exception as e:
        print(f"  [warn] Growth failed: {e}")
        refined = {"coords": mc, "edge_index": me, "node_types": mt}

    all_fields.append(road)
    all_graphs.append(refined)

# ── Plot ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, N_SEEDS, figsize=(4 * N_SEEDS, 8))
for i in range(N_SEEDS):
    axes[0, i].imshow(all_fields[i], cmap="gray_r", vmin=0, vmax=1)
    axes[0, i].set_title(f"Field {i}")
    axes[0, i].axis("off")

    g = all_graphs[i]
    if len(g["coords"]) > 0:
        px = g["coords"] * [RES, RES]
        axes[1, i].imshow(all_fields[i], cmap="gray_r", vmin=0, vmax=1, alpha=0.3)
        for u, v in g["edge_index"]:
            axes[1, i].plot(
                [px[u, 0], px[v, 0]], [px[u, 1], px[v, 1]], color="#e74c3c", lw=0.8, alpha=0.7
            )
        colors = [
            (
                "#4CAF50"
                if t == 0
                else (
                    "#2196F3"
                    if t == 1
                    else (
                        "#FF9800"
                        if t == 2
                        else "#F44336" if t == 3 else "#9C27B0" if t == 4 else "#888"
                    )
                )
            )
            for t in g["node_types"]
        ]
        axes[1, i].scatter(px[:, 0], px[:, 1], c=colors, s=12, edgecolors="black", lw=0.3, zorder=3)
    axes[1, i].set_title(f"Graph {i}")
    axes[1, i].axis("off")

plt.tight_layout()
plt.savefig(f"{OUT}/demo_e2e.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[Demo] Saved to {OUT}/demo_e2e.png")

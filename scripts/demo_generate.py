#!/usr/bin/env python3
"""
Generate road fields using the full Generator pipeline (with anchor tokens).

Outputs are saved to ``analysis/generated/{vq_name}+{tfm_name}/``.

Usage:
    conda activate road-weaver
    python scripts/demo_generate.py                               # 2km model (default)
    python scripts/demo_generate.py --help                         # all options
    python scripts/demo_generate.py --run-full                     # include graph + growth
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network_generator.backbone.generator import Generator
from network_generator.growth.config import GrowthConfig
from network_generator.growth.growth import grow
from network_generator.topology.connector import EndpointConnector
from network_generator.topology.graph_ops import (
    NT_ROUNDABOUT,
    detect_roundabouts,
    merge_close_nodes,
    simplify_chains,
)
from network_generator.topology.raster_to_graph import field_to_graph

DEFAULT_VQ = "runtimes/vq_vae_2km/checkpoints/best.pth"
DEFAULT_TFM = "runtimes/transformer_2km_style/checkpoints/best.pth"
DEFAULT_CACHE = "cache/masked_code_maps_2km_style/train.npz"

STYLES = {
    "Gridiron": [0.8, 0.05, 0.05, 0.05, 0.05, 0.0],
    "Linear": [0.05, 0.85, 0.0, 0.05, 0.05, 0.0],
    "No pattern": [1 / 6] * 6,
    "Organic": [0.05, 0.05, 0.0, 0.8, 0.05, 0.05],
    "Radial": [0.05, 0.05, 0.0, 0.05, 0.8, 0.05],
    "Tributary": [0.0, 0.0, 0.0, 0.05, 0.05, 0.9],
}
DENSITIES = [5, 10, 15, 20, 25, 30, 40]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vq", default=DEFAULT_VQ)
    p.add_argument("--transformer", default=DEFAULT_TFM)
    p.add_argument("--cache", default=DEFAULT_CACHE)
    p.add_argument("--res", type=int, default=128)
    p.add_argument("--codes", type=int, default=512)
    p.add_argument("--code-map", type=int, default=32)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--run-full", action="store_true", help="run full pipeline (field → graph → growth)"
    )
    return p.parse_args()


def main():
    args = parse_args()
    vq_name = Path(args.vq).parent.parent.name
    tfm_name = Path(args.transformer).parent.parent.name
    out_dir = f"analysis/generated/{vq_name}+{tfm_name}"
    os.makedirs(out_dir, exist_ok=True)

    gen = Generator(
        vq_checkpoint=args.vq,
        model_checkpoint=args.transformer,
        cache_path=args.cache,
        device=args.device,
        d_model=args.d_model,
        num_layers=args.num_layers,
        nhead=args.nhead,
        num_codes=args.codes,
        resolution=args.res,
        code_map_size=args.code_map,
    )
    print(f"Output → {out_dir}/")

    # ── Style sweep ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    with torch.no_grad():
        for (name, sv), ax in zip(STYLES.items(), axes.flat):
            cond = torch.zeros(1, 11, device=args.device)
            cond[0, :6] = torch.tensor(sv, device=args.device)
            cond[0, 6] = 1.0
            code_map = gen.generate_code_map(cond, temperature=0.75, top_p=0.65)
            field = torch.sigmoid(gen.vq.decode_from_code(code_map))
            ax.imshow(field[0, 5].cpu().numpy(), cmap="gray_r", vmin=0, vmax=1)
            ax.set_title(f"{name}")
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/style_sweep.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  style_sweep.png")

    # ── Density sweep ───────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    with torch.no_grad():
        for d, ax in zip(DENSITIES, axes.flat):
            cond = torch.zeros(1, 11, device=args.device)
            cond[0, 6] = d / 20.0
            code_map = gen.generate_code_map(cond, temperature=0.75, top_p=0.65)
            field = torch.sigmoid(gen.vq.decode_from_code(code_map))
            ax.imshow(field[0, 5].cpu().numpy(), cmap="gray_r", vmin=0, vmax=1)
            ax.set_title(f"density={d}")
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/density_sweep.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  density_sweep.png")

    # ── Diversity ───────────────────────────────────────────────────────
    cond = torch.zeros(1, 11, device=args.device)
    cond[0, :6] = torch.tensor(STYLES["Gridiron"], device=args.device)
    cond[0, 6] = 1.0
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    with torch.no_grad():
        for ax in axes.flat:
            code_map = gen.generate_code_map(cond, temperature=0.75, top_p=0.65)
            field = torch.sigmoid(gen.vq.decode_from_code(code_map))
            ax.imshow(field[0, 5].cpu().numpy(), cmap="gray_r", vmin=0, vmax=1)
            ax.set_title(f"{len(code_map[0].unique())} codes")
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/diversity.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  diversity.png")

    # ── Full pipeline (optional) ────────────────────────────────────────
    if args.run_full:
        MS = 2000.0 if args.res == 128 else 5000.0
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for i, d in enumerate([5, 10, 20, 30]):
            cond = torch.zeros(1, 11, device=args.device)
            cond[0, 6] = d / 20.0
            with torch.no_grad():
                code_map = gen.generate_code_map(cond, temperature=0.75, top_p=0.65)
                field = torch.sigmoid(gen.vq.decode_from_code(code_map))
            road = field[0, 5].cpu().numpy()

            axes[0, i].imshow(road, cmap="gray_r")
            axes[0, i].set_title(f"Field d={d}")
            axes[0, i].axis("off")

            g_out = {
                "coords": np.zeros((0, 2)),
                "edge_index": np.zeros((0, 2), dtype=np.int64),
                "node_types": np.zeros(0, dtype=np.int64),
            }
            raw = field_to_graph(
                road, road_threshold=0.10, resolution=args.res, cleanup=True, min_edge_len=0.008
            )
            if len(raw["coords"]) >= 5:
                conn = EndpointConnector(map_size_m=MS).run(
                    raw, road, max_connections=20, connect_remaining=True
                )
                if len(conn["coords"]) >= 5:
                    ct = detect_roundabouts(
                        conn["coords"],
                        conn["edge_index"],
                        conn.get("node_types", np.zeros(len(conn["coords"]), dtype=np.int64)),
                        map_size_m=MS,
                    )
                    simp = simplify_chains(
                        conn["coords"],
                        conn["edge_index"],
                        force_keep={j for j, t in enumerate(ct) if t == NT_ROUNDABOUT} or None,
                    )
                    sc, se, st = simp[:3]
                    st = detect_roundabouts(sc, se, st, map_size_m=MS)
                    merged = merge_close_nodes(sc, se, st, merge_dist=0.008, map_size_m=MS)
                    mc, me, mt = merged[:3]
                    if len(mc) >= 3:
                        try:
                            gc = GrowthConfig.from_condition(
                                cond[0].cpu().numpy(), local_spacing_m=80.0, map_size_m=MS
                            )
                            mc_m = mc * MS
                            mc_m[:, 1] = MS - mc_m[:, 1]
                            grown = grow(mc_m, me, mt, road, gc)
                            grown["coords"][:, 1] = MS - grown["coords"][:, 1]
                            g_out = {
                                "coords": grown["coords"] / MS,
                                "edge_index": grown["edge_index"],
                                "node_types": grown["node_types"],
                            }
                        except Exception:
                            g_out = {"coords": mc, "edge_index": me, "node_types": mt}

            g = g_out
            if len(g["coords"]) > 0:
                px = g["coords"] * args.res
                axes[1, i].imshow(road, cmap="gray_r", alpha=0.3)
                for u, v in g["edge_index"]:
                    axes[1, i].plot(
                        [px[u, 0], px[v, 0]], [px[u, 1], px[v, 1]], "r-", lw=0.8, alpha=0.7
                    )
                colors = [
                    (
                        "#4CAF50"
                        if t == 0
                        else (
                            "#2196F3"
                            if t == 1
                            else "#FF9800" if t == 2 else "#F44336" if t == 3 else "#9C27B0"
                        )
                    )
                    for t in g["node_types"]
                ]
                axes[1, i].scatter(
                    px[:, 0], px[:, 1], c=colors, s=10, edgecolors="black", lw=0.3, zorder=3
                )
            axes[1, i].set_title(f"Graph (n={len(g['coords'])})")
            axes[1, i].axis("off")

        plt.tight_layout()
        plt.savefig(f"{out_dir}/full_pipeline.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  full_pipeline.png")

    print(f"\nAll outputs → {out_dir}/")


if __name__ == "__main__":
    main()

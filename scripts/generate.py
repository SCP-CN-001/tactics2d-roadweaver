#!/usr/bin/env python3
"""
CLI entry point for RoadWeaver graph generation.

Usage::
    conda activate road-weaver
    python scripts/generate.py                          # 6 samples
    python scripts/generate.py --n-samples 24           # more samples
    python scripts/generate.py --map-w 3000 --map-h 2000  # non-square
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Args ──
import argparse

from assemble_hdmap import graph_to_map, quick_vis
from generate_graph import generate_branch, generate_skeleton
from network_generator.backbone.config import CONFIG
from network_generator.backbone.dataset import make_field_dataloader
from network_generator.backbone.generator import Generator
from network_generator.topology.graph_ops import NT_ENDPOINT, NT_JUNCTION, NT_ROUNDABOUT

parser = argparse.ArgumentParser()
parser.add_argument("--n-samples", type=int, default=6)
parser.add_argument("--map-w", type=float, default=2000.0)
parser.add_argument("--map-h", type=float, default=2000.0)
parser.add_argument("--vq-ckpt", default="runtimes/vq_vae_2km_phase_a/checkpoints/best.pth")
parser.add_argument("--model-ckpt", default="runtimes/transformer_2km_phase_a/checkpoints/best.pth")
parser.add_argument("--cache", default="cache/masked_code_maps_2km_phase_a/train.npz")
parser.add_argument("--output", default="analysis/e2e_phase_a")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

OUT = Path(args.output)
OUT.mkdir(parents=True, exist_ok=True)
N = args.n_samples
MAP_W, MAP_H = args.map_w, args.map_h
VQ_MAP = 2000.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load ──
print(f"[Generate] Loading models...")
CONFIG.resolution = 128
CONFIG.code_map_size = 32
CONFIG.val_split_path = "data/urban_prior/2km/splits_style/val.parquet"

gen = Generator(
    vq_checkpoint=args.vq_ckpt,
    model_checkpoint=args.model_ckpt,
    cache_path=args.cache,
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

dl = make_field_dataloader("val", batch_size=N, num_workers=0, limit_samples=N, cache_fields=False)
batch = next(iter(dl))
style, struct = batch["style_vector"], batch["structural_priors"]
cond = torch.cat([style, struct], dim=1).to(DEVICE)
pnames = ["Gridiron", "Linear", "No Pat", "Organic", "Radial", "Trib"]

# ── Generate ──
all_results = []
for i in range(N):
    pid = int(style[i].argmax())
    d_val = float(struct[i, 0])
    print(f"\n[{i}] {pnames[pid]} density={d_val:.1f} {MAP_W:.0f}x{MAP_H:.0f}m")

    try:
        skel = generate_skeleton(
            gen,
            cond[i],
            struct[i],
            map_w=MAP_W,
            map_h=MAP_H,
            vq_map_size_m=VQ_MAP,
            seed=args.seed + i,
        )
        branch = generate_branch(
            skel["coords"],
            skel["edge_index"],
            skel["road_field"],
            skel["condition"],
            map_w=MAP_W,
            map_h=MAP_H,
            density=skel["density"],
            gridness=skel["gridness"],
            organic=skel["organic"],
        )
    except Exception as e:
        print(f"  FAILED: {e}")
        all_results.append(None)
        continue

    c_int, ei_int = branch["coords_int"], branch["edge_index_int"]
    nt = branch["node_types"]
    nj = int((nt == 1).sum())
    nr = int((nt == 3).sum())
    ne = int((nt == 4).sum())
    print(f"  {len(c_int)}n {len(ei_int)}e J={nj} R={nr} E={ne}")

    # ── Assemble HD Map ──
    try:
        hd_map = graph_to_map(
            branch["coords_int"],
            branch["edge_index_int"],
            branch["geometries"],
            node_types=branch["node_types"],
            lanes_per_dir=branch["lanes_per_dir"],
            road_class=branch["road_class"],
            map_w=MAP_W,
            map_h=MAP_H,
            name=f"rw_{i}",
            scenario_type="urban",
        )
        vis_path = OUT / f"hdmap_{i}.png"
        quick_vis(hd_map, str(vis_path), dpi=300)
    except Exception as e:
        print(f"  HDMap FAILED: {e}")
        hd_map = None

    all_results.append({"skel": skel, "branch": branch, "hdmap": hd_map})

# ── Plot ──
valid = [r for r in all_results if r is not None]
if not valid:
    print("No valid results")
    sys.exit(0)

fig, axes = plt.subplots(2, N, figsize=(4 * N, 7))
scaled = abs(MAP_W / VQ_MAP - 1) > 0.01 or abs(MAP_H / VQ_MAP - 1) > 0.01

for i in range(N):
    r = all_results[i]
    for row in (0, 1):
        ax = axes[row, i] if N > 1 else axes[row]
        if r is None:
            ax.text(
                0.5, 0.5, "FAILED", ha="center", va="center", transform=ax.transAxes, fontsize=10
            )
            ax.axis("off")
            continue

        if row == 0:
            # Raw skeleton for reference
            c = r["skel"]["coords"]
            e = r["skel"]["edge_index"]
            fld = r["skel"]["road_field"]
            px = c * 128 if len(c) else []
            if not scaled:
                ax.imshow(fld, cmap="gray_r", vmin=0, vmax=1, origin="lower")
            else:
                ax.set_facecolor("#f5f5f5")
            if len(px):
                for u, v in e:
                    ax.plot([px[u, 0], px[v, 0]], [px[u, 1], px[v, 1]], "k-", lw=0.3, alpha=0.4)
                ax.scatter(px[:, 0], px[:, 1], c="#F9A825", s=2, zorder=3, alpha=0.5)
            ax.set_title(f"Skeleton: {len(c)}n {len(e)}e", fontsize=9)
        else:
            # Intersection graph
            b = r["branch"]
            c_int, e_int = b["coords_int"], b["edge_index_int"]
            nt = b["node_types"]
            lanes = b["lanes_per_dir"]
            geoms = b.get("geometries", [])
            px = c_int * 128 if len(c_int) else []
            fld = r["skel"]["road_field"]
            if not scaled:
                ax.imshow(fld, cmap="gray_r", vmin=0, vmax=1, alpha=0.25, origin="lower")
            else:
                ax.set_facecolor("#f5f5f5")
            if len(px):
                for j, (u, v) in enumerate(e_int):
                    l2 = min(2.5, 0.3 + 0.3 * int(lanes[j]) if j < len(lanes) else 0.5)
                    geom = geoms[j] if j < len(geoms) else None
                    if geom is not None and len(geom) > 2:
                        gpx = geom * 128
                        ax.plot(gpx[:, 0], gpx[:, 1], "k-", lw=l2, alpha=0.5)
                    else:
                        ax.plot([px[u, 0], px[v, 0]], [px[u, 1], px[v, 1]], "k-", lw=l2, alpha=0.5)
                jm = nt == NT_JUNCTION
                rm = nt == NT_ROUNDABOUT
                em = nt == NT_ENDPOINT
                if jm.any():
                    ax.scatter(
                        px[jm, 0],
                        px[jm, 1],
                        c="#1565C0",
                        s=30,
                        edgecolors="black",
                        lw=0.4,
                        zorder=3,
                    )
                if rm.any():
                    ax.scatter(
                        px[rm, 0],
                        px[rm, 1],
                        c="#E53935",
                        s=30,
                        edgecolors="black",
                        lw=0.4,
                        zorder=3,
                    )
                if em.any():
                    ax.scatter(
                        px[em, 0],
                        px[em, 1],
                        c="#FDD835",
                        s=20,
                        edgecolors="black",
                        lw=0.3,
                        zorder=3,
                    )
            ax.set_title(f"Int: {len(c_int)}n {len(e_int)}e", fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")

plt.tight_layout()
plt.savefig(OUT / "e2e_grid.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"\nSaved {OUT / 'e2e_grid.png'}")

for i, r in enumerate(all_results):
    if r is None:
        print(f"  [{i}] FAILED")
        continue
    nt = r["branch"]["node_types"]
    print(
        f"  [{i}] {len(r['branch']['coords_int']):4d}n {len(r['branch']['edge_index_int']):4d}e "
        f"J={int((nt==1).sum())} R={int((nt==3).sum())} E={int((nt==4).sum())}"
    )

# ── HD Map comparison grid (6+1) ──
hd_valid = [(i, r) for i, r in enumerate(all_results) if r is not None and r["hdmap"] is not None]
if len(hd_valid) > 0:
    n_hd = len(hd_valid)
    rows = (n_hd + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
    axes = axes.flatten() if rows > 1 else [axes]
    for idx, (i, r) in enumerate(hd_valid):
        ax = axes[idx]
        hd = r["hdmap"]
        lanes = list(hd.lanes.values())
        road_lanes = [l for l in lanes if l.id_.startswith("e")]
        ax.set_facecolor("#FAFAFA")
        for lane in road_lanes:
            for side in (lane.left_side, lane.right_side):
                try:
                    if side is not None and not side.is_empty:
                        x, y = side.xy
                        ax.plot(x, y, color="#222", lw=0.4, alpha=0.7)
                except Exception:
                    pass
        for junc in (hd.junctions or {}).values():
            shape = getattr(junc, "custom_tags", {}).get("shape", [])
            if shape and len(shape) > 2:
                xs, ys = zip(*shape)
                ax.fill(xs, ys, alpha=0.06, color="#888")
        for area in (hd.areas or {}).values():
            try:
                if area.geometry and not area.geometry.is_empty:
                    x, y = area.geometry.exterior.xy
                    ax.fill(x, y, alpha=0.08, color="#aaa")
            except Exception:
                pass
        dead = sum(1 for l in road_lanes if not l.successors)
        nj = len(hd.junctions or {})
        na = len(hd.areas or {})
        ax.set_title(f"[{i}] {dead} dead lanes, {nj} J, {na} A", fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")
    # Hide unused axes
    for idx in range(n_hd, len(axes)):
        axes[idx].axis("off")
    plt.tight_layout()
    fig.savefig(OUT / "hdmap_grid.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {OUT / 'hdmap_grid.png'} ({n_hd} maps)")

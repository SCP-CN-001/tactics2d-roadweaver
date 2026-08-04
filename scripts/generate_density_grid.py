#!/usr/bin/env python3
"""Generate RoadWeaver 2km maps — density grid or a single 40-node map.

Two modes (argparse subcommands):
    grid    — batch generate maps across road densities, with actual rho stats
    organic — deterministically reproduce a single ~40-node map (+ params JSON)

Output:
    analysis/density_grid/     (grid)
    analysis/organic_40node/   (organic)

Usage:
    conda activate road-weaver
    python scripts/generate_density_grid.py grid --densities 10,15,20,25 --seeds 10
    python scripts/generate_density_grid.py organic --density 9 --seed 3
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch

from network_generator import run_pipeline
from network_generator.backbone.config import CONFIG
from network_generator.backbone.generator import make_generator
from network_generator.topology.graph_utils import NT_ENDPOINT, NT_JUNCTION, NT_ROUNDABOUT
from utils.render import render_map

REPO = Path(__file__).resolve().parent.parent
VQ_CKPT = str(REPO / "runtimes/vq_vae_2km/best.pth")
TFM_CKPT = str(REPO / "runtimes/transformer_2km/best.pth")
CACHE = str(REPO / "cache/masked_code_maps_2km/train.npz")
VQ_MAP_M = 2000.0
MAP_W = MAP_H = 2000.0

# 6-dim style vector (soft one-hot over the 6 patterns).  Style control is not
# effective in the current model, so this is just a fixed condition slot.
STYLE = [0.05, 0.05, 0.0, 0.8, 0.05, 0.05]


def measure_lane_density(hd_map, map_w: float, map_h: float) -> float:
    """Actual road density = total lane-centreline length / map area (km/km²)."""
    area_km2 = (map_w * map_h) / 1e6
    total = 0.0
    for lane in hd_map.lanes.values():
        lt, rt = lane.left_side, lane.right_side
        if lt is None or rt is None or lt.is_empty or rt.is_empty:
            continue
        lta = np.array(list(lt.coords))
        rta = np.array(list(rt.coords))
        n = min(len(lta), len(rta))
        if n < 2:
            continue
        mid = (lta[:n] + rta[:n]) / 2
        total += float(np.sqrt(((mid[1:] - mid[:-1]) ** 2).sum(axis=1)).sum())
    return total / 1000.0 / area_km2


def build_condition(
    density: float,
    gridness: float = 0.25,
    radialness: float = 0.10,
    organic: float = 0.75,
    bearing_entropy: float = 0.70,
) -> torch.Tensor:
    """11-dim condition = [style_vector(6) | structural_priors(5)].

    ``density`` is the raw road density in km/km² (index 6), the same
    convention the model was trained on.  Do *not* scale it by 1/20 here.
    """
    cond = torch.zeros(11)
    cond[:6] = torch.tensor(STYLE)
    cond[6] = density
    cond[7] = gridness
    cond[8] = radialness
    cond[9] = organic
    cond[10] = bearing_entropy
    return cond


def _make_gen(device: str):
    """Load the VQ + Transformer generator (shared by both modes)."""
    CONFIG.resolution = 128
    CONFIG.code_map_size = 32
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("[Gen] CUDA unavailable -> falling back to CPU")
    return make_generator(VQ_CKPT, TFM_CKPT, cache_path=CACHE, device=device)


# ═══════════════════════════════════════════════════════════════════════
#  grid mode — density sweep with actual rho stats
# ═══════════════════════════════════════════════════════════════════════


def run_grid_mode(args):
    densities = [float(d) for d in args.densities.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    out = Path(args.output)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    CONFIG.val_split_path = "data/urban_prior/2km/splits_style/val.parquet"
    print(f"[Gen] Loading models from runtimes/ ...")
    gen = _make_gen(device)

    rows: list[tuple[float, int, int, int, int, int, int, int, float]] = []
    for d in densities:
        for s in range(args.seeds):
            torch.manual_seed(s)
            torch.cuda.manual_seed_all(s)
            cond = build_condition(d).to(device)
            t0 = time.time()
            try:
                result = run_pipeline(
                    gen,
                    cond,
                    cond[6:],
                    map_w=MAP_W,
                    map_h=MAP_H,
                    vq_map_size_m=VQ_MAP_M,
                    seed=s,
                    name=f"rw_d{int(d)}_{s}",
                    scenario_type="urban",
                    assemble_hdmap=True,
                )
            except Exception as e:
                print(f"  [FAIL] density={d:.0f} seed={s}: {e}")
                continue
            branch, hd_map = result["branch"], result["hdmap"]
            nt = branch["node_types"]
            n_nodes = len(branch["coords_int"])
            n_edges = len(branch["edge_index_int"])
            nj = int((nt == NT_JUNCTION).sum())
            nr = int((nt == NT_ROUNDABOUT).sum())
            ne = int((nt == NT_ENDPOINT).sum())
            n_lanes = len(hd_map.lanes)
            rho = measure_lane_density(hd_map, MAP_W, MAP_H)
            dt = time.time() - t0
            png = out / f"d{int(d):02d}_seed{s}_rho{rho:.1f}.png"
            try:
                render_map(hd_map, str(png), resolution=1800, dpi=150)
            except Exception as e:
                print(f"  [FAIL] render density={d:.0f} seed={s}: {e}")
                continue
            print(
                f"  d={d:5.0f} s={s}  {n_nodes:4d}n {n_edges:4d}e "
                f"J={nj:3d} R={nr:3d} E={ne:3d} lanes={n_lanes:4d}  "
                f"rho={rho:5.1f}  {dt:5.1f}s -> {png.name}"
            )
            rows.append((d, s, n_nodes, n_edges, nj, nr, ne, n_lanes, rho))

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'=' * 74}\nSummary (mean over seeds per density)\n{'=' * 74}")
    for d in densities:
        rs = [r for r in rows if r[0] == d]
        if not rs:
            print(f"  density {d:5.0f}: 0/{args.seeds} maps generated")
            continue
        avg_n = np.mean([r[2] for r in rs])
        avg_e = np.mean([r[3] for r in rs])
        avg_l = np.mean([r[7] for r in rs])
        avg_rho = np.mean([r[8] for r in rs])
        rho_min = min(r[8] for r in rs)
        rho_max = max(r[8] for r in rs)
        print(
            f"  density {d:5.0f}: {len(rs)}/{args.seeds} maps, "
            f"avg {avg_n:6.1f}n {avg_e:6.1f}e {avg_l:6.1f} lanes, "
            f"actual rho {avg_rho:5.1f} km/km² (min {rho_min:4.1f} max {rho_max:4.1f})"
        )
    print(f"\nAll outputs -> {out}/")


# ═══════════════════════════════════════════════════════════════════════
#  organic mode — deterministic single ~40-node map + params JSON
# ═══════════════════════════════════════════════════════════════════════


def run_organic_mode(args):
    os.chdir(REPO)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gen = _make_gen(device)
    cond = build_condition(args.density).to(device)

    # Deterministic: seed global torch RNG before generation.  The generator's
    # ``torch.multinomial`` sampling uses the global torch RNG, so the ``seed``
    # arg alone only fixes anchor placement, not token sampling.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    result = run_pipeline(
        gen,
        cond,
        cond[6:],
        map_w=MAP_W,
        map_h=MAP_H,
        vq_map_size_m=VQ_MAP_M,
        seed=args.seed,
        name=f"organic_{args.density}_{args.seed}",
        scenario_type="urban",
        assemble_hdmap=True,
    )
    br = result["branch"]
    nt = br["node_types"]
    n, e = len(br["coords_int"]), len(br["edge_index_int"])
    nj = int((nt == 1).sum())
    nr = int((nt == 3).sum())
    ne = int((nt == 4).sum())
    print(f"density={args.density} K={args.seed}  ->  {n}N {e}E J={nj} R={nr} E={ne}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    render_map(result["hdmap"], str(out / f"hdmap_d{args.density}_s{args.seed}.png"), dpi=200)

    (out / f"params_d{args.density}_s{args.seed}.json").write_text(
        json.dumps(
            {
                "density": args.density,
                "seed": args.seed,
                "map_w": MAP_W,
                "map_h": MAP_H,
                "vq_map_size_m": VQ_MAP_M,
                "style_vector": STYLE,
                "gridness": 0.25,
                "radialness": 0.10,
                "organic": 0.75,
                "bearing_entropy": 0.70,
                "n_nodes": n,
                "n_edges": e,
                "junctions": nj,
                "roundabouts": nr,
                "endpoints": ne,
                "vq_ckpt": VQ_CKPT,
                "transformer_ckpt": TFM_CKPT,
                "cache": CACHE,
            },
            indent=2,
        )
    )
    print(f"Saved {out / f'hdmap_d{args.density}_s{args.seed}.png'}")


# ═══════════════════════════════════════════════════════════════════════
#  entry point
# ═══════════════════════════════════════════════════════════════════════


def main():
    ap = argparse.ArgumentParser(
        description="Generate RoadWeaver 2km maps (density grid or single)."
    )
    sub = ap.add_subparsers(dest="mode", required=True)

    g = sub.add_parser("grid", help="batch generate across road densities (with actual rho stats)")
    g.add_argument(
        "--densities", default="10,15,20,25", help="comma-separated km/km² (default 10,15,20,25)"
    )
    g.add_argument("--seeds", type=int, default=10, help="maps per density (default 10)")
    g.add_argument(
        "--output", default="analysis/density_grid", help="output dir (cleared on start)"
    )
    g.add_argument("--device", default="cuda")
    g.set_defaults(func=run_grid_mode)

    o = sub.add_parser("organic", help="deterministically reproduce a single ~40-node map")
    o.add_argument("--density", type=float, default=9.0)
    o.add_argument("--seed", type=int, default=3, help="torch.manual_seed(K), also anchor seed")
    o.add_argument("--out", default=str(REPO / "analysis" / "organic_40node"))
    o.set_defaults(func=run_organic_mode)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Visualize RoadWeaver generation results (pipeline / sweep / VQ recon).

One script, three modes (argparse subcommands):
    pipeline — per-map 8-panel pipeline visualization (``run_pipeline``
               ``return_intermediates`` states + tactics2d HD Map)
    sweep    — style / density / diversity road-field sweeps
    vq       — VQ-VAE reconstruction panels (per-sample folder + grid)

Output:
    analysis/pipeline/   pipeline mode   (map_{idx}_{w}x{h}/ + combined.png)
    analysis/sweep/      sweep mode      (style/ density/ diversity/ + combined.png)
    analysis/vq_recon/   vq mode         (recon_{i}/ + grid.png)

Usage:
    conda activate road-weaver
    python scripts/visualize_generation.py pipeline [--n 12] [--device cuda]
    python scripts/visualize_generation.py sweep [--run-full]
    python scripts/visualize_generation.py vq
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from network_generator.backbone.config import CONFIG
from network_generator.backbone.dataset import make_field_dataloader
from network_generator.backbone.generator import make_generator
from network_generator.backbone.vq_vae import VQVAE
from network_generator.growth.config import GrowthConfig
from network_generator.growth.growth import grow
from network_generator.pipeline import run_pipeline
from network_generator.topology.graph_connector import EndpointConnector
from network_generator.topology.graph_intersection import detect_roundabouts
from network_generator.topology.graph_merge import merge_close_nodes
from network_generator.topology.graph_simplify import simplify_chains
from network_generator.topology.graph_utils import NT_ROUNDABOUT
from network_generator.topology.raster_to_graph import field_to_graph
from utils.render import render_map
from utils.visualization import _draw_field, _draw_graph, draw_intersection_graph, plot_recon_grid

# ── Shared checkpoints ─────────────────────────────────────────────────
VQ_CKPT = "runtimes/vq_vae_2km/best.pth"
TFM_CKPT = "runtimes/transformer_2km/best.pth"
CACHE = "cache/masked_code_maps_2km/train.npz"

# ── pipeline mode ──────────────────────────────────────────────────────
VQ_MAP_SIZE_M = 2000.0
MAP_SIZES = [(2000, 2000), (3000, 2000), (2000, 3000), (4000, 2000), (2000, 4000), (5000, 2000)]
SKEL = "#F9A825"

# ── sweep mode ─────────────────────────────────────────────────────────
STYLES = {
    "Gridiron": [0.8, 0.05, 0.05, 0.05, 0.05, 0.0],
    "Linear": [0.05, 0.85, 0.0, 0.05, 0.05, 0.0],
    "No pattern": [1 / 6] * 6,
    "Organic": [0.05, 0.05, 0.0, 0.8, 0.05, 0.05],
    "Radial": [0.05, 0.05, 0.0, 0.05, 0.8, 0.05],
    "Tributary": [0.0, 0.0, 0.0, 0.05, 0.05, 0.9],
}
DENSITIES = [5, 10, 15, 20, 25, 30, 40]

# ── vq mode ────────────────────────────────────────────────────────────
VQ_OUTPUT = "analysis/vq_recon"
N_SAMPLES = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════════════════════════════════
#  pipeline mode
# ═══════════════════════════════════════════════════════════════════════


def _save_panel(folder: Path, fname: str, title: str, c, e, bg, nc, lw):
    """Save one simple graph/field panel (panels 1-5)."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(title, fontsize=12)
    if bg is not None:
        _draw_field(ax, bg, extent=(0, 1, 0, 1), origin="lower")
    else:
        ax.set_facecolor("#f5f5f5")
    if len(c):
        _draw_graph(
            ax,
            c,
            e,
            np.zeros(len(c), dtype=np.int64),
            edge_color="k",
            edge_lw=lw,
            edge_alpha=0.5,
            node_color=nc,
            node_size=lambda n: max(2, 12 - n // 50),
            fixed_limits=None,
            hide_axes=False,
        )
        xs, ys = c[:, 0], c[:, 1]
        mx = max(0.02, (xs.max() - xs.min()) * 0.05)
        my = max(0.02, (ys.max() - ys.min()) * 0.05)
        ax.set_xlim(xs.min() - mx, xs.max() + mx)
        ax.set_ylim(ys.min() - my, ys.max() + my)
    else:
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(str(folder / fname), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _combined(folder: Path, names: list[str]):
    """Stitch the panel PNGs in *names* into one ``combined.png``."""
    n = len(names)
    n_cols = 4
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    for ax, fname in zip(np.atleast_1d(axes).ravel(), names):
        img = plt.imread(str(folder / fname))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(fname, fontsize=8)
    for ax in np.atleast_1d(axes).ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(folder / "combined.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_pipeline_mode(args):
    """8-panel per-map pipeline visualization from run_pipeline intermediates."""
    CONFIG.resolution = 128
    CONFIG.code_map_size = 32
    CONFIG.val_split_path = "data/urban_prior/2km/splits_style/val.parquet"

    print("[Viz] Loading...")
    gen = make_generator(VQ_CKPT, TFM_CKPT, cache_path=CACHE, device=args.device)
    dl = make_field_dataloader(
        "val", batch_size=6, num_workers=0, limit_samples=6, cache_fields=False
    )
    batch = next(iter(dl))
    style, struct = batch["style_vector"], batch["structural_priors"]
    pnames = ["Gridiron", "Linear", "No Pat", "Organic", "Radial", "Trib"]
    cond_all = torch.cat([style, struct], dim=1).to(args.device)

    OUT = Path("analysis/pipeline")
    idx = 0
    for mw, mh in MAP_SIZES:
        if idx >= args.n:
            break
        for i in range(2):
            if idx >= args.n:
                break
            sid = i % 6
            pid = int(style[sid].argmax())
            d_val = float(struct[sid, 0])
            print(f"[{idx}] {pnames[pid]} {mw}x{mh} d={d_val:.1f}")

            try:
                result = run_pipeline(
                    gen,
                    cond_all[sid],
                    struct[sid],
                    map_w=mw,
                    map_h=mh,
                    vq_map_size_m=VQ_MAP_SIZE_M,
                    name=f"rw_{idx}",
                    scenario_type="urban",
                    seed=sid,
                    return_intermediates=True,
                )
            except Exception as exc:
                print(f"  skip ({exc})")
                idx += 1
                continue

            skel, branch, hd = result["skeleton"], result["branch"], result["hdmap"]
            fld = skel["road_field"]
            c1, e1 = skel["intermediates"]["raw"]
            c2, e2 = skel["intermediates"]["simplified"]
            c4, e4 = skel["coords"], skel["edge_index"]
            c5, e5 = branch["intermediates"]["growth"]
            c6, e6 = branch["intermediates"]["cleanup"]
            c7, e7 = branch["coords_int"], branch["edge_index_int"]
            nt, lanes = branch["node_types"], branch["lanes_per_dir"]
            geoms, rc = branch["geometries"], branch["road_class"]
            nj = int((nt == 1).sum())
            nr = int((nt == 3).sum())
            ne = int((nt == 4).sum())
            print(f"  {len(c7)}n {len(e7)}e J={nj} R={nr} E={ne}")

            folder = OUT / f"map_{idx}_{mw}x{mh}"
            folder.mkdir(parents=True, exist_ok=True)

            # Panel 0: Field only
            fig0, ax0 = plt.subplots(figsize=(8, 8))
            ax0.set_title("0 Field Only", fontsize=12)
            _draw_field(ax0, fld, extent=(0, 1, 0, 1), origin="lower")
            ax0.set_xlim(-0.02, 1.02)
            ax0.set_ylim(-0.02, 1.02)
            ax0.set_aspect("equal")
            ax0.axis("off")
            fig0.savefig(str(folder / "0_Field_Only.png"), dpi=200, bbox_inches="tight")
            plt.close(fig0)

            # Panels 1-5: simple graph / field renders
            _save_panel(folder, "1_VQ_Field.png", "1 VQ Field", c1, e1, fld, SKEL, 0.7)
            _save_panel(folder, "2_Skeleton.png", "2 Skeleton", c2, e2, fld, SKEL, 0.7)
            _save_panel(folder, "3_Scaled+clean.png", "3 Scaled+clean", c4, e4, None, SKEL, 0.7)
            _save_panel(folder, "4_Growth.png", "4 Growth", c5, e5, None, "#888", 0.5)
            _save_panel(folder, "5_Cleanup.png", "5 Cleanup", c6, e6, None, "#888", 0.5)

            # Panel 6: Intersection graph (road_class HUSL + node types)
            fig6, ax6 = plt.subplots(figsize=(10, 8))
            ax6.set_title("6 Intersection Graph", fontsize=12)
            draw_intersection_graph(ax6, c7, e7, nt, lanes, rc, geoms, legend=True)
            ax6.set_aspect("equal")
            ax6.axis("off")
            fig6.savefig(str(folder / "6_Intersection_Graph.png"), dpi=200, bbox_inches="tight")
            plt.close(fig6)

            # Panel 7: HD Map (tactics2d official renderer)
            try:
                render_map(hd, str(folder / "7_HD_Map.png"))
            except Exception as exc:
                print(f"  HD Map failed: {exc}")
                fig7, ax7 = plt.subplots(figsize=(8, 8))
                ax7.text(
                    0.5,
                    0.5,
                    f"HD Map failed:\n{exc}",
                    ha="center",
                    va="center",
                    transform=ax7.transAxes,
                    fontsize=12,
                )
                ax7.axis("off")
                fig7.savefig(str(folder / "7_HD_Map.png"), dpi=200)
                plt.close(fig7)

            _combined(
                folder,
                [
                    "0_Field_Only.png",
                    "1_VQ_Field.png",
                    "2_Skeleton.png",
                    "3_Scaled+clean.png",
                    "4_Growth.png",
                    "5_Cleanup.png",
                    "6_Intersection_Graph.png",
                    "7_HD_Map.png",
                ],
            )
            print(f"  → {folder}")
            idx += 1

    print(f"Done ({idx} maps)")


# ═══════════════════════════════════════════════════════════════════════
#  sweep mode
# ═══════════════════════════════════════════════════════════════════════


def run_sweep_mode(args):
    """Style / density / diversity road-field sweeps (per-panel + grid + combined)."""
    vq_name = Path(args.vq).parent.name
    tfm_name = Path(args.transformer).parent.name
    out_root = Path(f"analysis/sweep/{vq_name}+{tfm_name}")
    os.makedirs(out_root, exist_ok=True)

    gen = make_generator(
        args.vq,
        args.transformer,
        cache_path=args.cache,
        device=args.device,
        d_model=args.d_model,
        num_layers=args.num_layers,
        nhead=args.nhead,
        num_codes=args.codes,
        resolution=args.res,
        code_map_size=args.code_map,
        use_adaln=True,  # phase_a trained with AdaLN (matches make_generator default)
    )
    print(f"Output → {out_root}/")

    def gen_field(cond):
        """Return (road field, code_map) for a 1×11 condition."""
        with torch.no_grad():
            code_map = gen.generate_code_map(cond, temperature=0.75, top_p=0.65)
            field = torch.sigmoid(gen.vq.decode_from_code(code_map))
        return field, code_map

    def save_panels(cat: str, items: list[tuple[str, torch.Tensor]], n_cols: int, suptitle: str):
        """Save each panel independently + a ``grid.png``; returns the folder."""
        folder = out_root / cat
        folder.mkdir(parents=True, exist_ok=True)
        for name, fld in items:
            fig, ax = plt.subplots(figsize=(3.2, 3.2))
            ax.imshow(fld[0, 5].cpu().numpy(), cmap="gray_r", vmin=0, vmax=1)
            ax.set_title(name, fontsize=9)
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(str(folder / f"{cat}_{name}.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)
        n = len(items)
        nrow = (n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(nrow, n_cols, figsize=(3.2 * n_cols, 3.2 * nrow))
        for ax, (name, fld) in zip(axes.flat, items):
            ax.imshow(fld[0, 5].cpu().numpy(), cmap="gray_r", vmin=0, vmax=1)
            ax.set_title(name, fontsize=9)
            ax.axis("off")
        fig.suptitle(suptitle, fontsize=12)
        fig.tight_layout()
        fig.savefig(str(folder / "grid.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  {cat}: {len(items)} panels + grid.png")
        return folder

    # Style sweep
    style_items = []
    for name, sv in STYLES.items():
        cond = torch.zeros(1, 11, device=args.device)
        cond[0, :6] = torch.tensor(sv, device=args.device)
        cond[0, 6] = 1.0
        style_items.append((name, gen_field(cond)[0]))
    save_panels("style", style_items, n_cols=3, suptitle="Style sweep")

    # Density sweep
    density_items = []
    for d in DENSITIES:
        cond = torch.zeros(1, 11, device=args.device)
        cond[0, 6] = d / 20.0
        density_items.append((f"density={d}", gen_field(cond)[0]))
    save_panels("density", density_items, n_cols=4, suptitle="Density sweep")

    # Diversity (same condition, different samples)
    cond = torch.zeros(1, 11, device=args.device)
    cond[0, :6] = torch.tensor(STYLES["Gridiron"], device=args.device)
    cond[0, 6] = 1.0
    diversity_items = []
    for i in range(6):
        fld, code_map = gen_field(cond)
        diversity_items.append((f"sample {i} ({len(code_map[0].unique())} codes)", fld))
    save_panels("diversity", diversity_items, n_cols=3, suptitle="Diversity (Gridiron)")

    # Combined: the three grids side by side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, cat in zip(axes, ["style", "density", "diversity"]):
        img = plt.imread(str(out_root / cat / "grid.png"))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(cat, fontsize=10)
    fig.suptitle("Condition sweep overview", fontsize=13)
    fig.tight_layout()
    fig.savefig(str(out_root / "combined.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  combined.png")

    # Full pipeline (optional)
    if args.run_full:
        MS = 2000.0 if args.res == 128 else 5000.0
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for i, d in enumerate([5, 10, 20, 30]):
            cond = torch.zeros(1, 11, device=args.device)
            cond[0, 6] = d / 20.0
            road = gen_field(cond)[0][0, 5].cpu().numpy()

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
        plt.savefig(str(out_root / "full_pipeline.png"), dpi=300, bbox_inches="tight")
        plt.close()
        print("  full_pipeline.png")

    print(f"\nAll outputs → {out_root}/")


# ═══════════════════════════════════════════════════════════════════════
#  vq mode
# ═══════════════════════════════════════════════════════════════════════


def run_vq_mode(args):
    """VQ-VAE reconstruction panels (per-sample folder + grid)."""
    os.makedirs(VQ_OUTPUT, exist_ok=True)

    CONFIG.train_split_path = "data/urban_prior/2km/splits/train.parquet"
    CONFIG.val_split_path = "data/urban_prior/2km/splits/val.parquet"

    loader = make_field_dataloader(
        "val", batch_size=N_SAMPLES, shuffle=True, num_workers=2, limit_samples=200, resolution=128
    )

    model = VQVAE(resolution=128, num_codes=512, code_map_size=32).to(DEVICE)
    state = torch.load(VQ_CKPT, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {VQ_CKPT}")

    batch = next(iter(loader))
    field = batch["field"].to(DEVICE)

    with torch.no_grad():
        recon, indices, info = model(field)

    orig_bin = (field[:, 0] > 0.5).float()
    rec_bin = (torch.sigmoid(recon[:, 0]) > 0.5).float()
    inter = (orig_bin * rec_bin).sum(dim=(1, 2))
    union = orig_bin.sum(dim=(1, 2)) + rec_bin.sum(dim=(1, 2)) - inter
    ious = inter / (union + 1e-8)

    dirs = plot_recon_grid(
        field.cpu(), torch.sigmoid(recon).cpu(), ious.tolist(), VQ_OUTPUT, N_SAMPLES
    )
    print(f"  Saved {len(dirs)} sample folders → {VQ_OUTPUT}/")
    print(f"  Avg IoU: {ious.mean():.4f}")


# ═══════════════════════════════════════════════════════════════════════
#  entry point
# ═══════════════════════════════════════════════════════════════════════


def main():
    ap = argparse.ArgumentParser(description="Visualise RoadWeaver generation results.")
    sub = ap.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("pipeline", help="8-panel per-map pipeline intermediates")
    p.add_argument("--n", type=int, default=12, help="max maps to render")
    p.add_argument("--device", default="cuda")
    p.set_defaults(func=run_pipeline_mode)

    s = sub.add_parser("sweep", help="style / density / diversity road-field sweeps")
    s.add_argument("--vq", default=VQ_CKPT)
    s.add_argument("--transformer", default=TFM_CKPT)
    s.add_argument("--cache", default=CACHE)
    s.add_argument("--res", type=int, default=128)
    s.add_argument("--codes", type=int, default=512)
    s.add_argument("--code-map", type=int, default=32)
    s.add_argument("--d-model", type=int, default=256)
    s.add_argument("--num-layers", type=int, default=6)
    s.add_argument("--nhead", type=int, default=4)
    s.add_argument("--device", default="cuda")
    s.add_argument(
        "--run-full", action="store_true", help="run full pipeline (field → graph → growth)"
    )
    s.set_defaults(func=run_sweep_mode)

    v = sub.add_parser("vq", help="VQ-VAE reconstruction panels")
    v.set_defaults(func=run_vq_mode)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

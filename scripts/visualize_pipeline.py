#!/usr/bin/env python3
"""Per-sample pipeline visualization — 2x3 grid + HD Map.

Layout:
  Row 0: Field | Raw skeleton | Skeleton (scaled+cleaned)
  Row 1: Growth | Cleanup | Compressed (edges=road_class HUSL color, nodes=type)
  Row 2: HD Map (tactics2d lanes + intersection/roundabout geometry)

Generates 24 images covering 6 map sizes × 4 density/style combinations.

Usage:
    conda activate road-weaver
    python scripts/visualize_pipeline.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import husl
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from shapely import is_empty

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tactics2d"))
from hdmap_generator import assign_lanes, graph_to_map
from hdmap_generator.geometry import chaikin
from network_generator.backbone.config import CONFIG
from network_generator.backbone.dataset import make_field_dataloader
from network_generator.backbone.generator import Generator, make_generator
from network_generator.growth.config import GrowthConfig
from network_generator.growth.growth import grow
from network_generator.topology.connector import EndpointConnector
from network_generator.topology.graph_cleanup import (
    clean_parallel_roads,
    clean_sharp_angles,
    fix_edge_crossings,
    keep_lcc,
    prune_dead_ends,
    snap_endpoints,
)
from network_generator.topology.graph_intersection import (
    classify_nodes,
    compress_to_intersection_graph,
    detect_roundabouts,
    propagate_road_class,
)
from network_generator.topology.graph_merge import merge_close_nodes
from network_generator.topology.graph_refine import (
    align_geometries_to_nodes,
    fix_abnormal_edges,
    merge_compressed_graph,
    merge_nearby_junctions,
    merge_parallel_edges,
)
from network_generator.topology.graph_simplify import simplify_chains
from network_generator.topology.graph_utils import (
    NT_CURVE,
    NT_ENDPOINT,
    NT_JUNCTION,
    NT_ROUNDABOUT,
    NT_WAYPOINT,
)
from utils.geometry import segment_intersection as _segment_intersection
from utils.visualization import _draw_field, _draw_graph


def _fix_growth_crossings(coords, edge_index, map_max):
    """Add nodes where non-adjacent growth-graph edges cross."""
    adj = {i: set() for i in range(len(coords))}
    for u, v in edge_index:
        adj[int(u)].add(int(v))
        adj[int(v)].add(int(u))
    max_n = len(coords)
    new_c = list(coords)
    # Track edge splits: each crossing splits both edges, creating 2 new edges each
    splits = {}  # {old_edge_idx: [(fraction, node_idx), ...]}
    for i in range(len(edge_index)):
        u1, v1 = int(edge_index[i, 0]), int(edge_index[i, 1])
        p1, p2 = coords[u1], coords[v1]
        for j in range(i + 1, len(edge_index)):
            u2, v2 = int(edge_index[j, 0]), int(edge_index[j, 1])
            if {u1, v1} & {u2, v2}:
                continue
            inter = _segment_intersection(p1, p2, coords[u2], coords[v2])
            if inter is not None:
                nid = len(new_c)
                new_c.append(inter)
                splits.setdefault(i, []).append(
                    (np.linalg.norm(inter - coords[u1]) / max(np.linalg.norm(p2 - p1), 1e-12), nid)
                )
                splits.setdefault(j, []).append(
                    (np.linalg.norm(inter - coords[u2]) / max(np.linalg.norm(p2 - p1), 1e-12), nid)
                )
    if not splits:
        return coords, edge_index
    # Rebuild edge list with splits
    new_e = []
    for k in range(len(edge_index)):
        if k not in splits:
            new_e.append(tuple(edge_index[k]))
        else:
            u, v = int(edge_index[k, 0]), int(edge_index[k, 1])
            fracs = sorted(splits[k], key=lambda x: x[0])
            prev = u
            for _, nid in fracs:
                new_e.append((prev, nid))
                prev = nid
            new_e.append((prev, v))
    print(f"  [GrowthCross] {len(splits)} edges split, {len(new_c) - max_n} nodes added")
    return np.array(new_c), np.array(new_e, dtype=np.int64)


def main():
    CONFIG.resolution = 128
    CONFIG.code_map_size = 32
    CONFIG.val_split_path = "data/urban_prior/2km/splits_style/val.parquet"
    VQ_CKPT = "runtimes/vq_vae_2km_phase_a/checkpoints/best.pth"
    MODEL_CKPT = "runtimes/transformer_2km_phase_a/checkpoints/best.pth"
    CACHE = "cache/masked_code_maps_2km_phase_a/train.npz"
    DEVICE = "cuda"
    VQ_MAP_SIZE_M = 2000.0
    MIN_SPACING_M = 80.0
    MAP_SIZES = [(2000, 2000), (3000, 2000), (2000, 3000), (4000, 2000), (2000, 4000), (5000, 2000)]
    N_PER_SIZE = 2

    # ── Load ──
    print("[Viz] Loading...")
    gen = make_generator(VQ_CKPT, MODEL_CKPT, cache_path=CACHE, device=DEVICE)

    dl = make_field_dataloader(
        "val", batch_size=6, num_workers=0, limit_samples=6, cache_fields=False
    )
    batch = next(iter(dl))
    style, struct = batch["style_vector"], batch["structural_priors"]
    cond = torch.cat([style, struct], dim=1).to(DEVICE)
    pnames = ["Gridiron", "Linear", "No Pat", "Organic", "Radial", "Trib"]

    # ── Generate ──
    all_data = []
    total = len(MAP_SIZES) * N_PER_SIZE
    done = 0
    for mw, mh in MAP_SIZES:
        map_max = max(mw, mh)
        for i in range(N_PER_SIZE):
            sid = i % 6
            pid = int(style[sid].argmax())
            d_val = float(struct[sid, 0])
            done += 1
            t0 = time.time()
            print(f"[{done}/{total}] {pnames[pid]} {mw}x{mh} d={d_val:.1f}")
            with torch.no_grad():
                raw = gen.generate(
                    cond[sid : sid + 1], anchor_ratio=0.08, temperature=0.75, top_p=0.65, seed=sid
                )
            fld = raw["road_field"]
            c1, e1 = raw["coords"], raw["edge_index"]
            if len(c1) < 5:
                print("  skip")
                continue
            try:
                conn = EndpointConnector(map_size_m=VQ_MAP_SIZE_M).run(
                    raw,
                    fld,
                    max_connections=30,
                    connect_remaining=True,
                    max_remaining_m=600,
                    simplify=False,
                )
                simp = simplify_chains(
                    conn["coords"],
                    conn["edge_index"],
                    angle_threshold_deg=15,
                    dp_epsilon_norm=0.002,
                )
                c2, e2 = simp[0], simp[1]
            except:
                c2, e2 = c1, e1

            sx, sy = mw / VQ_MAP_SIZE_M, mh / VQ_MAP_SIZE_M
            c3 = c2 * np.array([sx, sy])
            # Always pass field for A* guidance; even if map is non-square,
            # partial coverage is better than none
            fg = fld
            md = max(0.005, MIN_SPACING_M / map_max * 0.25)
            merged = merge_close_nodes(
                c3, e2, np.zeros(len(c3), dtype=np.int64), merge_dist=md, map_size_m=map_max
            )
            c4, e4 = merged[0], merged[1]

            try:
                gc = GrowthConfig.from_condition(
                    cond[sid].cpu().numpy(), local_spacing_m=80.0, map_size_m=map_max
                )
                gc.map_width_m = mw
                gc.map_height_m = mh
                grown = grow(c4 * np.array([mw, mh]), e4, np.zeros(len(c4), dtype=np.int64), fg, gc)
                c5 = grown["coords"] / np.array([mw, mh])
                e5 = grown["edge_index"]
                rc = grown.get("road_class", np.ones(len(e5), dtype=np.int64))
            except:
                continue

            # Merge close nodes in growth graph before cleanup
            _md = max(0.003, 30.0 / map_max)
            _mrg = merge_close_nodes(
                c5, e5, np.zeros(len(c5), dtype=np.int64), merge_dist=_md, map_size_m=map_max
            )
            c5, e5 = _mrg[0], _mrg[1]
            # Fix edge crossings in growth graph (add nodes at crossing points)
            c5, e5 = _fix_growth_crossings(c5, e5, map_max)

            c6, e6 = prune_dead_ends(c5, e5, 120.0, map_max)
            c6, e6 = keep_lcc(c6, e6)
            c6, e6 = clean_sharp_angles(c6, e6, min_deg=15.0)
            c6, e6 = snap_endpoints(c6, e6, map_max, snap_dist_m=50.0)
            c6, e6 = keep_lcc(c6, e6)
            # Compress → merge nearby junctions → fix abnormal → crossings → parallel cleanup
            c7, e7, geoms = compress_to_intersection_graph(c6, e6)
            c7, e7, geoms = merge_compressed_graph(c7, e7, geoms, map_max, merge_dist_m=30.0)
            fix_abnormal_edges(c7, e7, geoms, map_max)
            c7, e7, geoms = fix_edge_crossings(c7, e7, geoms, map_max)
            c7, e7, geoms = clean_parallel_roads(
                c7, e7, geoms, map_max, angle_deg=20.0, max_dist_m=30.0
            )
            nt = classify_nodes(c7, e7, map_max, merge_dist_m=50.0, compressed=True)

            # ── Inter-edge parallel dedup (non-adjacent edges) ────────
            c7, e7, geoms = merge_parallel_edges(
                c7, e7, geoms, map_max, angle_deg=20.0, dist_m=30.0
            )
            # Re-classify after structural changes (merge_parallel may remove nodes)
            nt = classify_nodes(c7, e7, map_max, merge_dist_m=50.0, compressed=True)
            # ── Physically merge nearby junction nodes ────────────────
            c7, e7, nt, geoms = merge_nearby_junctions(
                c7, e7, nt, geoms, map_max, merge_dist_m=50.0
            )
            # NOTE: the tactics2d Intersection builder supports 3-6 arms
            # directly, so high-degree junctions are NOT split here.

            # ── Second parallel-road cleanup (after structural changes)
            c7, e7, geoms = clean_parallel_roads(
                c7, e7, geoms, map_max, angle_deg=20.0, max_dist_m=30.0
            )

            # ── Re-fix crossings created by structural ops ───────────
            # merge_nearby_junctions moves edge endpoints and can introduce
            # new geometric crossings; fix them after all structural changes
            # (matches pipeline.py generate_branch).
            c7, e7, geoms = fix_edge_crossings(c7, e7, geoms, map_max)

            # ── Re-align geometry endpoints to node positions ────────
            geoms = align_geometries_to_nodes(c7, e7, geoms)

            # ── Smooth graph geometries before lane building ─────────
            geoms = [chaikin(g, iterations=2) if len(g) >= 3 else g for g in geoms]

            # Re-run classify_nodes after structural changes
            nt = classify_nodes(c7, e7, map_max, merge_dist_m=50.0, compressed=True)

            # Propagate road_class through compression (rc indices are pre-compress)
            rc_int = propagate_road_class(c6, e6, rc, e7)

            # Lane assignment via shared public function
            lanes = assign_lanes(c7, e7, geoms, rc_int, density=d_val)
            all_data.append(
                {
                    "field": fld,
                    "pattern": pnames[pid],
                    "density": d_val,
                    "c1": c1,
                    "e1": e1,
                    "c2": c2,
                    "e2": e2,
                    "c4": c4,
                    "e4": e4,
                    "c5": c5,
                    "e5": e5,
                    "c6": c6,
                    "e6": e6,
                    "c7": c7,
                    "e7": e7,
                    "geoms": geoms,
                    "nt": nt,
                    "lanes": lanes,
                    "rc_int": rc_int,
                    "map_w": mw,
                    "map_h": mh,
                    "time": time.time() - t0,
                    "road_class": rc_int,
                }
            )
            nj = int((nt == 1).sum())
            nr = int((nt == 3).sum())
            ne = int((nt == 4).sum())
            print(f"  {len(c7)}n {len(e7)}e J={nj} R={nr} E={ne} {time.time()-t0:.1f}s")

    # ── Plot ──
    print(f"\nPlotting {len(all_data)}...")
    OUT = Path("analysis/roadweaver")
    SKEL = "#F9A825"

    for idx, d in enumerate(all_data):
        folder = OUT / f"map_{idx}_{d['map_w']}x{d['map_h']}"
        folder.mkdir(parents=True, exist_ok=True)

        # ---- Panel 0: Field only (no graph overlay) --------------
        fig0, ax0 = plt.subplots(figsize=(8, 8))
        ax0.set_title("0 Field Only", fontsize=12)
        _draw_field(ax0, d["field"], extent=(0, 1, 0, 1), origin="lower")
        ax0.set_xlim(-0.02, 1.02)
        ax0.set_ylim(-0.02, 1.02)
        ax0.set_aspect("equal")
        ax0.axis("off")
        fig0.savefig(folder / "0_Field_Only.png", dpi=200, bbox_inches="tight")
        plt.close(fig0)

        # Panel definitions: (filename, title, coords, edges, bg_field, node_color, edge_lw)
        panels = [
            ("1_VQ_Field.png", "1 VQ Field", d["c1"], d["e1"], d["field"], SKEL, 0.7),
            ("2_Skeleton.png", "2 Skeleton", d["c2"], d["e2"], d["field"], SKEL, 0.7),
            ("3_Scaled+clean.png", "3 Scaled+clean", d["c4"], d["e4"], None, SKEL, 0.7),
            ("4_Growth.png", "4 Growth", d["c5"], d["e5"], None, "#888", 0.5),
            ("5_Cleanup.png", "5 Cleanup", d["c6"], d["e6"], None, "#888", 0.5),
        ]

        # ---- Panels 1-5: simple graph / field renders ------------
        for fname, title, c, e, bg, nc, lw in panels:
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
            fig.savefig(folder / fname, dpi=200, bbox_inches="tight")
            plt.close(fig)

        # ---- Panel 6: Intersection Graph -------------------------
        c7, e7 = d["c7"], d["e7"]
        nt = d["nt"]
        lanes_arr = d["lanes"]
        rc_arr = d.get("rc_int")
        geoms = d["geoms"]

        fig6, ax6 = plt.subplots(figsize=(10, 8))
        ax6.set_title("6 Intersection Graph", fontsize=12)
        ax6.set_facecolor("#f5f5f5")

        n_j = int((nt == NT_JUNCTION).sum())
        n_r = int((nt == NT_ROUNDABOUT).sum())
        n_e = int((nt == NT_ENDPOINT).sum())

        if len(c7):
            # Draw edges colored by road class
            rc_colors = {
                1: husl.husl_to_hex(150, 65, 55),  # green
                2: husl.husl_to_hex(280, 65, 55),  # purple
                3: husl.husl_to_hex(40, 65, 55),
            }  # orange
            for j, (u, v) in enumerate(e7):
                lw = min(3.0, 0.3 + 0.35 * int(lanes_arr[j]) if j < len(lanes_arr) else 0.5)
                rc = int(rc_arr[j]) if rc_arr is not None and j < len(rc_arr) else 2
                color = rc_colors.get(rc, "#888")
                geom = geoms[j] if j < len(geoms) else None
                if geom is not None and len(geom) > 2:
                    ax6.plot(geom[:, 0], geom[:, 1], color=color, lw=lw, alpha=0.7)
                else:
                    ax6.plot(
                        [c7[u, 0], c7[v, 0]], [c7[u, 1], c7[v, 1]], color=color, lw=lw, alpha=0.7
                    )

            # Draw nodes
            if n_j:
                ax6.scatter(
                    c7[nt == NT_JUNCTION, 0],
                    c7[nt == NT_JUNCTION, 1],
                    c="#1565C0",
                    s=40,
                    edgecolors="black",
                    lw=0.5,
                    zorder=3,
                )
            if n_r:
                ax6.scatter(
                    c7[nt == NT_ROUNDABOUT, 0],
                    c7[nt == NT_ROUNDABOUT, 1],
                    c="#E53935",
                    s=40,
                    edgecolors="black",
                    lw=0.5,
                    zorder=3,
                )
            if n_e:
                ax6.scatter(
                    c7[nt == NT_ENDPOINT, 0],
                    c7[nt == NT_ENDPOINT, 1],
                    c="#FDD835",
                    s=16,
                    edgecolors="black",
                    lw=0.3,
                    zorder=3,
                )

            xs, ys = c7[:, 0], c7[:, 1]
            mx = max(0.02, (xs.max() - xs.min()) * 0.05)
            my = max(0.02, (ys.max() - ys.min()) * 0.05)
            ax6.set_xlim(xs.min() - mx, xs.max() + mx)
            ax6.set_ylim(ys.min() - my, ys.max() + my)

        # Legend outside the plot
        from matplotlib.lines import Line2D

        legend_items = []
        if n_j:
            legend_items.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="#1565C0",
                    markeredgecolor="black",
                    markersize=8,
                    label=f"Junction ({n_j})",
                )
            )
        if n_r:
            legend_items.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="#E53935",
                    markeredgecolor="black",
                    markersize=8,
                    label=f"Roundabout ({n_r})",
                )
            )
        if n_e:
            legend_items.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="#FDD835",
                    markeredgecolor="black",
                    markersize=6,
                    label=f"Endpoint ({n_e})",
                )
            )
        for rc_label, rc_val, hue in [
            ("Primary (1)", 1, 150),
            ("Secondary (2)", 2, 280),
            ("Local (3)", 3, 40),
        ]:
            legend_items.append(
                Line2D([0], [0], color=husl.husl_to_hex(hue, 65, 55), lw=2, label=rc_label)
            )
        for lw_val, label in [(0.65, "1 lane"), (1.0, "2 lanes"), (1.35, "3+ lanes")]:
            legend_items.append(Line2D([0], [0], color="#555", lw=lw_val, label=label))

        ax6.legend(
            handles=legend_items,
            fontsize=8,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
            framealpha=0.9,
        )
        ax6.set_aspect("equal")
        ax6.axis("off")
        fig6.savefig(folder / "6_Intersection_Graph.png", dpi=200, bbox_inches="tight")
        plt.close(fig6)

        # ---- Panel 7: HD Map (tactics2d renderer) ----------------
        try:
            hd = graph_to_map(
                c7,
                e7,
                geoms,
                node_types=nt,
                lanes_per_dir=lanes_arr,
                road_class=rc_arr,
                map_w=d["map_w"],
                map_h=d["map_h"],
            )
            from render_tactics2d import render_map

            render_map(hd, str(folder / "7_HD_Map.png"))
        except Exception as exc:
            print(f"  [map_{idx}] HD Map failed: {exc}")
            # Fallback: save empty placeholder
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
            fig7.savefig(folder / "7_HD_Map.png", dpi=200)
            plt.close(fig7)

        print(f"  [{idx}] → {folder}")

    print(f"Done ({len(all_data)} maps)")


if __name__ == "__main__":
    main()

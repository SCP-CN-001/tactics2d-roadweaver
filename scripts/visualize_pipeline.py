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
    split_high_degree_junctions,
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
    OUT = Path("analysis/viz")
    OUT.mkdir(parents=True, exist_ok=True)
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
            # ── Split degree>=5 junctions into sub-junctions ──────────
            c7, e7, nt, geoms = split_high_degree_junctions(
                c7, e7, nt, geoms, map_max, split_radius_m=3.0
            )

            # ── Second parallel-road cleanup (after structural changes)
            c7, e7, geoms = clean_parallel_roads(
                c7, e7, geoms, map_max, angle_deg=20.0, max_dist_m=30.0
            )

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
    SKEL = "#F9A825"

    for idx, d in enumerate(all_data):
        fig = plt.figure(figsize=(18, 9))
        gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.3], hspace=0.2, wspace=0.15)
        fig.suptitle(
            f"{d['pattern']} dens={d['density']:.1f} {d['map_w']}x{d['map_h']}m {d['time']:.1f}s",
            fontsize=13,
            y=0.98,
        )

        panels = [
            (fig.add_subplot(gs[0, 0]), "1 VQ Field", d["c1"], d["e1"], d["field"], SKEL, 0.7),
            (fig.add_subplot(gs[0, 1]), "2 Skeleton", d["c2"], d["e2"], d["field"], SKEL, 0.7),
            (fig.add_subplot(gs[0, 2]), "3 Scaled+clean", d["c4"], d["e4"], None, SKEL, 0.7),
            (fig.add_subplot(gs[1, 0]), "4 Growth", d["c5"], d["e5"], None, "#888", 0.5),
            (fig.add_subplot(gs[1, 1]), "5 Cleanup", d["c6"], d["e6"], None, "#888", 0.5),
            (fig.add_subplot(gs[1, 2]), "6 Intersection Graph", d["c7"], d["e7"], None, None, None),
        ]

        for ax, title, c, e, bg, nc, lw in panels:
            ax.set_title(title, fontsize=10)
            if bg is not None:
                _draw_field(ax, bg, extent=(0, 1, 0, 1), origin="lower")
            else:
                ax.set_facecolor("#f5f5f5")
            px = c if len(c) else []
            if title.startswith("6"):
                nt = d["nt"]
                lanes_arr = d["lanes"]
                rc_arr = d.get("rc_int")
                if len(px):
                    for j, (u, v) in enumerate(e):
                        lw = min(3.0, 0.3 + 0.35 * int(lanes_arr[j]) if j < len(lanes_arr) else 0.5)
                        rc = int(rc_arr[j]) if rc_arr is not None and j < len(rc_arr) else 2
                        hue = {1: 150, 2: 280, 3: 40}.get(rc, 100)
                        color = husl.husl_to_hex(hue, 65, 55)
                        geom = d["geoms"][j] if j < len(d["geoms"]) else None
                        if geom is not None and len(geom) > 2:
                            ax.plot(geom[:, 0], geom[:, 1], color=color, lw=lw, alpha=0.7)
                        else:
                            ax.plot(
                                [px[u, 0], px[v, 0]],
                                [px[u, 1], px[v, 1]],
                                color=color,
                                lw=lw,
                                alpha=0.7,
                            )
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
                            label=f"J ({jm.sum()})",
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
                            label=f"R ({rm.sum()})",
                        )
                    if em.any():
                        ax.scatter(
                            px[em, 0],
                            px[em, 1],
                            c="#FDD835",
                            s=12,
                            edgecolors="black",
                            lw=0.2,
                            zorder=3,
                            label=f"E ({em.sum()})",
                        )
                ax.legend(fontsize=7, loc="lower right", framealpha=0.8)
                n1, n2, n3 = (
                    int((lanes_arr == 1).sum()),
                    int((lanes_arr == 2).sum()),
                    int((lanes_arr >= 3).sum()),
                )
                ax.text(
                    0.02,
                    0.98,
                    f"L:1x{n1} 2x{n2} 3+x{n3}",
                    transform=ax.transAxes,
                    fontsize=8,
                    va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )
            else:
                if len(px):
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
            ax.text(
                0.02,
                0.02,
                f"{len(c)}n {len(e)}e",
                transform=ax.transAxes,
                fontsize=8,
                va="bottom",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )
            if len(px):
                xs, ys = px[:, 0], px[:, 1]
                mx = max(0.02, (xs.max() - xs.min()) * 0.05)
                my = max(0.02, (ys.max() - ys.min()) * 0.05)
                ax.set_xlim(xs.min() - mx, xs.max() + mx)
                ax.set_ylim(ys.min() - my, ys.max() + my)
            else:
                ax.set_xlim(-0.02, 1.02)
                ax.set_ylim(-0.02, 1.02)
            ax.set_aspect("equal")
            ax.axis("off")

        # ---- Right column: HD Map (spans both rows) ----
        ax_hd = fig.add_subplot(gs[0:2, 3])
        try:
            hd = graph_to_map(
                d["c7"],
                d["e7"],
                d["geoms"],
                node_types=d["nt"],
                lanes_per_dir=d["lanes"],
                road_class=d.get("road_class"),
                map_w=d["map_w"],
                map_h=d["map_h"],
            )
            for lane in hd.lanes.values():
                try:
                    if lane.id_.startswith("e"):
                        c, lw = "#222", 0.5
                    elif lane.id_.startswith("ra"):
                        c, lw = "#666", 0.4
                    elif lane.id_.startswith("int"):
                        c, lw = "#999", 0.3
                    else:
                        c, lw = "#222", 0.3
                    for side in (lane.left_side, lane.right_side):
                        if side is not None and not is_empty(side):
                            x, y = side.xy
                            ax_hd.plot(x, y, color=c, lw=lw, alpha=0.85)
                except:
                    pass
            for junc in (hd.junctions or {}).values():
                try:
                    shape = getattr(junc, "custom_tags", {}).get("shape", [])
                    if shape and len(shape) > 2:
                        xs, ys = zip(*shape)
                        ax_hd.fill(xs, ys, alpha=0.25, color="#888")
                        ax_hd.plot(xs + (xs[0],), ys + (ys[0],), color="#555", lw=1.5, alpha=0.7)
                except:
                    pass
            for area in (hd.areas or {}).values():
                try:
                    if area.geometry and not is_empty(area.geometry):
                        x, y = area.geometry.exterior.xy
                        ax_hd.fill(x, y, alpha=0.25, color="#aaa")
                except:
                    pass
            all_l = list(hd.lanes.values())
            n_road = sum(1 for l in all_l if l.id_.startswith("e"))
            n_ra = sum(1 for l in all_l if l.id_.startswith("ra"))
            n_int = sum(1 for l in all_l if l.id_.startswith("int"))
            n_junc = len(hd.junctions or {})
            n_area = len(hd.areas or {})
            dead = sum(1 for l in all_l if l.id_.startswith("e") and not l.successors)
            ax_hd.set_title(
                f"HD Map -- {n_road}r {n_ra}ra {n_int}int {n_junc}junc {dead}dead", fontsize=10
            )
        except Exception as exc:
            ax_hd.text(
                0.5,
                0.5,
                f"HD Map failed: {exc}",
                ha="center",
                va="center",
                transform=ax_hd.transAxes,
                fontsize=10,
            )
        ax_hd.set_aspect("equal")
        ax_hd.axis("off")

        plt.savefig(
            OUT / f"s{idx}_{d['pattern']}_{d['map_w']}x{d['map_h']}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    print(f"Done ({len(all_data)} images)")


if __name__ == "__main__":
    main()

"""
G1: individual front growth (full coverage) + A* endpoint closure.
G2: raster-based face infill.
After all growth: merge close nodes.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point, Polygon

from .config import GrowthConfig
from .tensor_field import GraphTensorField, normalize
from utils.pathfinding import astar_connect_path, cost_map_from_road
from utils.graph_ops import merge_close_nodes


def _nid(G: nx.Graph) -> int:
    m = -1
    for n in G.nodes:
        if isinstance(n, int) and n > m:
            m = n
    return m + 1


# ── graph I/O ─────────────────────────────────────────────────────────


def to_graph(coords_m, ei, nt, level=0):
    G = nx.Graph()
    for i, p in enumerate(coords_m):
        G.add_node(i, pos=p.copy())
    for u, v in ei:
        u, v = int(u), int(v)
        pu, pv = coords_m[u], coords_m[v]
        G.add_edge(u, v, geom=LineString([pu, pv]),
                    length=float(np.linalg.norm(pv - pu)), level=level)
    return G


def to_dict(G, map_m):
    nodes = sorted(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    c = np.array([G.nodes[n]["pos"] for n in nodes], dtype=np.float32)
    e, rc, el = [], [], []
    deg = {n: 0 for n in nodes}
    for u, v, d in G.edges(data=True):
        e.append([idx[u], idx[v]])
        lv = int(d.get("level", 0))
        rc.append(min(lv + 1, 3))
        el.append(float(d.get("length", 1)))
        deg[u] += 1; deg[v] += 1
    e = np.array(e, dtype=np.int64).reshape(-1, 2) if e else np.empty((0, 2), dtype=np.int64)
    nt = np.array([1 if deg[n] >= 3 else (4 if deg[n] <= 1 else 0) for n in nodes], dtype=np.int64)
    return {"coords": c, "edge_index": e, "edge_lengths_m": np.array(el, dtype=np.float32),
            "road_class": np.array(rc, dtype=np.int64), "node_types": nt, "map_size_m": map_m}


def add_e(G, u, v, level):
    if G.has_edge(u, v): return
    pu, pv = G.nodes[u]["pos"], G.nodes[v]["pos"]
    G.add_edge(u, v, geom=LineString([pu, pv]),
                length=float(np.linalg.norm(pv - pu)), level=level)


def split_e(G, u, v, pt):
    if not G.has_edge(u, v): return None
    d = G.edges[u, v]
    g = d["geom"]
    da = g.project(Point(pt))
    if da < 1e-6: return u
    if g.length - da < 1e-6: return v
    nid = _nid(G)
    G.add_node(nid, pos=pt.copy())
    lv = d.get("level", 3)
    G.remove_edge(u, v)
    G.add_edge(u, nid, geom=LineString([np.asarray(g.coords[0]), pt]),
               length=float(np.linalg.norm(np.asarray(g.coords[0]) - pt)), level=lv)
    G.add_edge(nid, v, geom=LineString([pt, np.asarray(g.coords[-1])]),
               length=float(np.linalg.norm(pt - np.asarray(g.coords[-1]))), level=lv)
    return nid


# ── G1: seeds ─────────────────────────────────────────────────────────


def gen_seeds(G, config, rng=None):
    """Sample seeds along H edges (both sides)."""
    if rng is None:
        rng = random.Random(config.random_seed)
    seeds = []
    spacing = config.g1_seed_spacing_m
    edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("level", 0) == 0]

    for u, v, d in edges:
        pu, pv = G.nodes[u]["pos"], G.nodes[v]["pos"]
        ev = pv - pu
        length = float(np.linalg.norm(ev))
        if length < spacing * 0.3:
            continue
        ed = ev / length
        normal = np.array([-ed[1], ed[0]])

        # Number of seeds along this edge
        n = max(1, int(length / spacing * 1.2))
        t_vals = np.linspace(0.15, 0.85, n)
        for t in t_vals:
            pt = pu + t * ev
            # Split H edge at seed point — node becomes part of H topology
            nid = split_e(G, u, v, pt)
            if nid is None:
                continue
            for sign in (-1.0, 1.0):
                direction = normalize(sign * normal + rng.uniform(-0.15, 0.15) * ed)
                seeds.append({"nid": nid, "pos": pt.copy(), "dir": direction, "pe": (u, nid)})
    return seeds


# ── G1: growth direction ─────────────────────────────────────────────


def grow_dir(pos, prev, tf, config):
    tensor_dir = tf.choose_direction(pos, prev)
    r = config.w_tensor * tensor_dir + 0.30 * prev
    r = normalize(r)
    if np.linalg.norm(r) < 1e-9: return prev.copy()
    dv = np.clip(np.dot(r, prev), -1.0, 1.0)
    ang = math.degrees(math.acos(dv))
    if ang > config.max_turn_deg:
        ratio = config.max_turn_deg / max(ang, 1e-6)
        r = normalize((1.0 - ratio) * prev + ratio * r)
    return r


# ── G1: snap check ────────────────────────────────────────────────────


def find_snap(G, pos, p_edges, snap_r, exclude_level=None):
    pt = Point(pos)
    best = None
    for u, v, d in G.edges(data=True):
        if (u, v) in p_edges or (v, u) in p_edges: continue
        if exclude_level is not None and d.get("level", 0) == exclude_level:
            continue
        dist = pt.distance(d["geom"])
        if dist > snap_r: continue
        proj = np.asarray(d["geom"].interpolate(d["geom"].project(pt)).coords[0])
        if best is None or dist < best["d"]:
            best = {"u": u, "v": v, "pt": proj, "d": float(dist)}
    return best


# ── G1: growth loop ───────────────────────────────────────────────────


def grow_g1(G, tf, config, rng=None):
    """Individual front growth from H edges. Covers the full map."""
    if rng is None:
        rng = random.Random(config.random_seed)

    seeds = gen_seeds(G, config, rng)
    fronts = [{"nid": s["nid"], "pos": s["pos"].copy(), "dir": s["dir"].copy(),
                "trav": 0.0, "pe": s["pe"], "alive": True} for s in seeds]

    step_m = config.g1_step_m
    max_steps = config.g1_max_steps
    max_len = config.g1_max_length_m
    snap_r = config.snap_radius_m

    for _ in range(max_steps):
        active = [f for f in fronts if f["alive"] and f["trav"] < max_len]
        if not active:
            break

        new_fronts = []
        for fr in active:
            d = grow_dir(fr["pos"], fr["dir"], tf, config)
            np_pos = fr["pos"] + d * step_m

            if not (0 < np_pos[0] < config.map_width_m and
                    0 < np_pos[1] < config.map_height_m):
                fr["alive"] = False
                continue

            p_set = {fr["pe"], (fr["pe"][1], fr["pe"][0])}
            snap = find_snap(G, np_pos, p_set, snap_r, exclude_level=0)
            if snap is not None:
                sid = split_e(G, snap["u"], snap["v"], snap["pt"])
                if sid is not None and sid != fr["nid"]:
                    add_e(G, fr["nid"], sid, 1)
                    fr["alive"] = False
                    continue

            nid = _nid(G)
            G.add_node(nid, pos=np_pos.copy())
            add_e(G, fr["nid"], nid, 1)
            fr["nid"] = nid
            fr["pos"] = np_pos.copy()
            fr["dir"] = d.copy()
            fr["trav"] += step_m
            fr["pe"] = (fr["nid"], nid)

    return G


# ── A* endpoint closure ──────────────────────────────────────────────


def close_endpoints(G, road_field, config):
    """A* from each G1 degree-1 endpoint to the nearest road."""
    H, W = (128, 128) if road_field is None else road_field.shape[:2]
    cost = cost_map_from_road(road_field) if road_field is not None else None
    map_m = config.map_width_m

    # Find G1 endpoints
    deg1 = set()
    for u, v, d in G.edges(data=True):
        if d.get("level", 0) != 1: continue
        for n in (u, v):
            if G.degree(n) == 1:
                deg1.add(n)
    if not deg1: return G

    max_dist = config.snap_radius_m * 4.0
    max_edges = min(80, len(deg1))
    connected = set()

    # Build candidates sorted by distance
    cand = []
    for ep in sorted(deg1):
        if ep in connected: continue
        p_ep = G.nodes[ep]["pos"]
        p_set = set()
        for u, v in G.edges(ep):
            p_set.add((u, v)); p_set.add((v, u))

        best_d, best_t, best_cp = float('inf'), None, None
        for u, v, d in G.edges(data=True):
            if (u, v) in p_set or (v, u) in p_set: continue
            dist = Point(p_ep).distance(d["geom"])
            if dist < best_d:
                best_d = dist
                best_t = (u, v)
                cp = d["geom"].interpolate(d["geom"].project(Point(p_ep)))
                best_cp = np.asarray(cp.coords[0])
        if best_t is None or best_d > max_dist: continue
        cand.append((best_d, ep, best_t, best_cp))

    cand.sort(key=lambda x: x[0])

    for d_ab, ep, (tu, tv), cp in cand:
        if len(connected) >= max_edges: break
        if ep in connected: continue

        # A* if road_field available
        if cost is not None:
            src_n = G.nodes[ep]["pos"] / map_m
            tgt_n = cp / map_m
            path = astar_connect_path(src_n, tgt_n, road_field, cost, W, H, max_steps=8000)
            if path is not None and len(path) >= 3:
                prev = ep
                step_loc = config.g1_step_m / map_m
                cum = 0.0
                for k in range(1, len(path)):
                    ds = float(np.linalg.norm(path[k] - path[k-1]))
                    cum += ds
                    if cum >= step_loc or k == len(path) - 1:
                        if k == len(path) - 1:
                            sid = split_e(G, tu, tv, cp)
                            if sid is not None:
                                add_e(G, prev, sid, 1)
                        else:
                            nid = _nid(G)
                            G.add_node(nid, pos=(path[k] * map_m).copy())
                            add_e(G, prev, nid, 1)
                            prev = nid
                        cum = 0.0
                connected.add(ep)
                continue

        # Fallback: direct line
        cand_line = LineString([G.nodes[ep]["pos"], cp])
        illegal = any(cand_line.crosses(d["geom"])
                      for u, v, d in G.edges(data=True) if not ({u, v} & {ep}))
        if illegal: continue

        sid = split_e(G, tu, tv, cp)
        if sid is not None and sid != ep:
            add_e(G, ep, sid, 1)
            connected.add(ep)

    return G


# ── Merge ─────────────────────────────────────────────────────────────


def final_merge(G, config):
    """Snap G1 nodes onto H edges. Skip full-graph merge (preserves level)."""
    snap_r = config.snap_radius_m

    h_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("level", 0) == 0]
    g1_nodes = set()
    for u, v, d in G.edges(data=True):
        if d.get("level", 0) != 1: continue
        g1_nodes.add(u); g1_nodes.add(v)

    h_nodes = set()
    for u, v, d in G.edges(data=True):
        if d.get("level", 0) == 0:
            h_nodes.add(u); h_nodes.add(v)

    for g1n in sorted(g1_nodes):
        if g1n in h_nodes: continue
        pg = G.nodes[g1n]["pos"]
        best_d, best_info = float('inf'), None
        for hu, hv, hd in h_edges:
            dist = Point(pg).distance(hd["geom"])
            if dist < best_d:
                best_d = dist
                cp = hd["geom"].interpolate(hd["geom"].project(Point(pg)))
                best_info = (hu, hv, np.asarray(cp.coords[0]))
        if best_d < snap_r * 0.8 and best_info is not None:
            hu, hv, cp = best_info
            sid = split_e(G, hu, hv, cp)
            if sid is not None and sid != g1n:
                nbrs = list(G.neighbors(g1n))
                for nb in nbrs:
                    if G.has_edge(g1n, nb):
                        lv = G.edges[g1n, nb].get("level", 1)
                        G.remove_edge(g1n, nb)
                        if not G.has_edge(sid, nb):
                            add_e(G, sid, nb, lv)
                G.remove_node(g1n)

    return G


# ── G2: raster ────────────────────────────────────────────────────────


def _empty_regs(G, map_m, min_a, res=64):
    from scipy.ndimage import label, binary_dilation
    cm = map_m / res
    occ = np.zeros((res, res), dtype=bool)
    for _, _, d in G.edges(data=True):
        g = d["geom"]
        n = max(2, int(np.ceil(g.length / cm)))
        for k in range(n):
            pt = np.asarray(g.interpolate(k * g.length / max(n - 1, 1)).coords[0])
            cx = int(np.clip(pt[0] / cm, 0, res - 1))
            cy = int(np.clip(pt[1] / cm, 0, res - 1))
            occ[cy, cx] = True
    occ = binary_dilation(occ, iterations=2)
    lab, nl = label(~occ)
    regs = []
    for li in range(1, nl + 1):
        mask = lab == li
        a = float(mask.sum()) * cm ** 2
        if a < min_a: continue
        ys, xs = np.where(mask)
        regs.append((np.array([float(np.mean(ys)) * cm, float(np.mean(xs)) * cm]),
                     float(ys.max() - ys.min() + 1) * cm,
                     float(xs.max() - xs.min() + 1) * cm))
    regs.sort(key=lambda r: -r[1] * r[2])
    return regs


def grow_g2(G, tf, config):
    map_m = config.map_width_m
    for _ in range(config.g2_max_cuts_per_pass * 5):
        regs = _empty_regs(G, map_m, config.g2_min_face_area_m2)
        if not regs: break
        cut = False
        for c, w, h in regs[:3]:
            maj, min_, _ = tf.axes(c)
            d = min_ if w >= h else maj
            hf = min(w, h) * 0.3
            if hf < config.g2_min_cut_length_m * 0.5: continue
            p0, p1 = c - d * hf, c + d * hf
            e0 = _nearest(G, p0); e1 = _nearest(G, p1)
            n0 = split_e(G, e0[0], e0[1], p0)
            n1 = split_e(G, e1[0], e1[1], p1)
            if n0 is None: n0 = _nid(G); G.add_node(n0, pos=p0.copy())
            if n1 is None: n1 = _nid(G); G.add_node(n1, pos=p1.copy())
            if n0 == n1: continue
            mid = (G.nodes[n0]["pos"] + G.nodes[n1]["pos"]) / 2
            ok = all(Point(mid).distance(ed["geom"]) >= config.snap_radius_m * 0.5
                     for uu, vv, ed in G.edges(data=True) if not ({uu, vv} & {n0, n1}))
            if not ok: continue
            add_e(G, n0, n1, 2)
            cut = True; break
        if not cut: break
    return G


def _nearest(G, pt):
    p = Point(pt)
    best, bd = None, float('inf')
    for u, v, d in G.edges(data=True):
        dist = p.distance(d["geom"])
        if dist < bd: bd, best = dist, (u, v)
    return best


# ── main ──────────────────────────────────────────────────────────────


def grow(coords_m, ei, nt, road_field, config):
    G = to_graph(coords_m, ei, nt, level=0)
    tf = GraphTensorField.from_h_graph(coords_m, ei,
                                        step_m=config.tensor_sample_step_m,
                                        radius_m=config.tensor_radius_m,
                                        sigma_m=config.tensor_sigma_m)
    # G1: full coverage growth from H edges (seeds split H edges)
    G = grow_g1(G, tf, config)
    # A* endpoint closure
    if road_field is not None:
        G = close_endpoints(G, road_field, config)
    # G2: face infill on raw G1 topology
    G = grow_g2(G, tf, config)
    # Final cleanup: snap G1 nodes to H edges
    G = final_merge(G, config)
    return to_dict(G, config.map_width_m)

"""Road network growth and infill pipeline."""

from __future__ import annotations

import math
import random

import networkx as nx
import numpy as np
from scipy.ndimage import binary_dilation, label
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from network_generator.topology.graph_pathfinding import astar_connect_path, cost_map_from_road

from .tensor_field import GraphTensorField, normalize


def _next_node_id(G: nx.Graph) -> int:
    m = -1
    for n in G.nodes:
        if isinstance(n, int) and n > m:
            m = n
    return m + 1


# ── graph I/O ─────────────────────────────────────────────────────────


def to_graph(coords_m, ei, nt, level=0):
    """Build a NetworkX graph from coords and edges."""
    G = nx.Graph()
    for i, p in enumerate(coords_m):
        G.add_node(i, pos=p.copy())
    for u, v in ei:
        u, v = int(u), int(v)
        pu, pv = coords_m[u], coords_m[v]
        G.add_edge(
            u, v, geom=LineString([pu, pv]), length=float(np.linalg.norm(pv - pu)), level=level
        )
    return G


def to_dict(G, map_size_m):
    """Convert a NetworkX graph to array dict."""
    nodes = sorted(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    c = np.array([G.nodes[n]["pos"] for n in nodes], dtype=np.float32)
    e, road_class, edge_lengths = [], [], []
    deg = {n: 0 for n in nodes}
    for u, v, d in G.edges(data=True):
        e.append([idx[u], idx[v]])
        lv = int(d.get("level", 0))
        road_class.append(min(lv + 1, 3))
        edge_lengths.append(float(d.get("length", 1)))
        deg[u] += 1
        deg[v] += 1
    e = np.array(e, dtype=np.int64).reshape(-1, 2) if e else np.empty((0, 2), dtype=np.int64)
    nt = np.array([1 if deg[n] >= 3 else (4 if deg[n] <= 1 else 0) for n in nodes], dtype=np.int64)
    return {
        "coords": c,
        "edge_index": e,
        "edge_lengths_m": np.array(edge_lengths, dtype=np.float32),
        "road_class": np.array(road_class, dtype=np.int64),
        "node_types": nt,
        "map_size_m": map_size_m,
    }


def _add_edge(G, u, v, level):
    """Add an edge between two nodes if absent."""
    if G.has_edge(u, v):
        return
    pu, pv = G.nodes[u]["pos"], G.nodes[v]["pos"]
    G.add_edge(u, v, geom=LineString([pu, pv]), length=float(np.linalg.norm(pv - pu)), level=level)


def _split_edge_at_point(G, u, v, pt):
    """Split an edge at a point, returning the new node."""
    if not G.has_edge(u, v):
        return None
    d = G.edges[u, v]
    g = d["geom"]
    da = g.project(Point(pt))
    if da < 1e-6:
        return u
    if g.length - da < 1e-6:
        return v
    nid = _next_node_id(G)
    G.add_node(nid, pos=pt.copy())
    lv = d.get("level", 3)
    G.remove_edge(u, v)
    G.add_edge(
        u,
        nid,
        geom=LineString([np.asarray(g.coords[0]), pt]),
        length=float(np.linalg.norm(np.asarray(g.coords[0]) - pt)),
        level=lv,
    )
    G.add_edge(
        nid,
        v,
        geom=LineString([pt, np.asarray(g.coords[-1])]),
        length=float(np.linalg.norm(pt - np.asarray(g.coords[-1]))),
        level=lv,
    )
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
        cur_u, cur_v = u, v  # remaining H sub-edge to split at each seed
        for t in t_vals:
            pt = pu + t * ev
            # The first split removes edge (u, v); later t's lie on the
            # remaining sub-edge (cur_u, v), so split that instead.
            nid = _split_edge_at_point(G, cur_u, cur_v, pt)
            if nid is None or nid == cur_u or nid == cur_v:
                continue
            for sign in (-1.0, 1.0):
                direction = normalize(
                    sign * normal + rng.uniform(-config.g1_seed_jitter, config.g1_seed_jitter) * ed
                )
                seeds.append({"nid": nid, "pos": pt.copy(), "dir": direction, "pe": (cur_u, nid)})
            cur_u = nid  # remaining edge is now (nid, v)
    return seeds


# ── G1: growth direction ─────────────────────────────────────────────


def grow_dir(pos, prev, tf, config, rng=None):
    """Growth direction with per-step style jitter."""
    tensor_dir = tf.choose_direction(pos, prev)
    r = config.w_tensor * tensor_dir + 0.30 * prev
    r = normalize(r)
    if np.linalg.norm(r) < 1e-9:
        return prev.copy()
    dv = np.clip(np.dot(r, prev), -1.0, 1.0)
    ang = math.degrees(math.acos(dv))
    if ang > config.max_turn_deg:
        ratio = config.max_turn_deg / max(ang, 1e-6)
        r = normalize((1.0 - ratio) * prev + ratio * r)
    # Per-step jitter: small direction perturbation each step
    if config.per_step_jitter_deg > 0.5 and rng is not None:
        j = math.radians(config.per_step_jitter_deg * (rng.random() * 2.0 - 1.0))
        ca, sa = math.cos(j), math.sin(j)
        rx, ry = r[0], r[1]
        r = normalize(np.array([ca * rx - sa * ry, sa * rx + ca * ry]))
    return r


# ── G1: snap check ────────────────────────────────────────────────────


def find_snap(G, pos, p_edges, snap_r, exclude_level=None, tree=None, edge_pairs=None):
    """Find the nearest growth edge within ``snap_r`` of ``pos``.

    When ``tree`` / ``edge_pairs`` are provided (built once per G1 step), the
    edge set is pruned with a spatial index so only geometrically near edges
    are tested exactly.  Edges that were split/removed since the index was
    built are skipped via ``G.has_edge``.
    """
    pt = Point(pos)
    best = None

    if tree is not None and edge_pairs is not None:
        for idx in tree.query(pt.buffer(snap_r)):
            u, v = edge_pairs[idx]
            if (u, v) in p_edges or (v, u) in p_edges:
                continue
            if not G.has_edge(u, v):  # edge split since the index was built
                continue
            d = G[u][v]
            if exclude_level is not None and d.get("level", 0) == exclude_level:
                continue
            dist = pt.distance(d["geom"])
            if dist > snap_r:
                continue
            proj = np.asarray(d["geom"].interpolate(d["geom"].project(pt)).coords[0])
            if best is None or dist < best["d"]:
                best = {"u": u, "v": v, "pt": proj, "d": float(dist)}
        return best

    # Fallback: full scan (used when no index is supplied).
    for u, v, d in G.edges(data=True):
        if (u, v) in p_edges or (v, u) in p_edges:
            continue
        if exclude_level is not None and d.get("level", 0) == exclude_level:
            continue
        dist = pt.distance(d["geom"])
        if dist > snap_r:
            continue
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
    fronts = [
        {
            "nid": s["nid"],
            "pos": s["pos"].copy(),
            "dir": s["dir"].copy(),
            "trav": 0.0,
            "pe": s["pe"],
            "alive": True,
        }
        for s in seeds
    ]

    step_m = config.g1_step_m
    max_steps = config.g1_max_steps
    max_len = config.g1_max_length_m
    snap_r = config.snap_radius_m * config.snap_radius_scale

    for _ in range(max_steps):
        active = [f for f in fronts if f["alive"] and f["trav"] < max_len]
        if not active:
            break

        # Spatial index over current growth edges (level 1), rebuilt each step
        # because fronts add/split edges as they advance.
        edge_pairs = []
        edge_lines = []
        for u, v, d in G.edges(data=True):
            if d.get("level", 0) == 0:
                continue
            edge_pairs.append((u, v))
            edge_lines.append(d["geom"])
        snap_tree = STRtree(edge_lines) if edge_lines else None

        new_fronts = []
        for fr in active:
            d = grow_dir(fr["pos"], fr["dir"], tf, config, rng)
            np_pos = fr["pos"] + d * step_m

            if not (0 < np_pos[0] < config.map_width_m and 0 < np_pos[1] < config.map_height_m):
                fr["alive"] = False
                continue

            p_set = {fr["pe"], (fr["pe"][1], fr["pe"][0])}
            snap = find_snap(
                G, np_pos, p_set, snap_r, exclude_level=0, tree=snap_tree, edge_pairs=edge_pairs
            )
            if snap is not None:
                sid = _split_edge_at_point(G, snap["u"], snap["v"], snap["pt"])
                if sid is not None and sid != fr["nid"]:
                    _add_edge(G, fr["nid"], sid, 1)
                    fr["alive"] = False
                    continue

            nid = _next_node_id(G)
            G.add_node(nid, pos=np_pos.copy())
            _add_edge(G, fr["nid"], nid, 1)
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
    map_size_m = config.map_width_m

    # Find G1 endpoints
    deg1 = set()
    for u, v, d in G.edges(data=True):
        if d.get("level", 0) != 1:
            continue
        for n in (u, v):
            if G.degree(n) == 1:
                deg1.add(n)
    if not deg1:
        return G

    max_dist = config.snap_radius_m * 4.0
    max_edges = min(80, len(deg1))
    connected = set()

    # Build candidates sorted by distance (spatial index prunes the per-endpoint
    # full-edge scan; phase-1 tree is static, phase-2 A* mutations happen after).
    all_edges = list(G.edges(data=True))
    edge_tree = STRtree([d["geom"] for _, _, d in all_edges]) if all_edges else None

    cand = []
    for ep in sorted(deg1):
        if ep in connected:
            continue
        p_ep = G.nodes[ep]["pos"]
        p_set = set()
        for u, v in G.edges(ep):
            p_set.add((u, v))
            p_set.add((v, u))

        best_d, best_t, best_cp = float("inf"), None, None
        if edge_tree is not None:
            for idx in edge_tree.query(Point(p_ep).buffer(max_dist)):
                u, v, d = all_edges[idx]
                if (u, v) in p_set or (v, u) in p_set:
                    continue
                dist = Point(p_ep).distance(d["geom"])
                if dist < best_d:
                    best_d = dist
                    best_t = (u, v)
                    cp = d["geom"].interpolate(d["geom"].project(Point(p_ep)))
                    best_cp = np.asarray(cp.coords[0])
        if best_t is None or best_d > max_dist:
            continue
        cand.append((best_d, ep, best_t, best_cp))

    cand.sort(key=lambda x: x[0])

    for d_ab, ep, (tu, tv), cp in cand:
        if len(connected) >= max_edges:
            break
        if ep in connected:
            continue

        # A* if road_field available
        if cost is not None:
            src_n = G.nodes[ep]["pos"] / map_size_m
            tgt_n = cp / map_size_m
            path = astar_connect_path(src_n, tgt_n, road_field, cost, W, H, max_steps=8000)
            if path is not None and len(path) >= 3:
                prev = ep
                step_loc = config.g1_step_m / map_size_m
                cum = 0.0
                for k in range(1, len(path)):
                    ds = float(np.linalg.norm(path[k] - path[k - 1]))
                    cum += ds
                    if cum >= step_loc or k == len(path) - 1:
                        if k == len(path) - 1:
                            sid = _split_edge_at_point(G, tu, tv, cp)
                            if sid is not None:
                                _add_edge(G, prev, sid, 1)
                        else:
                            nid = _next_node_id(G)
                            G.add_node(nid, pos=(path[k] * map_size_m).copy())
                            _add_edge(G, prev, nid, 1)
                            prev = nid
                        cum = 0.0
                connected.add(ep)
                continue

        # Fallback: direct line
        cand_line = LineString([G.nodes[ep]["pos"], cp])
        illegal = any(
            cand_line.crosses(d["geom"]) for u, v, d in G.edges(data=True) if not ({u, v} & {ep})
        )
        if illegal:
            continue

        sid = _split_edge_at_point(G, tu, tv, cp)
        if sid is not None and sid != ep:
            _add_edge(G, ep, sid, 1)
            connected.add(ep)

    return G


# ── Merge ─────────────────────────────────────────────────────────────


def final_merge(G, config):
    """Snap G1 nodes onto H edges. Skip full-graph merge (preserves level)."""
    snap_r = config.snap_radius_m

    h_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("level", 0) == 0]
    g1_nodes = set()
    for u, v, d in G.edges(data=True):
        if d.get("level", 0) != 1:
            continue
        g1_nodes.add(u)
        g1_nodes.add(v)

    h_nodes = set()
    for u, v, d in G.edges(data=True):
        if d.get("level", 0) == 0:
            h_nodes.add(u)
            h_nodes.add(v)

    # Spatial index over H edges: only test edges whose geometry passes within
    # the snap threshold of the node (avoids O(g1 × h_edges) Shapely scans).
    h_tree = STRtree([d["geom"] for _, _, d in h_edges]) if h_edges else None

    for g1n in sorted(g1_nodes):
        if g1n in h_nodes:
            continue
        pg = G.nodes[g1n]["pos"]
        best_d, best_info = float("inf"), None
        if h_tree is not None:
            for idx in h_tree.query(Point(pg).buffer(snap_r * 0.8)):
                hu, hv, hd = h_edges[idx]
                dist = Point(pg).distance(hd["geom"])
                if dist < best_d:
                    best_d = dist
                    cp = hd["geom"].interpolate(hd["geom"].project(Point(pg)))
                    best_info = (hu, hv, np.asarray(cp.coords[0]))
        if best_d < snap_r * 0.8 and best_info is not None:
            hu, hv, cp = best_info
            sid = _split_edge_at_point(G, hu, hv, cp)
            if sid is not None and sid != g1n:
                nbrs = list(G.neighbors(g1n))
                for nb in nbrs:
                    if G.has_edge(g1n, nb):
                        lv = G.edges[g1n, nb].get("level", 1)
                        G.remove_edge(g1n, nb)
                        if not G.has_edge(sid, nb):
                            _add_edge(G, sid, nb, lv)
                G.remove_node(g1n)

    return G


# ── G2: raster ────────────────────────────────────────────────────────


def _find_empty_regions(G, map_size_m, min_area, res=64):
    cell_size_m = map_size_m / res
    occ = np.zeros((res, res), dtype=bool)
    for _, _, d in G.edges(data=True):
        g = d["geom"]
        n = max(2, int(np.ceil(g.length / cell_size_m)))
        for k in range(n):
            pt = np.asarray(g.interpolate(k * g.length / max(n - 1, 1)).coords[0])
            cx = int(np.clip(pt[0] / cell_size_m, 0, res - 1))
            cy = int(np.clip(pt[1] / cell_size_m, 0, res - 1))
            occ[cy, cx] = True
    occ = binary_dilation(occ, iterations=2)
    lab, nl = label(~occ)
    regs = []
    for li in range(1, nl + 1):
        mask = lab == li
        a = float(mask.sum()) * cell_size_m**2
        if a < min_area:
            continue
        ys, xs = np.where(mask)
        regs.append(
            (
                np.array([float(np.mean(ys)) * cell_size_m, float(np.mean(xs)) * cell_size_m]),
                float(ys.max() - ys.min() + 1) * cell_size_m,
                float(xs.max() - xs.min() + 1) * cell_size_m,
            )
        )
    regs.sort(key=lambda r: -r[1] * r[2])
    return regs


def grow_g2(G, tf, config, rng=None):
    """Fill empty regions with G2 infill roads."""
    map_size_m = config.map_width_m
    if rng is None:
        rng = random.Random(config.random_seed + 1)
    for _ in range(config.g2_max_cuts_per_pass * 5):
        regs = _find_empty_regions(G, map_size_m, config.g2_min_face_area_m2)
        if not regs:
            break
        cut = False
        for c, w, h in regs[:3]:
            maj, min_, _ = tf.axes(c)
            d = min_ if w >= h else maj
            # Style-aware jitter: organic → random cut direction
            if config.g2_jitter_deg > 0.5:
                j = math.radians(config.g2_jitter_deg * (rng.random() * 2.0 - 1.0))
                ca, sa = math.cos(j), math.sin(j)
                d = normalize(np.array([ca * d[0] - sa * d[1], sa * d[0] + ca * d[1]]))
            hf = min(w, h) * 0.3
            if hf < config.g2_min_cut_length_m * 0.5:
                continue
            p0, p1 = c - d * hf, c + d * hf
            e0 = _nearest_edge(G, p0)
            e1 = _nearest_edge(G, p1)
            n0 = _split_edge_at_point(G, e0[0], e0[1], p0)
            n1 = _split_edge_at_point(G, e1[0], e1[1], p1)
            if n0 is None:
                n0 = _next_node_id(G)
                G.add_node(n0, pos=p0.copy())
            if n1 is None:
                n1 = _next_node_id(G)
                G.add_node(n1, pos=p1.copy())
            if n0 == n1:
                continue
            mid = (G.nodes[n0]["pos"] + G.nodes[n1]["pos"]) / 2
            ok = all(
                Point(mid).distance(ed["geom"]) >= config.snap_radius_m * 0.5
                for uu, vv, ed in G.edges(data=True)
                if not ({uu, vv} & {n0, n1})
            )
            if not ok:
                continue
            _add_edge(G, n0, n1, 2)
            cut = True
            break
        if not cut:
            break
    return G


def _nearest_edge(G, pt):
    p = Point(pt)
    best, bd = None, float("inf")
    for u, v, d in G.edges(data=True):
        dist = p.distance(d["geom"])
        if dist < bd:
            bd, best = dist, (u, v)
    return best


# ── main ──────────────────────────────────────────────────────────────


def grow(coords_m, ei, nt, road_field, config):
    """Grow a road network from an H-graph."""
    G = to_graph(coords_m, ei, nt, level=0)
    tf = GraphTensorField.from_h_graph(
        coords_m,
        ei,
        step_m=config.tensor_sample_step_m,
        radius_m=config.tensor_radius_m,
        sigma_m=config.tensor_sigma_m,
        tangent_smooth_radius=config.tensor_smooth_radius_m,
    )
    rng = random.Random(config.random_seed)
    # G1: full coverage growth from H edges (seeds split H edges)
    G = grow_g1(G, tf, config, rng)
    # A* endpoint closure
    if road_field is not None:
        G = close_endpoints(G, road_field, config)
    # G2: face infill on raw G1 topology
    G = grow_g2(G, tf, config, rng)
    # Final cleanup: snap G1 nodes to H edges
    G = final_merge(G, config)
    return to_dict(G, config.map_width_m)

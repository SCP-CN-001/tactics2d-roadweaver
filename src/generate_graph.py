"""
RoadWeaver graph generation pipeline — two-phase design.

Phase 1 — ``generate_skeleton``:  VQ + Transformer → raw skeleton → clean → scale
Phase 2 — ``generate_branch``:     Growth → cleanup → compress → classify → lanes

Usage::

    from generate_graph import generate_skeleton, generate_branch

    skeleton = generate_skeleton(gen, cond, struct, map_w=2000, map_h=2000)
    result   = generate_branch(*skeleton, cond, map_w=2000, map_h=2000)
    # result == {"coords_int", "edge_index_int", "node_types", "lanes", "geoms", "road_class"}
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

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
from network_generator.topology.graph_ops import (
    classify_nodes,
    compress_to_intersection_graph,
    merge_close_nodes,
    simplify_chains,
)

# ── Growth-graph crossing fix ────────────────────────────────────


def _segment_intersection(a, b, c, d):
    """Return intersection point of segments ab and cd, or ``None``."""
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    x4, y4 = d
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    if 0 < t < 1 and 0 < u < 1:
        return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])
    return None


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


# ── Abnormal edge fix ────────────────────────────────────────────


def fix_abnormal_edges(coords, edge_index, geometries, map_max):
    """Detect and repair self-intersecting or overly-curved edge geometries.

    Modifies *geometries* in-place by simplifying self-intersecting polylines
    and capping curvature via Douglas-Peucker simplification.
    """
    from shapely import simplify as shapely_simplify
    from shapely.geometry import LineString

    for j in range(len(edge_index)):
        if j >= len(geometries) or len(geometries[j]) < 3:
            continue
        geom = np.asarray(geometries[j])
        ls = LineString(geom)
        if not ls.is_simple:
            # Self-intersecting — simplify with DP tolerance
            simplified = shapely_simplify(ls, tolerance=0.002 * map_max / 2000.0)
            if simplified.geom_type == "LineString" and len(simplified.coords) >= 2:
                geometries[j] = np.array(simplified.coords)
            else:
                # Fallback: straight line between endpoints
                geometries[j] = np.array([geom[0], geom[-1]])


# ── Compressed graph merge ─────────────────────────────────────────


def merge_compressed_graph(
    coords: np.ndarray,
    edge_index: np.ndarray,
    geometries: list[np.ndarray],
    map_max: float,
    merge_dist_m: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Merge compressed-graph nodes closer than *merge_dist_m*."""
    if len(coords) < 2:
        return coords, edge_index, geometries

    dist_norm = merge_dist_m / map_max
    N = len(coords)
    import networkx as nx

    G = nx.Graph()
    for i in range(N):
        G.add_node(i)
    for u, v in edge_index:
        G.add_edge(int(u), int(v))

    # Find merge pairs — merge lower-index into higher-index nodes.
    merge_map: dict[int, int] = {}
    for i in range(N):
        if i in merge_map:
            continue
        for j in range(i + 1, N):
            if j in merge_map:
                continue
            if np.linalg.norm(coords[i] - coords[j]) < dist_norm:
                merge_map[j] = i

    if not merge_map:
        return coords, edge_index, geometries

    # Apply merges: translate node indices
    new_coords = []
    new_idx = {}
    old2new = {}
    for n in range(N):
        target = n
        while target in merge_map:
            target = merge_map[target]
        old2new[n] = target

    # Rebuild edge index
    new_edges_set: set[tuple[int, int]] = set()
    geom_map: dict[tuple[int, int], np.ndarray] = {}  # (u,v) → geometry
    for e, (u, v) in enumerate(edge_index):
        nu, nv = old2new[int(u)], old2new[int(v)]
        if nu == nv:  # self-loop after merge
            continue
        key = (nu, nv) if nu < nv else (nv, nu)
        if key in new_edges_set:
            continue  # duplicate parallel edge
        new_edges_set.add(key)
        geom = geometries[e] if e < len(geometries) else np.array([coords[u], coords[v]])
        geom_map[key] = geom

    # Build output — use original indices for geom_map lookup, then remap
    uniq = sorted(set(old2new.values()))
    idx_of = {n: i for i, n in enumerate(uniq)}
    c2 = np.array([coords[n] for n in uniq])
    e2 = []
    g2 = []
    for u, v in sorted(new_edges_set):  # sorted for determinism
        e2.append([idx_of[u], idx_of[v]])
        key = (u, v) if u < v else (v, u)
        g2.append(geom_map.get(key, np.array([coords[u], coords[v]])))
    e2 = np.array(e2, dtype=np.int64)

    merged_count = N - len(c2)
    print(
        f"[Merge] {N}→{len(c2)} nodes ({merged_count} merged), "
        f"{len(edge_index)}→{len(e2)} edges"
    )
    return c2, e2, g2


# ── Merge nearby junction nodes (physical graph merge) ───────────


def merge_nearby_junctions(coords, edge_index, node_types, geometries, map_max, merge_dist_m=50.0):
    """Physically merge junction nodes closer than *merge_dist_m*.

    Unlike classify_nodes which only conceptually groups them, this
    function rewires the graph so that nearby junctions become a
    single node with combined incident edges.
    """
    if len(coords) < 2:
        return coords, edge_index, node_types, geometries

    norm = merge_dist_m / map_max
    junctions = [i for i, t in enumerate(node_types) if int(t) == 1]
    if len(junctions) < 2:
        return coords, edge_index, node_types, geometries

    # Build adjacency
    adj = {i: set() for i in range(len(coords))}
    for u, v in edge_index:
        adj[int(u)].add(int(v))
        adj[int(v)].add(int(u))

    # Group nearby junctions
    merged_junc = {i: i for i in range(len(coords))}
    for i in range(len(junctions)):
        ni = junctions[i]
        for j in range(i + 1, len(junctions)):
            nj = junctions[j]
            if np.linalg.norm(coords[ni] - coords[nj]) < norm:
                # Merge nj into ni (keep lower index)
                merged_junc[nj] = ni

    changes = sum(1 for i, m in merged_junc.items() if i != m)
    if not changes:
        return coords, edge_index, node_types, geometries

    # Rebuild: redirect edges, remove merged nodes
    keep = set()
    for u, v in edge_index:
        nu = merged_junc.get(int(u), int(u))
        nv = merged_junc.get(int(v), int(v))
        if nu != nv:
            keep.add((nu, nv) if nu < nv else (nv, nu))

    orphaned = sorted(set(range(len(coords))) - {v for uv in keep for v in uv})
    if not orphaned:
        return coords, edge_index, node_types, geometries

    old2new = {}
    new_coords = []
    for n in range(len(coords)):
        target = merged_junc.get(n, n)
        if target not in old2new:
            old2new[target] = len(new_coords)
            new_coords.append(coords[target])

    new_ei = np.array([[old2new[u], old2new[v]] for u, v in keep], dtype=np.int64)
    new_nt = np.array([1 if n in old2new else 0 for n in range(len(coords))], dtype=np.int64)[
        list(old2new.keys())
    ]
    # Geometry: rebuild per new edge
    ei_map = {(int(u), int(v)): j for j, (u, v) in enumerate(edge_index)}
    new_geoms = []
    for u, v in keep:
        key = (u, v) if u < v else (v, u)
        j = ei_map.get(key, -1)
        if j >= 0 and j < len(geometries) and len(geometries[j]) >= 2:
            new_geoms.append(geometries[j])
        else:
            new_geoms.append(np.array([new_coords[old2new[u]], new_coords[old2new[v]]]))

    print(
        f"[MergeJunc] {len(coords)}->{len(new_coords)} nodes, "
        f"{len(edge_index)}->{len(new_ei)} edges"
    )
    return np.array(new_coords), new_ei, new_nt, new_geoms


# ── Split degree>=5 junctions ────────────────────────────────────


def split_high_degree_junctions(
    coords, edge_index, node_types, geometries, map_max, split_radius_m=3.0
):
    """Split junction nodes with degree >= 5 into multiple degree-3/4 sub-nodes.

    The tactics2d Intersection builder only supports 3 or 4 arms.
    High-degree junctions are subdivided into 2-3 sub-junctions
    connected by very short internal edges.
    """
    adj = {i: set() for i in range(len(coords))}
    for u, v in edge_index:
        adj[int(u)].add(int(v))
        adj[int(v)].add(int(u))

    split_rad = split_radius_m / map_max
    new_coords = list(coords)
    new_ei = list(edge_index)
    new_geoms = list(geometries)
    new_nt = list(node_types)

    # Loop until no degree>=5 junctions remain (splitting one can create another)
    for _ in range(10):
        adj = {i: set() for i in range(len(new_coords))}
        for u, v in new_ei:
            adj[int(u)].add(int(v))
            adj[int(v)].add(int(u))
        high_deg = [
            n
            for n in range(len(new_coords))
            if n < len(new_nt) and int(new_nt[n]) == 1 and len(adj[n]) >= 5
        ]
        if not high_deg:
            break

        for n in sorted(high_deg, reverse=True):
            nbs = list(adj[n])
            angles = [
                (
                    float(
                        np.arctan2(
                            new_coords[nb][1] - new_coords[n][1],
                            new_coords[nb][0] - new_coords[n][0],
                        )
                    ),
                    nb,
                )
                for nb in nbs
            ]
            angles.sort()
            n_edges = len(angles)

            # Group edges: 3 per group, last may have 4
            groups, i = [], 0
            while i < n_edges:
                end = min(i + 3, n_edges)
                if end - i < 2 and i > 0:
                    groups[-1].extend(angles[i:end])
                else:
                    groups.append(angles[i:end])
                i = end
            if len(groups) <= 1:
                continue

            sub_nodes = []
            for g_idx in range(1, len(groups)):
                dir_mean = np.mean([a for a, _ in groups[g_idx]], axis=0)
                offset = split_rad * np.array([np.cos(dir_mean), np.sin(dir_mean)])
                sub_idx = len(new_coords)
                new_coords.append(new_coords[n] + offset)
                new_nt.append(1)
                sub_nodes.append(sub_idx)

            # Rewire edges from n to sub-nodes
            for g_idx, sub_idx in enumerate(sub_nodes):
                for _, nb in groups[g_idx + 1]:
                    for e_idx in range(len(new_ei)):
                        u, v = int(new_ei[e_idx][0]), int(new_ei[e_idx][1])
                        if (u == n and v == nb) or (u == nb and v == n):
                            if u == n:
                                new_ei[e_idx] = [sub_idx, nb]
                            else:
                                new_ei[e_idx] = [nb, sub_idx]
                # Connect sub-node back to original junction
                new_ei.append([n, sub_idx])
                new_geoms.append(np.array([new_coords[n], new_coords[sub_idx]]))

            print(f"  [SplitJunc] Node {n} ({n_edges} arms) -> " f"{len(groups)} sub-junctions")

    return (
        np.array(new_coords),
        np.array(new_ei, dtype=np.int64),
        np.array(new_nt, dtype=np.int64),
        new_geoms,
    )


# ── Merge inter-edge parallel roads ──────────────────────────────


def merge_parallel_edges(coords, edge_index, geometries, map_max, angle_deg=30.0, dist_m=40.0):
    """Remove edges that are geometrically parallel and close.

    Unlike clean_parallel_roads which only checks edges sharing
    a node, this function checks ALL pairs of non-adjacent edges.
    """
    if len(edge_index) < 3:
        return coords, edge_index, geometries

    from shapely.geometry import LineString

    adj = {i: set() for i in range(len(coords))}
    for u, v in edge_index:
        adj[int(u)].add(int(v))
        adj[int(v)].add(int(u))

    angle_rad = np.radians(angle_deg)
    dist_norm = dist_m / map_max

    to_remove = set()
    for i in range(len(edge_index)):
        u1, v1 = int(edge_index[i, 0]), int(edge_index[i, 1])
        if i in to_remove:
            continue
        # Direction of edge i
        p1 = coords[u1]
        p2 = coords[v1]
        d1 = p2 - p1
        n1 = np.linalg.norm(d1)
        if n1 < 1e-12:
            continue
        d1 /= n1

        for j in range(i + 1, len(edge_index)):
            if j in to_remove:
                continue
            u2, v2 = int(edge_index[j, 0]), int(edge_index[j, 1])
            # Skip if they share a node (already handled by clean_parallel_roads)
            if u1 in (u2, v2) or v1 in (u2, v2):
                continue
            # Skip if nodes are adjacent
            if adj[u1] & {u2, v2} or adj[v1] & {u2, v2}:
                continue

            # Angular check
            p3 = coords[u2]
            p4 = coords[v2]
            d2 = p4 - p3
            n2 = np.linalg.norm(d2)
            if n2 < 1e-12:
                continue
            d2 /= n2
            dot = float(d1 @ d2)
            angle = abs(np.arccos(np.clip(dot, -1, 1)))
            if angle > angle_rad and abs(angle - np.pi) > angle_rad:
                continue

            # Distance check: Hausdorff-like min distance between endpoints
            ends1 = [coords[u1], coords[v1]]
            ends2 = [coords[u2], coords[v2]]
            min_dist = min(
                np.linalg.norm(coords[u1] - coords[u2]),
                np.linalg.norm(coords[u1] - coords[v2]),
                np.linalg.norm(coords[v1] - coords[u2]),
                np.linalg.norm(coords[v1] - coords[v2]),
            )
            if min_dist > dist_norm:
                continue

            # Remove the longer edge
            n1 = float(np.linalg.norm(p2 - p1))
            n2 = float(np.linalg.norm(p4 - p3))
            to_remove.add(i if n1 >= n2 else j)

    if not to_remove:
        return coords, edge_index, geometries

    keep_idx = [j for j in range(len(edge_index)) if j not in to_remove]
    keep_ei = edge_index[keep_idx]
    keep_geoms = [geometries[j] for j in keep_idx]

    # Remove orphaned nodes
    G = _build_nx(coords, keep_ei)
    orphaned = [n for n in range(len(coords)) if G.degree(n) == 0]
    if orphaned:
        c = np.delete(coords, orphaned, axis=0)
        keep_ei = np.array(
            [
                [u - sum(1 for o in orphaned if o < u), v - sum(1 for o in orphaned if o < v)]
                for u, v in keep_ei
            ],
            dtype=np.int64,
        )
        return c, keep_ei, keep_geoms
    return coords, keep_ei, keep_geoms


def _build_nx(coords, edge_index):
    """Build a networkx graph from coords + edge_index."""
    import networkx as nx

    G = nx.Graph()
    for i, p in enumerate(coords):
        G.add_node(i, pos=p.copy())
    for u, v in edge_index:
        G.add_edge(int(u), int(v))
    return G


# ── Snap edges to nearby nodes ────────────────────────────────────


def snap_edges_to_nodes(coords, edge_index, geometries, map_max, snap_dist_m=8.0):
    """Split compressed edges where they pass close to non-adjacent nodes.

    When an edge geometry passes within *snap_dist_m* of a node that is
    *not* one of its endpoints, the edge is subdivided at the closest
    point and a new connecting edge is inserted, creating an intersection.
    """
    from shapely.geometry import LineString, Point
    from shapely.ops import nearest_points

    adj: dict[int, set[int]] = {i: set() for i in range(len(coords))}
    for u, v in edge_index:
        adj[int(u)].add(int(v))
        adj[int(v)].add(int(u))

    split_edges: dict[int, list[tuple[float, int, np.ndarray]]] = {}
    # (edge_idx -> list of (frac_on_line, new_node_idx, split_position))
    new_nodes: list[np.ndarray] = []
    snap_norm = snap_dist_m / map_max

    for j in range(len(edge_index)):
        u, v = int(edge_index[j, 0]), int(edge_index[j, 1])
        if j < len(geometries) and len(geometries[j]) >= 3:
            pts = np.asarray(geometries[j])
        else:
            pts = np.array([coords[u], coords[v]])
        ls = LineString(pts)
        length = ls.length
        if length < 1e-12:
            continue

        for ni in range(len(coords)):
            if ni in (u, v) or ni in adj[u] or ni in adj[v]:
                continue
            node_pt = Point(coords[ni])
            dist = ls.distance(node_pt)
            if dist > snap_norm * 2:
                continue
            # Found a near-miss — project node onto edge
            proj = ls.project(node_pt)
            frac = proj / length
            if frac < 0.05 or frac > 0.95:
                continue  # too close to endpoint — would create degenerate edge
            near_pt = ls.interpolate(proj)
            new_idx = len(coords) + len(new_nodes)
            split_edges.setdefault(j, []).append((frac, new_idx, np.array([near_pt.x, near_pt.y])))
            new_nodes.append(np.array([near_pt.x, near_pt.y]))
            # Connect the new node to the nearby existing node
            split_edges.setdefault(-ni, []).append(  # sentinel: -ni-1 is the node-to-connect
                (0.0, new_idx, coords[ni])
            )

    if not split_edges:
        return coords, edge_index, geometries

    # Rebuild coords
    new_coords = np.vstack([coords] + new_nodes) if new_nodes else coords
    new_ei = []
    new_geoms = []

    for j in range(len(edge_index)):
        splits = sorted(split_edges.get(j, []), key=lambda x: x[0])
        u, v = int(edge_index[j, 0]), int(edge_index[j, 1])
        pts = (
            np.asarray(geometries[j])
            if j < len(geometries) and len(geometries[j]) >= 2
            else np.array([coords[u], coords[v]])
        )
        if not splits:
            new_ei.append([u, v])
            new_geoms.append(pts)
            continue
        # Split cumulative lengths
        seg_len = np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(axis=1))
        cum = np.concatenate([[0], seg_len.cumsum()])
        total = float(cum[-1]) if cum[-1] > 1e-12 else 1.0

        prev_idx = u
        prev_cum = 0.0
        for frac, ni, pos in splits:
            target_cum = frac * total
            # Collect intermediate points between prev_cum and target_cum
            sub = [pts[0] * 1]  # placeholder, will rebuild
            sub_pts = []
            for k in range(1, len(pts)):
                if cum[k - 1] >= target_cum:
                    break
                if cum[k] <= prev_cum:
                    continue
                # Point at distance d along this segment
                seg_start = max(prev_cum, cum[k - 1])
                seg_end = min(target_cum, cum[k])
                if seg_end <= seg_start:
                    continue
                t = (seg_start - cum[k - 1]) / (cum[k] - cum[k - 1] + 1e-12)
                sub_pts.append(pts[k - 1] + t * (pts[k] - pts[k - 1]))
            if not sub_pts:
                sub_pts.append(new_coords[prev_idx])
            sub_pts.append(pos)
            new_ei.append([prev_idx, ni])
            new_geoms.append(np.array(sub_pts))
            prev_idx = ni
            prev_cum = target_cum
        # Remaining segment to v
        sub_pts = [new_coords[prev_idx]]
        for k in range(1, len(pts)):
            if cum[k] <= prev_cum:
                continue
            t = (max(prev_cum, cum[k - 1]) - cum[k - 1]) / (cum[k] - cum[k - 1] + 1e-12)
            sub_pts.append(pts[k - 1] + t * (pts[k] - pts[k - 1]))
        if len(sub_pts) >= 2:
            new_ei.append([prev_idx, v])
            new_geoms.append(np.array(sub_pts))

    # Add connections from new nodes to their nearby existing nodes
    for key, data_list in split_edges.items():
        if key >= 0:
            continue
        for _, new_idx, _ in data_list:
            nearby_idx = int(-key - 1)
            new_ei.append([new_idx, nearby_idx])
            new_geoms.append(np.array([new_coords[new_idx], new_coords[nearby_idx]]))

    # Deduplicate edges
    seen = set()
    dedup_ei, dedup_geoms = [], []
    for e, g in zip(new_ei, new_geoms):
        key = (int(e[0]), int(e[1])) if e[0] < e[1] else (int(e[1]), int(e[0]))
        if key not in seen:
            seen.add(key)
            dedup_ei.append([key[0], key[1]])
            dedup_geoms.append(g)

    print(
        f"[Snap] {len(new_nodes)} nodes added, {len(new_ei)} edges "
        f"({len(new_ei) - len(dedup_ei)} deduped)"
    )
    return new_coords, np.array(dedup_ei, dtype=np.int64), dedup_geoms


# ── Geometry-node alignment ──────────────────────────────────────


def align_geometries_to_nodes(coords, edge_index, geometries):
    """Force geometry endpoints to match node positions.

    After structural changes (merge_nearby_junctions, split_high_degree_junctions,
    snap_edges_to_nodes), edge geometries can become misaligned: a geometry
    might start or end at a position that no longer matches the node coordinate
    (e.g. after a junction node was merged into another).  This function
    corrects the first and last point of every geometry so they exactly match
    the source/target node positions.  Intermediate points are left untouched.
    """
    changes = 0
    for j in range(len(edge_index)):
        u, v = int(edge_index[j, 0]), int(edge_index[j, 1])
        if j >= len(geometries) or len(geometries[j]) < 2:
            if j < len(geometries):
                geometries[j] = np.array([coords[u], coords[v]])
            continue
        geom = geometries[j]
        if isinstance(geom, list):
            geom = np.asarray(geom)
            geometries[j] = geom
        need = False
        if tuple(geom[0].round(10)) != tuple(coords[u].round(10)):
            geom[0] = coords[u].copy()
            need = True
        if tuple(geom[-1].round(10)) != tuple(coords[v].round(10)):
            geom[-1] = coords[v].copy()
            need = True
        if need:
            changes += 1
    if changes:
        print(f"[AlignGeom] {changes}/{len(edge_index)} geometries fixed")
    return geometries


# ── Phase 1: Skeleton ──────────────────────────────────────────────


def generate_skeleton(
    gen: Any,
    condition: torch.Tensor,
    structural_priors: torch.Tensor,
    *,
    map_w: float = 2000.0,
    map_h: float = 2000.0,
    vq_map_size_m: float = 2000.0,
    min_spacing_m: float = 80.0,
    anchor_ratio: float = 0.08,
    temperature: float = 0.75,
    top_p: float = 0.65,
    seed: int | None = None,
) -> dict:
    """Generate and clean a skeleton graph from VQ + Transformer.

    Returns
    -------
    dict with keys ``coords``, ``edge_index``, ``road_field``, ``map_max``,
    ``gridness``, ``organic``, ``density``, ``condition``.
    """
    density = float(structural_priors[0]) if structural_priors.dim() > 0 else 20.0
    gridness = float(structural_priors[1]) if structural_priors.numel() > 1 else 0.5
    organic = float(structural_priors[3]) if structural_priors.numel() > 3 else 0.5
    map_max = max(map_w, map_h)

    with torch.no_grad():
        raw = gen.generate(
            condition.unsqueeze(0) if condition.dim() == 1 else condition,
            anchor_ratio=anchor_ratio,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
    field = raw["road_field"]
    coords_n = raw["coords"].copy()  # normalized [0, 1] in VQ space
    ei = raw["edge_index"].copy()

    if len(coords_n) < 5:
        raise ValueError(f"Skeleton too small ({len(coords_n)} nodes)")

    # Endpoint connector (VQ native size)
    try:
        conn = EndpointConnector(map_size_m=vq_map_size_m).run(
            raw,
            field,
            max_connections=30,
            connect_remaining=True,
            max_remaining_m=600,
            simplify=False,
        )
        coords_n = conn["coords"]
        ei = conn["edge_index"]
    except Exception:
        pass

    # Simplify chains
    try:
        simp = simplify_chains(coords_n, ei, angle_threshold_deg=15, dp_epsilon_norm=0.002)
        coords_n, ei = simp[0], simp[1]
    except Exception:
        pass

    # Scale to target map size
    sx, sy = map_w / vq_map_size_m, map_h / vq_map_size_m
    coords_n = coords_n * np.array([sx, sy])

    # Spacing cleanup
    merge_dist = max(0.005, min_spacing_m / map_max * 0.25)
    merged = merge_close_nodes(
        coords_n,
        ei,
        np.zeros(len(coords_n), dtype=np.int64),
        merge_dist=merge_dist,
        map_size_m=map_max,
    )
    coords_n, ei = merged[0], merged[1]

    return {
        "coords": coords_n,
        "edge_index": ei,
        "road_field": field,
        "map_max": map_max,
        "density": density,
        "gridness": gridness,
        "organic": organic,
        "condition": condition,
    }


# ── Phase 2: Branch ────────────────────────────────────────────────


def generate_branch(
    coords: np.ndarray,
    edge_index: np.ndarray,
    road_field: np.ndarray,
    condition: torch.Tensor,
    *,
    map_w: float = 2000.0,
    map_h: float = 2000.0,
    density: float = 20.0,
    gridness: float = 0.5,
    organic: float = 0.5,
    local_spacing_m: float = 80.0,
    g1_branch_p: float = 0.04,
    grid_cuts: int = 15,
    organic_cuts: int = 20,
    prune_chain_m: float = 120.0,
    snap_dist_m: float = 50.0,
    angle_clean_deg: float = 15.0,
    angle_clean_compressed_deg: float = 20.0,
    parallel_angle_deg: float = 30.0,
    parallel_dist_m: float = 40.0,
    merge_junction_dist_m: float = 50.0,
) -> dict:
    """Grow collector roads (G1) and local roads (G2), then clean up.

    Parameters
    ----------
    coords, edge_index :
        Skeleton graph from ``generate_skeleton`` (normalised [0, 1] in map space).
    road_field :
        VQ road field (for A* endpoint closure).
    condition :
        11-dim style + structural condition tensor.
    density, gridness, organic :
        Structural priors (read from ``structural_priors`` if available).

    Returns
    -------
    dict with keys ``coords_int``, ``edge_index_int``, ``node_types``,
    ``lanes_per_dir``, ``geometries``, ``road_class``.
    """
    map_max = max(map_w, map_h)

    # ── Growth ─────────────────────────────────────────────────────
    gc = GrowthConfig.from_condition(
        condition.cpu().numpy() if condition.dim() > 0 else np.zeros(11),
        local_spacing_m=local_spacing_m,
        map_size_m=map_max,
    )
    gc.map_width_m = map_w
    gc.map_height_m = map_h

    # Style-aware G2 overrides
    if gridness > 0.6:
        gc.g2_max_cuts_per_pass = max(gc.g2_max_cuts_per_pass, grid_cuts)
        gc.g2_jitter_deg = 5.0
        gc.g1_seed_jitter = 0.15
        gc.per_step_jitter_deg = 2.0
    elif organic > 0.6:
        gc.g2_max_cuts_per_pass = max(gc.g2_max_cuts_per_pass, organic_cuts)
        gc.g2_jitter_deg = 35.0
        gc.g1_seed_jitter = 0.35
        gc.per_step_jitter_deg = 8.0
    gc.g1_branch_p = g1_branch_p

    grown = grow(
        coords * np.array([map_w, map_h]),
        edge_index,
        np.zeros(len(coords), dtype=np.int64),
        road_field,
        gc,
    )
    c_m = grown["coords"] / np.array([map_w, map_h])
    ei = grown["edge_index"].copy()
    rc = grown.get("road_class", np.ones(len(ei), dtype=np.int64))

    # ── Merge close nodes in growth graph ─────────────────────────
    _md = max(0.003, 30.0 / map_max)
    _mrg = merge_close_nodes(
        c_m, ei, np.zeros(len(c_m), dtype=np.int64), merge_dist=_md, map_size_m=map_max
    )
    c_m, ei = _mrg[0], _mrg[1]
    # Fix edge crossings in growth graph
    c_m, ei = _fix_growth_crossings(c_m, ei, map_max)

    # ── Cleanup raw graph ──────────────────────────────────────────
    c_m, ei = prune_dead_ends(c_m, ei, prune_chain_m, map_max)
    c_m, ei = keep_lcc(c_m, ei)
    c_m, ei = clean_sharp_angles(c_m, ei, min_deg=angle_clean_deg)
    c_m, ei = snap_endpoints(c_m, ei, map_max, snap_dist_m=snap_dist_m)
    c_m, ei = keep_lcc(c_m, ei)

    # ── Compress to intersection graph ─────────────────────────────
    c_int, ei_int, geoms = compress_to_intersection_graph(c_m, ei)

    # ── Merge nearby compressed junctions ─────────────────────────
    c_int, ei_int, geoms = merge_compressed_graph(c_int, ei_int, geoms, map_max, merge_dist_m=30.0)

    # ── Snap edges to nearby non-adjacent nodes ────────────────────
    # Creates intersections where roads pass close but don't share a node
    c_int, ei_int, geoms = snap_edges_to_nodes(c_int, ei_int, geoms, map_max, snap_dist_m=8.0)

    # ── Cleanup compressed graph ───────────────────────────────────
    # Fix self-intersecting / abnormal geometries
    fix_abnormal_edges(c_int, ei_int, geoms, map_max)
    # Fix geometric crossings
    c_int, ei_int, geoms = fix_edge_crossings(c_int, ei_int, geoms, map_max)
    c_int, ei_int, geoms = clean_parallel_roads(
        c_int, ei_int, geoms, map_max, angle_deg=parallel_angle_deg, max_dist_m=parallel_dist_m
    )

    # ── Inter-edge parallel road dedup (non-adjacent edges) ──────
    c_int, ei_int, geoms = merge_parallel_edges(
        c_int, ei_int, geoms, map_max, angle_deg=parallel_angle_deg, dist_m=parallel_dist_m
    )

    # ── Douglas-Peucker simplify edge geometries ──────────────────
    # Remove micro-oscillations while preserving overall shape (~5m tolerance)
    from shapely import simplify as dp_simplify
    from shapely.geometry import LineString

    dp_tol = 5.0 / map_max  # ~5m in normalised space
    for j in range(len(geoms)):
        if geoms[j] is not None and len(geoms[j]) >= 3:
            ls = LineString(geoms[j])
            simplified = dp_simplify(ls, tolerance=dp_tol)
            if simplified.geom_type == "LineString" and len(simplified.coords) >= 2:
                geoms[j] = np.array(simplified.coords)

    # ── Classify ───────────────────────────────────────────────────
    nt = classify_nodes(c_int, ei_int, map_max, merge_dist_m=merge_junction_dist_m, compressed=True)

    # ── Physically merge nearby junction nodes ────────────────────
    c_int, ei_int, nt, geoms = merge_nearby_junctions(
        c_int, ei_int, nt, geoms, map_max, merge_dist_m=merge_junction_dist_m
    )

    # ── Split degree>=5 junctions into 3-4 arm sub-junctions ─────
    c_int, ei_int, nt, geoms = split_high_degree_junctions(
        c_int, ei_int, nt, geoms, map_max, split_radius_m=3.0
    )

    # ── Second parallel-road cleanup (after structural changes) ──
    c_int, ei_int, geoms = clean_parallel_roads(
        c_int, ei_int, geoms, map_max, angle_deg=parallel_angle_deg, max_dist_m=parallel_dist_m
    )

    # ── Re-align geometry endpoints to node positions ────────────
    # Structural changes can leave geometry endpoints misaligned.
    geoms = align_geometries_to_nodes(c_int, ei_int, geoms)

    # ── Smooth compressed graph geometries ───────────────────────
    # Removes micro-oscillations in edge polylines before offset.
    from assemble_hdmap import _chaikin

    geoms = [_chaikin(g, iterations=2) if len(g) >= 3 else g for g in geoms]

    # ── Propagate road_class through compression ──────────────────
    from collections import deque

    growth_adj = {i: set() for i in range(len(c_m))}
    for a, b in ei:
        ia, ib = int(a), int(b)
        growth_adj[ia].add(ib)
        growth_adj[ib].add(ia)
    growth_rc = {}
    for ei_idx in range(len(ei)):
        u, v = int(ei[ei_idx, 0]), int(ei[ei_idx, 1])
        key = (u, v) if u < v else (v, u)
        growth_rc[key] = int(rc[ei_idx]) if ei_idx < len(rc) else 1

    def _trace_rc(start, end):
        parent = {start: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur == end:
                break
            for nb in growth_adj[cur]:
                if nb not in parent:
                    parent[nb] = cur
                    q.append(nb)
        cur, prev = end, None
        max_rc = 1
        while cur is not None:
            if prev is not None:
                key = (cur, prev) if cur < prev else (prev, cur)
                max_rc = max(max_rc, growth_rc.get(key, 1))
            prev, cur = cur, parent.get(cur)
        return max_rc

    rc_int = np.ones(len(ei_int), dtype=np.int64)
    for j in range(len(ei_int)):
        u, v = int(ei_int[j, 0]), int(ei_int[j, 1])
        rc_int[j] = _trace_rc(u, v)

    # ── Lane assignment ────────────────────────────────────────────
    # Factors: road_class, edge length, endpoint degree, density
    import networkx as nx

    G_lanes = nx.Graph()
    for i in range(len(c_int)):
        G_lanes.add_node(i)
    for u, v in ei_int:
        G_lanes.add_edge(int(u), int(v))

    lanes = np.ones(len(ei_int), dtype=np.int64)
    for j in range(len(ei_int)):
        cls = int(rc_int[j])
        u, v = int(ei_int[j, 0]), int(ei_int[j, 1])

        # Edge length (metres)
        if j < len(geoms) and len(geoms[j]) >= 2:
            pts = np.asarray(geoms[j]) * map_max
        else:
            pts = np.array([c_int[u] * map_max, c_int[v] * map_max])
        edge_len = sum(np.linalg.norm(pts[k + 1] - pts[k]) for k in range(len(pts) - 1))

        # Node importance = sum of degrees of both endpoints
        imp = G_lanes.degree(u) + G_lanes.degree(v)

        # Base from road class
        base = {1: 3, 2: 2, 3: 1}.get(cls, 1)

        # Length adjustment
        if edge_len > 400:
            base += 1
        elif edge_len < 80:
            base -= 1

        # Node importance adjustment (major junctions → more lanes)
        if imp >= 8:
            base += 1

        # Density bonus
        if density > 25 and base < 3:
            base += 1

        lanes[j] = max(1, min(4, base))

    return {
        "coords_int": c_int,
        "edge_index_int": ei_int,
        "node_types": nt,
        "lanes_per_dir": lanes,
        "geometries": geoms,
        "road_class": rc_int,
    }


# ── Public lane assignment ───────────────────────────────────────


def assign_lanes(
    coords_int: np.ndarray,
    edge_index_int: np.ndarray,
    geometries: list[np.ndarray],
    road_class: np.ndarray,
    density: float = 20.0,
) -> np.ndarray:
    """Assign per-direction lane counts (1-4) to each compressed edge.

    Factors: road_class, edge length (in metres), node importance, density.
    Looking at generate_branch for the authoritative in-pipeline version.
    """
    import networkx as nx

    G = nx.Graph()
    for i in range(len(coords_int)):
        G.add_node(i)
    for u, v in edge_index_int:
        G.add_edge(int(u), int(v))

    lanes = np.ones(len(edge_index_int), dtype=np.int64)
    for j in range(len(edge_index_int)):
        cls = int(road_class[j])
        u, v = int(edge_index_int[j, 0]), int(edge_index_int[j, 1])

        if j < len(geometries) and len(geometries[j]) >= 2:
            pts = np.asarray(geometries[j])
        else:
            pts = np.array([coords_int[u], coords_int[v]])
        edge_len = sum(np.linalg.norm(pts[k + 1] - pts[k]) for k in range(len(pts) - 1))

        imp = G.degree(u) + G.degree(v)
        base = {1: 3, 2: 2, 3: 1}.get(cls, 1)
        if edge_len > 400:
            base += 1
        elif edge_len < 80:
            base -= 1
        if imp >= 8:
            base += 1
        if density > 25 and base < 3:
            base += 1
        lanes[j] = max(1, min(4, base))
    return lanes


# ── Combined ───────────────────────────────────────────────────────


def generate_full(
    gen,
    condition: torch.Tensor,
    structural_priors: torch.Tensor,
    *,
    map_w: float = 2000.0,
    map_h: float = 2000.0,
    **kwargs,
) -> dict:
    """Run both phases: skeleton → branch, return merged result."""
    skel = generate_skeleton(gen, condition, structural_priors, map_w=map_w, map_h=map_h, **kwargs)
    branch = generate_branch(
        skel["coords"],
        skel["edge_index"],
        skel["road_field"],
        skel["condition"],
        map_w=map_w,
        map_h=map_h,
        density=skel["density"],
        gridness=skel["gridness"],
        organic=skel["organic"],
        **{
            k: v
            for k, v in kwargs.items()
            if k not in ("anchor_ratio", "temperature", "top_p", "vq_map_size_m", "min_spacing_m")
        },
    )
    return {**skel, **branch}

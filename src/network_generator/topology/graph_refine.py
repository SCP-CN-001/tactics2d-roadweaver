"""Compressed-graph refinement operations."""

from __future__ import annotations

import networkx as nx
import numpy as np
from shapely import simplify as shapely_simplify
from shapely.geometry import LineString, Point, box
from shapely.ops import nearest_points
from shapely.strtree import STRtree

from utils.geometry import segment_intersection as _segment_intersection

from .graph_utils import build_nx


# ===== moved from the old top-level pipeline module =====
def clean_growth_parallels(coords, edge_index, map_size_m, angle_deg=20.0, max_dist_m=30.0):
    """Remove near-parallel non-adjacent edges in the growth graph (no geometries)."""
    adj = {i: set() for i in range(len(coords))}
    for u, v in edge_index:
        adj[int(u)].add(int(v))
        adj[int(v)].add(int(u))
    angle_rad = np.radians(angle_deg)
    norm_d = max_dist_m / map_size_m
    n = len(edge_index)
    to_rm = set()

    # Spatial index: prune candidate pairs to those whose bounding boxes come
    # within norm_d of edge i (O(E log E) instead of O(E²)).  Candidate order
    # is kept ascending to preserve the exact removal sequence of the old loop.
    edge_lines = [LineString([coords[int(u)], coords[int(v)]]) for u, v in edge_index]
    tree = STRtree(edge_lines)

    for i in range(n):
        if i in to_rm:
            continue
        u1, v1 = int(edge_index[i, 0]), int(edge_index[i, 1])
        d1 = coords[v1] - coords[u1]
        l1 = float(np.linalg.norm(d1))
        if l1 < 1e-12:
            continue
            d1 /= l1
        xmin, ymin, xmax, ymax = edge_lines[i].bounds
        q = box(xmin - norm_d, ymin - norm_d, xmax + norm_d, ymax + norm_d)
        for j in sorted(tree.query(q)):
            j = int(j)
            if j <= i or j in to_rm:
                continue
            u2, v2 = int(edge_index[j, 0]), int(edge_index[j, 1])
            if {u1, v1} & {u2, v2}:
                continue
            d2 = coords[v2] - coords[u2]
            l2 = float(np.linalg.norm(d2))
            if l2 < 1e-12:
                continue
                d2 /= l2
            ang = abs(np.arccos(np.clip(float(d1 @ d2), -1, 1)))
            if ang > angle_rad and abs(ang - np.pi) > angle_rad:
                continue
            md = min(
                np.linalg.norm(coords[u1] - coords[u2]),
                np.linalg.norm(coords[u1] - coords[v2]),
                np.linalg.norm(coords[v1] - coords[u2]),
                np.linalg.norm(coords[v1] - coords[v2]),
            )
            if md > norm_d:
                continue
            to_rm.add(i if l1 <= l2 else j)
    if not to_rm:
        return coords, edge_index
    keep = edge_index[[j for j in range(n) if j not in to_rm]]
    used = set()
    for u, v in keep:
        used.add(int(u))
        used.add(int(v))
    orphaned = sorted(set(range(len(coords))) - used)
    if orphaned:
        c = np.delete(coords, orphaned, axis=0)
        keep = np.array(
            [
                [u - sum(1 for o in orphaned if o < u), v - sum(1 for o in orphaned if o < v)]
                for u, v in keep
            ],
            dtype=np.int64,
        )
        return c, keep
    return coords.copy(), keep


def fix_growth_crossings(coords, edge_index, map_size_m):
    """Add nodes where non-adjacent growth-graph edges cross."""
    adj = {i: set() for i in range(len(coords))}
    for u, v in edge_index:
        adj[int(u)].add(int(v))
        adj[int(v)].add(int(u))
    max_n = len(coords)
    new_c = list(coords)
    # Track edge splits: each crossing splits both edges, creating 2 new edges each
    splits = {}  # {old_edge_idx: [(fraction, node_idx), ...]}

    # Spatial index: only test pairs whose bounding boxes overlap (O(E log E)
    # instead of O(E²)).  Exact segment intersection is still verified below.
    edge_lines = [LineString([coords[int(u)], coords[int(v)]]) for u, v in edge_index]
    tree = STRtree(edge_lines)

    for i in range(len(edge_index)):
        u1, v1 = int(edge_index[i, 0]), int(edge_index[i, 1])
        p1, p2 = coords[u1], coords[v1]
        for j in tree.query(edge_lines[i]):
            j = int(j)
            if j <= i:  # avoid self and double-counting (i, j) / (j, i)
                continue
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


def fix_abnormal_edges(coords, edge_index, geometries, map_size_m):
    """Detect and repair self-intersecting or overly-curved edge geometries.

    Modifies *geometries* in-place by simplifying self-intersecting polylines
    and capping curvature via Douglas-Peucker simplification.
    """

    for j in range(len(edge_index)):
        if j >= len(geometries) or len(geometries[j]) < 3:
            continue
        geom = np.asarray(geometries[j])
        ls = LineString(geom)
        if not ls.is_simple:
            # Self-intersecting — simplify with DP tolerance
            simplified = shapely_simplify(ls, tolerance=0.002)
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
    map_size_m: float,
    merge_dist_m: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Merge compressed-graph nodes closer than *merge_dist_m*."""
    if len(coords) < 2:
        return coords, edge_index, geometries

    dist_norm = merge_dist_m / map_size_m
    N = len(coords)

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


def merge_nearby_junctions(
    coords, edge_index, node_types, geometries, map_size_m, merge_dist_m=50.0
):
    """Physically merge junction nodes closer than *merge_dist_m*.

    Unlike classify_nodes which only conceptually groups them, this
    function rewires the graph so that nearby junctions become a
    single node with combined incident edges.
    """
    if len(coords) < 2:
        return coords, edge_index, node_types, geometries

    norm = merge_dist_m / map_size_m

    # Build adjacency
    adj = {i: set() for i in range(len(coords))}
    for u, v in edge_index:
        adj[int(u)].add(int(v))
        adj[int(v)].add(int(u))

    # Use degree >= 3 to identify junction nodes (not node_types, which may
    # have been pre-emptively degrouped by classify_nodes).
    junctions = [i for i in range(len(coords)) if len(adj[i]) >= 3]
    if len(junctions) < 2:
        return coords, edge_index, node_types, geometries

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
    coords, edge_index, node_types, geometries, map_size_m, split_radius_m=3.0
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

    split_rad = split_radius_m / map_size_m
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
                new_coords.append(np.asarray(new_coords[n]) + offset)
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
        np.asarray(new_coords, dtype=np.float64),
        np.array(new_ei, dtype=np.int64),
        np.array(new_nt, dtype=np.int64),
        new_geoms,
    )


# ── Merge inter-edge parallel roads ──────────────────────────────


def merge_parallel_edges(coords, edge_index, geometries, map_size_m, angle_deg=30.0, dist_m=40.0):
    """Remove edges that are geometrically parallel and close.

    Unlike clean_parallel_roads which only checks edges sharing
    a node, this function checks ALL pairs of non-adjacent edges.
    """
    if len(edge_index) < 3:
        return coords, edge_index, geometries
    if isinstance(coords, np.ndarray):
        if coords.ndim == 1:
            coords = coords.reshape(-1, 2)
    elif isinstance(coords, (list, tuple)):
        if len(coords) > 0 and not isinstance(coords[0], (list, tuple, np.ndarray)):
            # Flat list: [x0, y0, x1, y1, ...]
            coords = np.array(coords, dtype=np.float64).reshape(-1, 2)
        else:
            coords = np.array(coords, dtype=np.float64)

    adj = {i: set() for i in range(len(coords))}
    for u, v in edge_index:
        adj[int(u)].add(int(v))
        adj[int(v)].add(int(u))

    angle_rad = np.radians(angle_deg)
    dist_norm = dist_m / map_size_m

    to_remove = set()
    for i in range(len(edge_index)):
        u1, v1 = int(edge_index[i, 0]), int(edge_index[i, 1])
        if i in to_remove:
            continue
        # Direction of edge i
        p1 = coords[u1]
        p2 = coords[v1]
        d1 = np.atleast_1d(np.asarray(p2, dtype=np.float64) - np.asarray(p1, dtype=np.float64))
        n1 = float(np.linalg.norm(d1))
        if n1 < 1e-12:
            continue
        d1 = d1 / n1

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
            d2 = np.atleast_1d(np.asarray(p4, dtype=np.float64) - np.asarray(p3, dtype=np.float64))
            n2 = float(np.linalg.norm(d2))
            if n2 < 1e-12:
                continue
            d2 = d2 / n2
            try:
                dot = float(d1 @ d2)
            except (ValueError, TypeError):
                try:
                    dot = float((d1 * d2).sum())
                except (ValueError, TypeError):
                    # Can't compute angle — assume parallel and check distance
                    dot = 1.0
            angle = abs(np.arccos(np.clip(dot, -1, 1)))
            if angle > angle_rad and abs(angle - np.pi) > angle_rad:
                continue

            # Distance check: true edge-to-edge distance (handles staggered overlaps)
            _g1 = (
                geometries[i]
                if i < len(geometries) and len(geometries[i]) >= 2
                else np.array([coords[u1], coords[v1]])
            )
            _g2 = (
                geometries[j]
                if j < len(geometries) and len(geometries[j]) >= 2
                else np.array([coords[u2], coords[v2]])
            )
            min_dist = LineString(_g1).distance(LineString(_g2))
            if min_dist > dist_norm:
                continue

            # Revert to length-based comparison for reliability
            n1 = float(np.linalg.norm(coords[v1] - coords[u1]))
            n2 = float(np.linalg.norm(coords[v2] - coords[u2]))
            to_remove.add(i if n1 >= n2 else j)

    if not to_remove:
        return coords, edge_index, geometries

    keep_idx = [j for j in range(len(edge_index)) if j not in to_remove]
    keep_ei = edge_index[keep_idx]
    keep_geoms = [geometries[j] for j in keep_idx]

    # Remove orphaned nodes
    G = build_nx(coords, keep_ei)
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


# ── Redundant chord removal ──────────────────────────────────────


def remove_redundant_chords(
    coords,
    edge_index,
    geometries,
    map_size_m,
    max_dist_m: float = 40.0,
    detour_ratio: float = 0.10,
    angle_deg: float = 20.0,
):
    """Remove a chord edge (u,v) that duplicates a two-edge path u-w-v.

    A chord (u,v) is redundant when a common neighbour ``w`` (so a two-edge
    path u-w-v exists) lies *on the same straight corridor* as the chord: near
    the (u, v) geometry, the path adds negligible extra length, and both
    segments stay near-collinear with the chord.  These three conditions are
    checked together so that legitimate roads which merely pass near the chord
    line (angled detours, genuine small blocks) are preserved.

    Remove the chord and keep the subdivided path, so ``w`` (and any roads
    branching from it) stay connected.  Existing parallel cleanup only compares
    *non-adjacent* edge pairs, so a chord that shares an endpoint with both
    path edges structurally escapes it.
    """
    if len(edge_index) < 3:
        return coords, edge_index, geometries
    dist_norm = max_dist_m / map_size_m
    angle_rad = np.radians(angle_deg)

    adj: dict[int, set[int]] = {i: set() for i in range(len(coords))}
    for u, v in edge_index:
        adj[int(u)].add(int(v))
        adj[int(v)].add(int(u))

    to_remove: set[int] = set()
    for e, (u, v) in enumerate(edge_index):
        u, v = int(u), int(v)
        if u == v or e in to_remove:
            continue
        common = adj[u] & adj[v]
        if not common:
            continue
        geom = (
            np.asarray(geometries[e], dtype=float)
            if e < len(geometries) and len(geometries[e]) >= 2
            else np.array([coords[u], coords[v]])
        )
        line = LineString(geom)
        chord_len = line.length
        if chord_len < 1e-12:
            continue
        cu = np.asarray(coords[u], dtype=float)
        cv = np.asarray(coords[v], dtype=float)
        uv = cv - cu
        vu = -uv
        for w in common:
            pw = np.asarray(coords[w], dtype=float)
            if line.distance(Point(pw)) >= dist_norm:
                continue
            wu = pw - cu
            wv = pw - cv
            path_len = float(np.linalg.norm(wu)) + float(np.linalg.norm(wv))
            if (path_len - chord_len) / chord_len >= detour_ratio:
                continue
            a1 = abs(np.degrees(np.arctan2(np.cross(uv, wu), np.dot(uv, wu))))
            a2 = abs(np.degrees(np.arctan2(np.cross(vu, wv), np.dot(vu, wv))))
            if max(a1, a2) >= angle_deg:
                continue
            to_remove.add(e)
            break

    if not to_remove:
        return coords, edge_index, geometries

    print(f"  [Chord] {len(to_remove)} redundant chords removed")
    keep_idx = [j for j in range(len(edge_index)) if j not in to_remove]
    keep_ei = edge_index[keep_idx]
    keep_geoms = [geometries[j] for j in keep_idx]

    # Drop nodes orphaned by the removals (chord endpoints stay connected via w).
    G = build_nx(coords, keep_ei)
    orphaned = [n for n in range(len(coords)) if G.degree(n) == 0]
    if orphaned:
        c = np.delete(coords, orphaned, axis=0)
        remap = {n: k for k, n in enumerate(n for n in range(len(coords)) if n not in orphaned)}
        keep_ei = np.array([[remap[int(u)], remap[int(v)]] for u, v in keep_ei], dtype=np.int64)
        return c, keep_ei, keep_geoms
    return coords.copy(), keep_ei, keep_geoms


# ── Snap edges to nearby nodes ────────────────────────────────────


def snap_edges_to_nodes(coords, edge_index, geometries, map_size_m, snap_dist_m=8.0):
    """Split compressed edges where they pass close to non-adjacent nodes.

    When an edge geometry passes within *snap_dist_m* of a node that is
    *not* one of its endpoints, the edge is subdivided at the closest
    point and a new connecting edge is inserted, creating an intersection.
    """

    adj: dict[int, set[int]] = {i: set() for i in range(len(coords))}
    for u, v in edge_index:
        adj[int(u)].add(int(v))
        adj[int(v)].add(int(u))

    split_edges: dict[int, list[tuple[float, int, np.ndarray]]] = {}
    # (edge_idx -> list of (frac_on_line, new_node_idx, split_position))
    new_nodes: list[np.ndarray] = []
    snap_norm = snap_dist_m / map_size_m

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


# ── Merge near-miss edges into intersections ──────────────────────


def merge_near_miss_edges(
    coords, edge_index, geometries, map_size_m, near_dist_m=15.0, angle_deg=30.0
):
    """Merge non-adjacent edges that pass close to each other (near-misses).

    Two edges that do not share a node but whose geometries come within
    ``near_dist_m`` of each other far from any node never intersect
    topologically, yet their lane envelopes overlap once lanes are widened.
    ``fix_edge_crossings`` only catches exact crossings, ``merge_parallel_edges``
    only catches near-parallel pairs, and ``snap_edges_to_nodes`` only catches
    edge-vs-node proximity.  This step closes the gap: it splits both edges at
    their closest-approach point and merges the two new nodes into one shared
    junction, so the roads properly intersect.

    Near-parallel pairs (angle within ``angle_deg`` of collinear) are left to
    ``merge_parallel_edges``, which removes the shorter duplicate road.
    """
    if len(edge_index) < 2:
        return coords, edge_index, geometries
    near_norm = near_dist_m / map_size_m

    c = [np.asarray(p, dtype=float) for p in coords]
    ei = [(int(u), int(v)) for u, v in edge_index]
    geoms = [np.asarray(g, dtype=float) for g in geometries]

    def _line(idx):
        g = geoms[idx]
        if len(g) >= 2:
            return LineString(g)
        u, v = ei[idx]
        return LineString([c[u], c[v]])

    lines = [_line(i) for i in range(len(ei))]
    tree = STRtree(lines)

    angle_rad = np.radians(angle_deg)

    # Two resolutions for a near-miss:
    #   crossing-like (angle between angle_deg and 180-angle_deg) -> the two
    #     roads visually cross, so give them a shared intersection node.
    #   near-parallel (angle within angle_deg of collinear)          -> the two
    #     roads are a duplicated corridor, so merge them by keeping the longer.
    junction_pairs: list[tuple[int, int, float, float, np.ndarray]] = []
    remove_set: set[int] = set()

    for i in range(len(ei)):
        if i in remove_set:
            continue
        u1, v1 = ei[i]
        for j in tree.query(lines[i]):
            j = int(j)
            if j <= i or j in remove_set:
                continue
            u2, v2 = ei[j]
            if len({u1, v1} & {u2, v2}) > 0:
                continue
            ls_i, ls_j = lines[i], lines[j]
            if ls_i.length < 1e-9 or ls_j.length < 1e-9:
                continue
            if ls_i.distance(ls_j) > near_norm:
                continue
            d1 = np.asarray(c[v1], dtype=float) - np.asarray(c[u1], dtype=float)
            d2 = np.asarray(c[v2], dtype=float) - np.asarray(c[u2], dtype=float)
            n1, n2 = float(np.linalg.norm(d1)), float(np.linalg.norm(d2))
            if n1 < 1e-12 or n2 < 1e-12:
                continue
            ang = abs(np.arccos(np.clip(float(np.dot(d1 / n1, d2 / n2)), -1, 1)))
            if angle_rad < ang < np.pi - angle_rad:
                # Crossing-like: split both edges and merge into a shared node.
                p_i, p_j = nearest_points(ls_i, ls_j)
                f_i = ls_i.project(p_i) / ls_i.length
                f_j = ls_j.project(p_j) / ls_j.length
                if f_i < 0.05 or f_i > 0.95 or f_j < 0.05 or f_j > 0.95:
                    continue  # too close to an endpoint — already a node
                mid = np.array([(p_i.x + p_j.x) / 2, (p_i.y + p_j.y) / 2])
                junction_pairs.append((i, j, f_i, f_j, mid))
            else:
                # Near-parallel duplicated corridor: keep the longer road.
                remove_set.add(i if ls_i.length >= ls_j.length else j)

    if not junction_pairs and not remove_set:
        return coords, edge_index, geometries

    # One shared junction node per crossing-like near-miss pair.
    node_of_mid: dict[tuple, int] = {}
    new_nodes: list[np.ndarray] = []

    def _node(mid):
        key = tuple(np.round(mid, 6))
        if key in node_of_mid:
            return node_of_mid[key]
        nid = len(c) + len(new_nodes)
        node_of_mid[key] = nid
        new_nodes.append(mid)
        return nid

    # edge idx -> list of (frac, node_id, node_pos)
    splits: dict[int, list[tuple[float, int, np.ndarray]]] = {}
    for i, j, f_i, f_j, mid in junction_pairs:
        nid = _node(mid)
        if i not in remove_set:
            splits.setdefault(i, []).append((f_i, nid, mid))
        if j not in remove_set:
            splits.setdefault(j, []).append((f_j, nid, mid))

    # Rebuild edges, splitting each at its cut points (mirrors fix_edge_crossings).
    out_c = list(c)
    out_ei: list[tuple[int, int]] = []
    out_geoms: list[np.ndarray] = []

    for idx in range(len(ei)):
        if idx in remove_set:
            continue
        u, v = ei[idx]
        if idx not in splits:
            out_ei.append((u, v))
            out_geoms.append(geoms[idx])
            continue
        g = geoms[idx]
        if len(g) < 2:
            g = np.array([c[u], c[v]])
        seg_len = np.sqrt(((g[1:] - g[:-1]) ** 2).sum(axis=1))
        cum = np.concatenate([[0], np.cumsum(seg_len)])
        total = float(cum[-1])
        if total < 1e-12:
            out_ei.append((u, v))
            out_geoms.append(g)
            continue

        cuts = sorted(splits[idx], key=lambda x: x[0])
        deduped: list[tuple[float, int, np.ndarray]] = []
        for f, nid, pos in cuts:
            if not deduped or f - deduped[-1][0] > 1e-4:
                deduped.append((f, nid, pos))

        prev_node = u
        prev_pt = np.asarray(c[u], dtype=float)
        prev_cum = 0.0
        for f, nid, pos in deduped:
            target_cum = f * total
            sub_pts = [prev_pt.copy()]
            for k in range(1, len(g)):
                if cum[k] <= prev_cum + 1e-12:
                    continue
                if cum[k - 1] >= target_cum - 1e-12:
                    break
                seg_start = max(prev_cum, cum[k - 1])
                seg_end = min(target_cum, cum[k])
                if seg_end <= seg_start:
                    continue
                t = (seg_start - cum[k - 1]) / (cum[k] - cum[k - 1] + 1e-12)
                sub_pts.append(g[k - 1] + t * (g[k] - g[k - 1]))
            sub_pts.append(pos)
            out_ei.append((prev_node, nid))
            out_geoms.append(np.array(sub_pts))
            prev_node = nid
            prev_pt = pos
            prev_cum = target_cum

        sub_pts = [prev_pt.copy()]
        for k in range(1, len(g)):
            if cum[k] <= prev_cum + 1e-12:
                continue
            t = (max(prev_cum, cum[k - 1]) - cum[k - 1]) / (cum[k] - cum[k - 1] + 1e-12)
            sub_pts.append(g[k - 1] + t * (g[k] - g[k - 1]))
        if len(sub_pts) >= 2:
            out_ei.append((prev_node, v))
            out_geoms.append(np.array(sub_pts))

    out_c = out_c + new_nodes

    # Drop nodes orphaned by edge removal and re-index.
    used = {a for a, b in out_ei} | {b for a, b in out_ei}
    old = sorted(used)
    remap = {n: k for k, n in enumerate(old)}
    out_c = np.array([out_c[n] for n in old], dtype=float)
    out_ei = np.array([[remap[a], remap[b]] for a, b in out_ei], dtype=np.int64)

    print(
        f"  [NearMiss] {len(junction_pairs)} crossing near-misses -> junctions, "
        f"{len(remove_set)} parallel duplicates removed"
    )
    return out_c, out_ei, out_geoms

"""Intersection-graph construction and node classification."""

from __future__ import annotations

from collections import deque

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from shapely import simplify as shapely_simplify
from shapely.geometry import LineString

from .graph_utils import NT_ENDPOINT, NT_JUNCTION, NT_ROUNDABOUT, NT_WAYPOINT


def detect_roundabouts(
    coords: np.ndarray,
    edge_index: np.ndarray,
    node_types: np.ndarray,
    map_size_m: float = 5000.0,
    max_cycle_size: int = 80,
    skip_area_check: bool = False,
    skip_deg2_check: bool = False,
) -> np.ndarray:
    """Detect roundabout cycles and return updated node_types with NT_ROUNDABOUT set.

    Filters:
      - Cycle size 3-*max_cycle_size* (default 80 for raw skeleton, 12 for simplified).
      - At least 40% of cycle nodes are deg-2.
      - Compactness CV < 0.5 (near-circular).
      - Area 300-30,000 m² (unless *skip_area_check*).
      - All cycle nodes degree 2-3 (pure ring + entry/exit only).

    Args:
        coords: (N, 2) in [0, 1]
        edge_index: (E, 2)
        node_types: (N,) current types
        map_size_m: map size in metres (for area check).
        max_cycle_size: max cycle nodes to consider.
        skip_area_check: skip area filter (for simplified graphs).

    Returns:
        updated node_types with roundabout nodes marked as NT_ROUNDABOUT (3).
    """
    N = len(coords)
    G = nx.Graph()
    for i in range(N):
        G.add_node(i)
    for u, v in edge_index:
        G.add_edge(int(u), int(v))

    try:
        cycles = nx.cycle_basis(G)
    except nx.NetworkXNoCycle:
        return node_types.copy()

    types = node_types.copy()
    deg = {i: G.degree(i) for i in range(N)}

    for cycle in cycles:
        if len(cycle) < 3 or len(cycle) > max_cycle_size:
            continue

        # All cycle nodes must be deg 2 or 3 (pure ring + entry/exit)
        ring_degs = [deg.get(n, 0) for n in cycle]
        if any(d > 3 for d in ring_degs):
            continue

        # At least 40% should be deg-2 (ring nodes), unless skip_deg2_check
        # (compressed intersection graphs have no degree-2 nodes)
        deg2_count = sum(1 for d in ring_degs if d == 2)
        if not skip_deg2_check and deg2_count < len(cycle) * 0.4:
            continue

        # At least one node must have deg >= 3 (connection to external road network)
        # This excludes isolated noise blobs where all nodes are deg=2.
        if not any(d >= 3 for d in ring_degs):
            continue

        # Order nodes along the cycle for polygon area
        cycle_set = set(cycle)
        ordered = [cycle[0]]
        prev = cycle[0]
        while len(ordered) < len(cycle):
            for nb in G.neighbors(ordered[-1]):
                if nb in cycle_set and nb != prev:
                    ordered.append(nb)
                    prev = ordered[-2]
                    break
            else:
                break  # safety break

        pts = np.array([coords[n] for n in ordered])
        cx, cy = pts.mean(axis=0)
        dists = np.linalg.norm(pts - [cx, cy], axis=1)
        mean_d = float(dists.mean())
        if mean_d < 5.0 / map_size_m:
            continue
        cv = float(dists.std()) / max(mean_d, 1e-8)
        if cv >= 0.45:
            continue

        # Area check: shoelace formula on ordered polygon
        if not skip_area_check and len(ordered) >= 3:
            xs, ys = pts[:, 0], pts[:, 1]
            area = 0.5 * abs(float(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1))))
            area_m2 = area * map_size_m * map_size_m
            if area_m2 < 300 or area_m2 > 5000:
                continue

        for n in cycle:
            types[n] = NT_ROUNDABOUT

    return types


# ── Graph compression ────────────────────────────────────────────────


def compress_to_intersection_graph(coords, edge_index):
    """Contract degree-2 nodes into an intersection graph."""
    if len(coords) < 5:
        return coords, edge_index, []

    G = nx.MultiGraph()
    for i, p in enumerate(coords):
        G.add_node(i, pos=p)
    for u, v in edge_index:
        G.add_edge(int(u), int(v))

    # --- Contract degree-2 nodes ---
    changed = True
    while changed:
        changed = False
        for n in list(G.nodes()):
            if G.degree(n) != 2:
                continue
            nbs = list(G.neighbors(n))
            if len(nbs) != 2:
                continue
            u, v = nbs
            G.add_edge(u, v)  # MultiGraph: adds parallel edge if u-v exists
            G.remove_node(n)
            changed = True
            break

    # --- Build geometry: BFS on original adjacency, avoiding already-traced chains ---
    orig_adj = {i: set() for i in range(len(coords))}
    for a, b in edge_index:
        ia, ib = int(a), int(b)
        orig_adj[ia].add(ib)
        orig_adj[ib].add(ia)

    def _trace(start, end, blocked=frozenset()):
        """BFS from *start* to *end* avoiding *blocked* nodes.

        Returns (node_path, coord_path).  Falls back to straight line
        when no path exists (parallel-chain edge case).
        """
        parent = {start: None}
        q = deque([start])
        found = False
        while q:
            cur = q.popleft()
            if cur == end:
                found = True
                break
            for nb in orig_adj[cur]:
                if nb not in parent and nb not in blocked:
                    parent[nb] = cur
                    q.append(nb)
        if not found:
            return [], np.array([coords[start], coords[end]])
        path = []
        cur = end
        while cur is not None:
            path.append(cur)
            cur = parent.get(cur)
        path.reverse()
        _coord_path = np.array([coords[n] for n in path])
        if len(_coord_path) >= 5:
            _simplified = shapely_simplify(LineString(_coord_path), tolerance=0.003)
            if _simplified.geom_type == "LineString" and len(_simplified.coords) >= 2:
                _coord_path = np.array(_simplified.coords, dtype=np.float64)
            # Second pass: Chaikin smoothing to further straighten zigzag
            if len(_coord_path) >= 4:
                from utils.geometry import chaikin as _chk

                _coord_path = _chk(_coord_path, iterations=1)
        return path, _coord_path

    nodes = sorted(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    c2 = np.array([G.nodes[n]["pos"] for n in nodes])
    e2, geoms = [], []
    traced_interior: set[int] = set()
    for u, v in G.edges():
        if u == v:
            continue  # self-loop, skip
        node_path, coord_path = _trace(u, v, blocked=frozenset(traced_interior))
        if len(node_path) > 2:
            traced_interior.update(node_path[1:-1])
        e2.append([idx[u], idx[v]])
        geoms.append(coord_path)
    return c2, np.array(e2, dtype=np.int64), geoms


def classify_nodes(coords, edge_index, map_size_m, merge_dist_m=50.0, compressed=False):
    """Classify graph nodes by degree and structure."""

    G = nx.Graph()
    for i in range(len(coords)):
        G.add_node(i, pos=coords[i].copy())
    G.add_edges_from(edge_index)

    nt = np.zeros(len(coords), dtype=np.int64)
    for n in range(len(coords)):
        d = G.degree(n)
        if d >= 3:
            nt[n] = NT_JUNCTION
        elif d <= 1:
            nt[n] = NT_ENDPOINT
        else:
            nt[n] = NT_WAYPOINT

    nt = detect_roundabouts(
        coords, edge_index, nt, map_size_m=map_size_m, max_cycle_size=12, skip_deg2_check=compressed
    )

    tree = cKDTree(coords)
    for i, j in tree.query_pairs(merge_dist_m / map_size_m):
        if nt[i] == NT_JUNCTION and nt[j] == NT_JUNCTION:
            nt[j] = NT_WAYPOINT
    return nt


def propagate_road_class(
    growth_coords, growth_edge_index, growth_road_class, compressed_edge_index
) -> np.ndarray:
    """Trace each compressed edge through the growth graph, returning its max road class."""
    adj = {i: set() for i in range(len(growth_coords))}
    for a, b in growth_edge_index:
        ia, ib = int(a), int(b)
        adj[ia].add(ib)
        adj[ib].add(ia)
    rc_by_edge = {}
    for i in range(len(growth_edge_index)):
        u, v = int(growth_edge_index[i, 0]), int(growth_edge_index[i, 1])
        key = (u, v) if u < v else (v, u)
        rc_by_edge[key] = int(growth_road_class[i]) if i < len(growth_road_class) else 1

    def _trace(start, end):
        parent = {start: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur == end:
                break
            for nb in adj[cur]:
                if nb not in parent:
                    parent[nb] = cur
                    q.append(nb)
        cur, prev = end, None
        max_rc = 1
        while cur is not None:
            if prev is not None:
                key = (cur, prev) if cur < prev else (prev, cur)
                max_rc = max(max_rc, rc_by_edge.get(key, 1))
            prev, cur = cur, parent.get(cur)
        return max_rc

    out = np.ones(len(compressed_edge_index), dtype=np.int64)
    for j in range(len(compressed_edge_index)):
        u, v = int(compressed_edge_index[j, 0]), int(compressed_edge_index[j, 1])
        out[j] = _trace(u, v)
    return out

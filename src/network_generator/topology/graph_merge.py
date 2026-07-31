"""Type-aware merging of nearby graph nodes."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .graph_utils import NT_CURVE, NT_ENDPOINT, NT_JUNCTION, NT_ROUNDABOUT, NT_WAYPOINT


def merge_close_nodes(
    coords: np.ndarray,
    edge_index: np.ndarray,
    node_types: np.ndarray,
    merge_dist: float,
    map_size_m: float = 5000.0,
    dp_epsilon_norm: float = 0.002,
    max_graph_hops: int = 1,
    closure_edges: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, dict[int, int]]:
    """Type-aware node merging with graph-distance constraint.

    When *closure_edges* is provided, the old-style (coords, edges, types,
    closure_edges, old2new) signature is used for backward compatibility.

    Only merges nodes within *max_graph_hops* hops on the graph topology,
    and for junction clusters checks direction alignment to avoid absorbing
    curve points on incoming roads.

    Args:
        coords: (N, 2)
        edge_index: (E, 2)
        node_types: (N,)
        merge_dist: Euclidean distance threshold (normalised [0,1]).
        map_size_m: map size in metres.
        dp_epsilon_norm: vertical deviation threshold for curve-point DP check.
        max_graph_hops: max graph hops for merging (default 1 = direct neighbour).
        closure_edges: if provided, merges in old-style mode (for backward compat).

    Returns:
        When closure_edges is None: (merged_coords, merged_edges, merged_types, None, old2new).
        When closure_edges is given: (merged_coords, merged_edges, merged_types, merged_closure_edges, old2new).
    """
    N = len(coords)
    if N == 0 or merge_dist <= 0:
        if closure_edges is not None:
            return coords, edge_index, node_types, closure_edges, {i: i for i in range(N)}
        return coords, edge_index, node_types, None, {i: i for i in range(N)}

    # Build graph adjacency (from edges) and proximity adjacency
    graph_adj: dict[int, set[int]] = {i: set() for i in range(N)}
    for u, v in edge_index:
        graph_adj[int(u)].add(int(v))
        graph_adj[int(v)].add(int(u))

    prox_adj: dict[int, set[int]] = {i: set() for i in range(N)}
    if N < 2:
        pass
    else:
        tree = cKDTree(coords)
        for i, j in tree.query_pairs(merge_dist):
            prox_adj[i].add(j)
            prox_adj[j].add(i)

    def _graph_distance(a: int, b: int) -> int:
        if a == b:
            return 0
        if b in graph_adj.get(a, set()):
            return 1
        if max_graph_hops <= 1:
            return max_graph_hops + 1
        visited = {a}
        queue = [(a, 0)]
        for node, d in queue:
            if d >= max_graph_hops:
                continue
            for nb in graph_adj.get(node, set()):
                if nb == b:
                    return d + 1
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, d + 1))
        return max_graph_hops + 1

    def _should_absorb(jn: int, target: int) -> bool:
        if _graph_distance(jn, target) > max_graph_hops:
            return False
        if int(node_types[target]) == NT_CURVE:
            vec = coords[target] - coords[jn]
            d_abs = float(np.linalg.norm(vec))
            if d_abs < 1e-10:
                return True
            for nb in graph_adj.get(jn, set()):
                edge_vec = coords[nb] - coords[jn]
                ed = float(np.linalg.norm(edge_vec))
                if ed < 1e-10:
                    continue
                cos_sim = float(np.dot(vec, edge_vec)) / (d_abs * ed)
                if cos_sim > 0.7:
                    return False
            return True
        return True

    # Proximity clusters
    visited: set[int] = set()
    clusters: list[list[int]] = []
    for i in range(N):
        if i in visited:
            continue
        stack, cl = [i], []
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            cl.append(n)
            for nb in prox_adj.get(n, []):
                if nb not in visited:
                    stack.append(nb)
        clusters.append(cl)

    if len(clusters) == N:
        if closure_edges is not None:
            return coords, edge_index, node_types, closure_edges, {i: i for i in range(N)}
        return coords, edge_index, node_types, None, {i: i for i in range(N)}

    old2new: dict[int, int] = {}
    new_coords_list: list[np.ndarray] = []
    new_types_list: list[int] = []

    for cl in clusters:
        if len(cl) == 1:
            n = cl[0]
            old2new[n] = len(new_coords_list)
            new_coords_list.append(coords[n])
            new_types_list.append(int(node_types[n]))
            continue

        types_in_cl = {int(node_types[n]) for n in cl}

        if NT_ROUNDABOUT in types_in_cl:
            for n in cl:
                old2new[n] = len(new_coords_list)
                new_coords_list.append(coords[n])
                new_types_list.append(int(node_types[n]))

        elif NT_JUNCTION in types_in_cl:
            jns = [n for n in cl if int(node_types[n]) == NT_JUNCTION]
            jn = jns[0]
            old2new[jn] = len(new_coords_list)
            new_coords_list.append(coords[jn])
            new_types_list.append(NT_JUNCTION)
            for n in cl:
                if n == jn:
                    continue
                if _should_absorb(jn, n):
                    old2new[n] = old2new[jn]
                else:
                    old2new[n] = len(new_coords_list)
                    new_coords_list.append(coords[n])
                    new_types_list.append(int(node_types[n]))
            if len(jns) > 1:
                for jn2 in jns[1:]:
                    old2new[jn2] = old2new[jn]

        elif NT_CURVE in types_in_cl:
            pts = np.array([coords[n] for n in cl])
            if len(cl) <= 3:
                cid = len(new_coords_list)
                for n in cl:
                    old2new[n] = cid
                new_coords_list.append(pts.mean(axis=0))
                new_types_list.append(NT_CURVE)
            else:
                end0, end1 = pts[0], pts[-1]
                v = end1 - end0
                vn = float(np.linalg.norm(v))
                if vn > 1e-10:
                    devs = [float(np.linalg.norm(np.cross(v, p - end0))) / vn for p in pts]
                    max_dev = max(devs)
                else:
                    max_dev = 0.0
                if max_dev < dp_epsilon_norm:
                    cid = len(new_coords_list)
                    for n in cl:
                        old2new[n] = cid
                    new_coords_list.append(pts.mean(axis=0))
                    new_types_list.append(NT_CURVE)
                else:
                    for n in cl:
                        old2new[n] = len(new_coords_list)
                        new_coords_list.append(coords[n])
                        new_types_list.append(int(node_types[n]))

        elif NT_ENDPOINT in types_in_cl:
            eps = [n for n in cl if int(node_types[n]) == NT_ENDPOINT]
            if eps:
                cid = len(new_coords_list)
                for n in cl:
                    old2new[n] = cid
                ep_pts = np.array([coords[n] for n in eps])
                new_coords_list.append(ep_pts.mean(axis=0))
                new_types_list.append(NT_ENDPOINT)
            else:
                cid = len(new_coords_list)
                for n in cl:
                    old2new[n] = cid
                new_coords_list.append(np.mean([coords[n] for n in cl], axis=0))
                new_types_list.append(NT_WAYPOINT)
        else:
            cid = len(new_coords_list)
            for n in cl:
                old2new[n] = cid
            new_coords_list.append(np.mean([coords[n] for n in cl], axis=0))
            new_types_list.append(NT_WAYPOINT)

    merged_coords = np.array(new_coords_list, dtype=np.float32)
    merged_types = np.array(new_types_list, dtype=np.int64)

    edge_set: set[tuple[int, int]] = set()
    for u, v in edge_index:
        nu, nv = old2new[int(u)], old2new[int(v)]
        if nu != nv:
            edge_set.add((nu, nv) if nu < nv else (nv, nu))
    merged_edges = (
        np.array(list(edge_set), dtype=np.int64).reshape(-1, 2)
        if edge_set
        else np.empty((0, 2), dtype=np.int64)
    )

    if closure_edges is not None:
        ce_list: list[tuple[int, int]] = []
        for a, b in closure_edges:
            na, nb = old2new.get(int(a)), old2new.get(int(b))
            if na is not None and nb is not None and na != nb:
                ce_list.append((na, nb) if na < nb else (nb, na))
        ce_set = set(ce_list)
        merged_ce = (
            np.array(list(ce_set), dtype=np.int64).reshape(-1, 2)
            if ce_set
            else np.empty((0, 2), dtype=np.int64)
        )
        return merged_coords, merged_edges, merged_types, merged_ce, old2new

    return merged_coords, merged_edges, merged_types, None, old2new

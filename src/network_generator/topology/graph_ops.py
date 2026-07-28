"""
Graph utility operations: adjacency, components, chain simplification, node merging.

All functions operate on normalized [0, 1] graph coordinates unless noted.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


def adjacency_from_edges(edge_index: np.ndarray, N: int) -> dict[int, list[int]]:
    """Build {node_id: [neighbor_ids]} from an (E, 2) edge array."""
    adj: dict[int, list[int]] = defaultdict(list)
    for u, v in edge_index:
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))
    return adj


def find_components(adj: dict[int, list[int]], node_count: int | None = None) -> list[set[int]]:
    """Return a list of connected-component sets."""
    all_nodes = set(adj.keys())
    if node_count is not None:
        all_nodes.update(range(node_count))
    visited: set[int] = set()
    comps: list[set[int]] = []
    for i in sorted(all_nodes):
        if i in visited:
            continue
        stack, cl = [i], set()
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            cl.add(n)
            for nb in adj.get(n, []):
                if nb not in visited:
                    stack.append(nb)
        comps.append(cl)
    return comps


def components_with_edges(coords: np.ndarray, edge_index: np.ndarray) -> list[set[int]]:
    """Find connected components from graph data directly."""
    adj = adjacency_from_edges(edge_index, len(coords))
    return find_components(adj, len(coords))


def component_of_nodes(coords: np.ndarray, edge_index: np.ndarray) -> dict[int, int]:
    """Return {node_id: component_id}."""
    comps = components_with_edges(coords, edge_index)
    comp_of: dict[int, int] = {}
    for ci, cl in enumerate(comps):
        for n in cl:
            comp_of[n] = ci
    return comp_of


def endpoint_nodes(adj: dict[int, list[int]]) -> list[int]:
    """Return list of node IDs with degree == 1."""
    return [i for i, nbrs in adj.items() if len(nbrs) == 1]


def simplify_chains(
    coords: np.ndarray,
    edge_index: np.ndarray,
    force_keep: set[int] | None = None,
    angle_threshold_deg: float = 30.0,
    dp_epsilon_norm: float = 0.0,
    max_seg_len_norm: float = 0.04,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int], dict[tuple[int, int], float]]:
    """Collapse degree-2 chains into single edges.

    When *angle_threshold_deg* > 0, degree-2 nodes where the turning angle
    exceeds this threshold are kept as split points — this preserves curves.

    When *dp_epsilon_norm* > 0, a second pass removes redundant curve points
    whose perpendicular deviation from the straight line between neighbours
    is below *dp_epsilon_norm* (Douglas-Peucker style).  *max_seg_len_norm*
    limits how far apart kept points can be (default 0.04 ≈ 200m on 5km map).

    Returns (simp_coords, simp_edges, simp_types, old2new_map, chain_lengths).
    ``simp_types``: 1 = junction (deg ≥ 3), 4 = endpoint (deg ≤ 1).
    ``chain_lengths``: {(simp_u, simp_v): sum_of_segment_lengths} — the
            polyline length of each simplified edge in normalised space.
    """
    N = len(coords)
    adj = adjacency_from_edges(edge_index, N)

    deg = {i: len(adj[i]) for i in range(N)}

    keep: set[int] = {i for i in range(N) if deg[i] != 2}
    if force_keep:
        keep.update(force_keep)
    if not keep:
        keep = set(range(N))

    klist = sorted(keep)
    old2new = {o: n for n, o in enumerate(klist)}
    simp_coords = [coords[i] for i in klist]  # list for dynamic append
    curve_newids: set[int] = set()  # new indices of curve-split points

    # Build simplified adjacency + track chain lengths
    simp_adj: dict[int, set[int]] = defaultdict(set)
    chain_lengths: dict[tuple[int, int], float] = {}

    def _walk_chain(start: int, first_nb: int):
        """Walk a degree-2 chain, splitting at sharp turns."""
        nonlocal simp_coords, curve_newids
        chain_nodes = [start, first_nb]
        prev, cur = start, first_nb
        while cur not in keep:
            nxt = [n for n in adj[cur] if n != prev]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            chain_nodes.append(cur)

        end = cur
        if end == start or end not in keep:
            return  # chain leads to dead end or self-loop

        seg_start = 0
        seg_len = 0.0
        for i in range(1, len(chain_nodes) - 1):
            p, c, n = chain_nodes[i - 1], chain_nodes[i], chain_nodes[i + 1]
            v1 = coords[c] - coords[p]
            v2 = coords[n] - coords[c]
            seg_len += float(np.linalg.norm(v1))

            dot = np.dot(v1, v2)
            norm = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12
            cos_angle = np.clip(dot / norm, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))

            if angle > angle_threshold_deg:
                split_node = chain_nodes[i]
                if split_node not in old2new:
                    old2new[split_node] = len(simp_coords)
                    simp_coords.append(coords[split_node])
                    curve_newids.add(old2new[split_node])

                a = old2new[chain_nodes[seg_start]]
                b = old2new[split_node]
                simp_adj[a].add(b)
                simp_adj[b].add(a)
                if a < b:
                    chain_lengths[(a, b)] = seg_len
                else:
                    chain_lengths[(b, a)] = seg_len
                seg_start = i
                seg_len = 0.0

        # Final segment
        seg_len += float(np.linalg.norm(coords[chain_nodes[-1]] - coords[chain_nodes[-2]]))
        a = old2new[chain_nodes[seg_start]]
        b = old2new[end]
        if a != b:
            simp_adj[a].add(b)
            simp_adj[b].add(a)
            if a < b:
                chain_lengths[(a, b)] = seg_len
            else:
                chain_lengths[(b, a)] = seg_len

    for s in keep:
        for nb in adj[s]:
            if nb in keep:
                a, b = old2new[s], old2new[nb]
                if a < b:
                    simp_adj[a].add(b)
                    simp_adj[b].add(a)
                    chain_lengths[(a, b)] = float(np.linalg.norm(coords[s] - coords[nb]))
                continue
            _walk_chain(s, nb)

    # ── Second pass: Douglas-Peucker cleanup ────────────────────────────
    # Walk degree-2 chains in the simplified graph and remove redundant
    # curve points whose perpendicular deviation is below dp_epsilon_norm.
    if dp_epsilon_norm > 0 and len(simp_coords) > 3:
        # Build deg-2 chain nodes (map new-id → [prev, next])
        simp_deg2 = {}
        for a in range(len(simp_coords)):
            bs = [b for b in simp_adj.get(a, set()) if b != a]
            if len(bs) == 2:
                simp_deg2[a] = bs
            elif len(bs) == 1:
                pass  # endpoint, skip
        # Collect continuous deg-2 chains
        visited_dp: set[int] = set()
        to_remove: set[int] = set()
        for start in list(simp_deg2.keys()):
            if start in visited_dp:
                continue
            # Walk forward to find chain endpoints
            chain_dp = [start]
            visited_dp.add(start)
            # Forward
            prev, cur = start, simp_deg2[start][0]
            while cur in simp_deg2 and cur not in visited_dp:
                visited_dp.add(cur)
                chain_dp.append(cur)
                nxt = [n for n in simp_deg2[cur] if n != prev]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
            # Backward
            prev, cur = start, simp_deg2[start][1]
            while cur in simp_deg2 and cur not in visited_dp:
                visited_dp.add(cur)
                chain_dp.insert(0, cur)
                nxt = [n for n in simp_deg2[cur] if n != prev]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
            if len(chain_dp) < 2:
                continue
            # Walk chain: keep endpoints always; remove middle points if
            # perpendicular deviation is below epsilon AND segments are short.
            # Uses iterative DP: for each consecutive triple, compute deviation.
            changed = True
            while changed:
                changed = False
                i = 1
                while i < len(chain_dp) - 1:
                    p = chain_dp[i - 1]
                    c = chain_dp[i]
                    n = chain_dp[i + 1]
                    # Perpendicular distance from c to line(p, n)
                    v1 = simp_coords[n] - simp_coords[p]
                    v2 = simp_coords[c] - simp_coords[p]
                    v1_norm = float(np.linalg.norm(v1))
                    if v1_norm < 1e-10:
                        # Degenerate: co-linear points, remove middle
                        to_remove.add(c)
                        chain_dp.pop(i)
                        changed = True
                        continue
                    # Project v2 onto v1
                    t = np.dot(v2, v1) / (v1_norm * v1_norm)
                    t = max(0.0, min(1.0, t))
                    proj = simp_coords[p] + t * v1
                    dev = float(np.linalg.norm(simp_coords[c] - proj))
                    # Segment lengths
                    d1 = float(np.linalg.norm(simp_coords[c] - simp_coords[p]))
                    d2 = float(np.linalg.norm(simp_coords[n] - simp_coords[c]))
                    if dev < dp_epsilon_norm and d1 < max_seg_len_norm and d2 < max_seg_len_norm:
                        to_remove.add(c)
                        chain_dp.pop(i)
                        changed = True
                    else:
                        i += 1
        if to_remove:
            # Rebuild simp_coords, old2new, simp_adj, curve_newids
            keep_ids = sorted(set(range(len(simp_coords))) - to_remove)
            remap = {o: n for n, o in enumerate(keep_ids)}
            simp_coords = np.array([simp_coords[i] for i in keep_ids], dtype=np.float32)
            curve_newids = {remap[i] for i in curve_newids if i not in to_remove}
            new_adj: dict[int, set[int]] = defaultdict(set)
            new_chain_lengths: dict[tuple[int, int], float] = {}
            for a in list(simp_adj.keys()):
                if a in to_remove:
                    continue
                na = remap.get(a)
                if na is None:
                    continue
                for b in list(simp_adj.get(a, set())):
                    if b in to_remove:
                        continue
                    nb = remap.get(b)
                    if nb is not None and na != nb:
                        new_adj[na].add(nb)
                        new_adj[nb].add(na)
                        key = (min(na, nb), max(na, nb))
                        if key not in new_chain_lengths:
                            new_chain_lengths[key] = chain_lengths.get(
                                (min(a, b), max(a, b)),
                                float(np.linalg.norm(simp_coords[na] - simp_coords[nb])),
                            )
            simp_adj = new_adj
            chain_lengths = new_chain_lengths
            old2new = {k: remap[v] for k, v in old2new.items() if v in remap and v not in to_remove}

    simp_coords = np.array(simp_coords, dtype=np.float32)

    simp_edges_list = []
    for a, bset in simp_adj.items():
        for b in bset:
            if a < b:
                simp_edges_list.append([a, b])
    simp_edges = (
        np.array(simp_edges_list, dtype=np.int64).reshape(-1, 2)
        if simp_edges_list
        else np.empty((0, 2), dtype=np.int64)
    )

    simp_types_list = []
    for o in sorted(old2new.keys()):
        nid = old2new[o]
        sd = len(simp_adj.get(nid, []))
        if nid in curve_newids:
            simp_types_list.append(2)  # curve point (deg=2 but sharp turn)
        elif sd >= 3:
            simp_types_list.append(1)  # junction
        elif sd == 2:
            simp_types_list.append(0)  # waypoint (deg=2, straight)
        else:
            simp_types_list.append(4)  # endpoint

    return (
        simp_coords,
        simp_edges,
        np.array(simp_types_list, dtype=np.int64),
        old2new,
        chain_lengths,
    )


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
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(coords[i] - coords[j]) < merge_dist:
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


def estimate_local_spacing(
    coords: np.ndarray,
    edge_index: np.ndarray,
    map_size_m: float = 2000.0,
    n_samples_per_edge: int = 5,
) -> np.ndarray:
    """Estimate local road spacing s(x) for each edge in metres.

    For each edge, samples points along its polyline and measures
    the distance to the nearest *non-incident* edge, taking the
    median as the local spacing.

    Args:
        coords: (N, 2) node coordinates in [0, 1].
        edge_index: (E, 2) edge indices.
        map_size_m: map side length in metres (for converting to m).
        n_samples_per_edge: number of sample points per edge.

    Returns:
        (E,) array of spacing estimates in metres.
    """
    E = len(edge_index)
    if E < 2:
        return np.full(E, map_size_m * 0.1, dtype=np.float32)

    # Build per-edge adjacency for incident check
    adj: dict[int, set[int]] = defaultdict(set)
    for u, v in edge_index:
        adj[int(u)].add(int(v))
        adj[int(v)].add(int(u))

    spacing = np.full(E, map_size_m * 0.1, dtype=np.float32)

    for ei, (u, v) in enumerate(edge_index):
        u, v = int(u), int(v)
        p_u, p_v = coords[u], coords[v]
        edge_len = float(np.linalg.norm(p_v - p_u))
        if edge_len < 1e-8:
            continue

        samples = [p_u + t * (p_v - p_u) for t in np.linspace(0.1, 0.9, n_samples_per_edge)]
        dists = []
        for pt in samples:
            best_d = float("inf")
            for ej, (a, b) in enumerate(edge_index):
                if ej == ei:
                    continue
                a, b = int(a), int(b)
                # Skip incident edges
                if a == u or a == v or b == u or b == v:
                    continue
                p_a, p_b = coords[a], coords[b]
                ev = p_b - p_a
                el = float(np.linalg.norm(ev))
                if el < 1e-8:
                    continue
                t = float(np.clip(np.dot(pt - p_a, ev) / (el * el), 0, 1))
                cp = p_a + t * ev
                d = float(np.linalg.norm(pt - cp))
                if d < best_d:
                    best_d = d
            if best_d < float("inf"):
                dists.append(best_d)
        if dists:
            spacing[ei] = float(np.median(dists)) * map_size_m

    # Clamp to reasonable range
    spacing = np.clip(spacing, map_size_m * 0.01, map_size_m * 0.5)
    return spacing


# ─── Node type constants ───────────────────────────────────────────────

NT_WAYPOINT = 0  # deg=2, straight through
NT_JUNCTION = 1  # deg >= 3
NT_CURVE = 2  # deg=2, sharp turn (preserved shape point)
NT_ROUNDABOUT = 3  # part of a detected roundabout cycle
NT_ENDPOINT = 4  # deg <= 1


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
    import networkx as nx

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
    """Contract all degree-2 nodes → only intersections / endpoints remain.

    Returns ``(int_coords, int_edge_index, geometries)``.
    """
    if len(coords) < 5:
        return coords, edge_index, []
    import networkx as nx

    G = nx.Graph()
    for i, p in enumerate(coords):
        G.add_node(i, pos=p)
    for u, v in edge_index:
        G.add_edge(int(u), int(v))

    # --- Contract degree-2 nodes (no geometry tracking) ---
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
            G.add_edge(u, v)
            G.remove_node(n)
            changed = True
            break

    # --- Build geometry AFTER contraction: BFS on original adjacency ---
    orig_adj = {i: set() for i in range(len(coords))}
    for a, b in edge_index:
        ia, ib = int(a), int(b)
        orig_adj[ia].add(ib)
        orig_adj[ib].add(ia)

    from collections import deque

    def _trace(start, end):
        parent = {start: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur == end:
                break
            for nb in orig_adj[cur]:
                if nb not in parent:
                    parent[nb] = cur
                    q.append(nb)
        path = []
        cur = end
        while cur is not None:
            path.append(coords[cur])
            cur = parent.get(cur)
        return path[::-1]  # start \u2192 end

    nodes = sorted(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    c2 = np.array([G.nodes[n]["pos"] for n in nodes])
    e2, geoms = [], []
    for u, v in G.edges():
        e2.append([idx[u], idx[v]])
        geoms.append(np.array(_trace(u, v)))
    return c2, np.array(e2, dtype=np.int64), geoms


def classify_nodes(coords, edge_index, map_m, merge_dist_m=50.0, compressed=False):
    """Classify nodes (0=waypoint, 1=junction, 3=roundabout, 4=endpoint).

    When ``compressed=True`` the degree-2 check for roundabout detection
    is skipped (for intersection-level graphs).
    """
    import networkx as nx

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
        coords, edge_index, nt, map_size_m=map_m, max_cycle_size=12, skip_deg2_check=compressed
    )

    merged = set()
    for i in range(len(coords)):
        if i in merged or nt[i] != NT_JUNCTION:
            continue
        for j in range(i + 1, len(coords)):
            if j in merged or nt[j] != NT_JUNCTION:
                continue
            if np.linalg.norm(coords[i] - coords[j]) < merge_dist_m / map_m:
                merged.add(j)
                nt[j] = NT_WAYPOINT
    return nt

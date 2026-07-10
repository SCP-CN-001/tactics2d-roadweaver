"""
Graph utility operations: adjacency, components, chain simplification, node merging.

All functions operate on normalized [0, 1] graph coordinates unless noted.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


def adjacency_from_edges(edge_index: np.ndarray, N: int
                         ) -> Dict[int, List[int]]:
    """Build {node_id: [neighbor_ids]} from an (E, 2) edge array."""
    adj: Dict[int, List[int]] = defaultdict(list)
    for u, v in edge_index:
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))
    return adj


def find_components(adj: Dict[int, List[int]],
                    node_count: Optional[int] = None) -> List[Set[int]]:
    """Return a list of connected-component sets."""
    all_nodes = set(adj.keys())
    if node_count is not None:
        all_nodes.update(range(node_count))
    visited: Set[int] = set()
    comps: List[Set[int]] = []
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


def components_with_edges(coords: np.ndarray, edge_index: np.ndarray
                          ) -> List[Set[int]]:
    """Find connected components from graph data directly."""
    adj = adjacency_from_edges(edge_index, len(coords))
    return find_components(adj, len(coords))


def component_of_nodes(coords: np.ndarray, edge_index: np.ndarray
                       ) -> Dict[int, int]:
    """Return {node_id: component_id}."""
    comps = components_with_edges(coords, edge_index)
    comp_of: Dict[int, int] = {}
    for ci, cl in enumerate(comps):
        for n in cl:
            comp_of[n] = ci
    return comp_of


def endpoint_nodes(adj: Dict[int, List[int]]) -> List[int]:
    """Return list of node IDs with degree == 1."""
    return [i for i, nbrs in adj.items() if len(nbrs) == 1]


def simplify_chains(coords: np.ndarray, edge_index: np.ndarray,
                    force_keep: Optional[Set[int]] = None
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, int]]:
    """Collapse degree-2 chains into single edges.

    Returns (simp_coords, simp_edges, simp_types, old2new_map).
    ``simp_types``: 1 = junction (deg ≥ 3), 4 = endpoint (deg ≤ 1).
    """
    N = len(coords)
    adj = adjacency_from_edges(edge_index, N)

    deg = {i: len(adj[i]) for i in range(N)}

    # Keep nodes with degree != 2 (junctions & endpoints) plus force_keep
    keep: Set[int] = {i for i in range(N) if deg[i] != 2}
    if force_keep:
        keep.update(force_keep)
    if not keep:
        keep = set(range(N))  # fallback: keep everything

    klist = sorted(keep)
    old2new = {o: n for n, o in enumerate(klist)}
    simp_coords = np.array([coords[i] for i in klist], dtype=np.float32)

    # Build simplified adjacency by walking chains
    simp_adj: Dict[int, Set[int]] = defaultdict(set)
    for s in keep:
        for nb in adj[s]:
            if nb in keep:
                a, b = old2new[s], old2new[nb]
                if a < b:
                    simp_adj[a].add(b)
                    simp_adj[b].add(a)
                continue
            # Walk degree-2 chain: s → ... → end
            prev, cur = s, nb
            while cur not in keep:
                nxt = [n for n in adj[cur] if n != prev]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
            if cur in keep and cur != s:
                a, b = old2new[s], old2new[cur]
                if a < b:
                    simp_adj[a].add(b)
                    simp_adj[b].add(a)

    simp_edges_list = []
    for a, bset in simp_adj.items():
        for b in bset:
            if a < b:
                simp_edges_list.append([a, b])
    simp_edges = np.array(simp_edges_list, dtype=np.int64).reshape(-1, 2) \
                 if simp_edges_list else np.empty((0, 2), dtype=np.int64)

    # Assign types based on simplified degree
    simp_types_list = []
    for o in klist:
        sd = len(simp_adj.get(old2new[o], []))
        simp_types_list.append(1 if sd >= 2 else 4)  # junction vs endpoint

    return simp_coords, simp_edges, np.array(simp_types_list, dtype=np.int64), old2new


def merge_close_nodes(coords: np.ndarray, edge_index: np.ndarray,
                      node_types: np.ndarray,
                      closure_edges: np.ndarray,
                      merge_dist: float
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                 np.ndarray, Dict[int, int]]:
    """Merge nodes closer than *merge_dist* into centroids.

    Nodes in the same proximity cluster are replaced by their centroid.
    Self-loops from merging are removed.  Returns
    (merged_coords, merged_edges, merged_types, merged_closure_edges, old2new).
    """
    N = len(coords)
    if N == 0 or merge_dist <= 0:
        return coords, edge_index, node_types, closure_edges, {i: i for i in range(N)}

    # Build proximity graph
    prox_adj: Dict[int, Set[int]] = {i: set() for i in range(N)}
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(coords[i] - coords[j]) < merge_dist:
                prox_adj[i].add(j)
                prox_adj[j].add(i)

    # Connected components in proximity graph
    visited: Set[int] = set()
    clusters: List[Set[int]] = []
    for i in range(N):
        if i in visited:
            continue
        stack, cl = [i], set()
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            cl.add(n)
            for nb in prox_adj.get(n, []):
                if nb not in visited:
                    stack.append(nb)
        clusters.append(cl)

    if len(clusters) == N:
        return coords, edge_index, node_types, closure_edges, {i: i for i in range(N)}

    # Merge each cluster → centroid
    old2new: Dict[int, int] = {}
    new_coords_list: List[np.ndarray] = []
    new_types_list: List[int] = []
    for ci, cl in enumerate(clusters):
        for old_n in cl:
            old2new[old_n] = ci
        pts = np.array([coords[n] for n in cl])
        new_coords_list.append(pts.mean(axis=0))
        # Type precedence: junction > endpoint
        tset = {int(node_types[n]) for n in cl}
        new_types_list.append(1 if 1 in tset else 4)

    new_coords = np.array(new_coords_list, dtype=np.float32)
    new_types = np.array(new_types_list, dtype=np.int64)

    # Remap edges (deduplicate, discard self-loops)
    edge_set: Set[Tuple[int, int]] = set()
    for u, v in edge_index:
        nu, nv = old2new[int(u)], old2new[int(v)]
        if nu != nv:
            edge_set.add((nu, nv) if nu < nv else (nv, nu))
    new_edges = np.array(list(edge_set), dtype=np.int64).reshape(-1, 2) \
                if edge_set else np.empty((0, 2), dtype=np.int64)

    # Remap closure_edges
    ce_list: List[Tuple[int, int]] = []
    for a, b in closure_edges:
        na, nb = old2new.get(int(a)), old2new.get(int(b))
        if na is not None and nb is not None and na != nb:
            ce_list.append((na, nb) if na < nb else (nb, na))
    ce_set = set(ce_list)
    new_ce = np.array(list(ce_set), dtype=np.int64).reshape(-1, 2) \
             if ce_set else np.empty((0, 2), dtype=np.int64)

    return new_coords, new_edges, new_types, new_ce, old2new

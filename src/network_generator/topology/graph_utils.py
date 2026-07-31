"""Low-level graph helpers and node-type constants."""

from __future__ import annotations

from collections import defaultdict

import networkx as nx
import numpy as np


def build_nx(coords: np.ndarray, edge_index: np.ndarray) -> nx.Graph:
    """Build an ``nx.Graph`` with ``pos`` attributes from coords + edges."""
    G = nx.Graph()
    for i in range(len(coords)):
        G.add_node(i, pos=coords[i].copy())
    G.add_edges_from(edge_index)
    return G


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


def endpoint_nodes(adj: dict[int, list[int]]) -> list[int]:
    """Return list of node IDs with degree == 1."""
    return [i for i, nbrs in adj.items() if len(nbrs) == 1]


# ─── Node type constants ───────────────────────────────────────────────

NT_WAYPOINT = 0  # deg=2, straight through
NT_JUNCTION = 1  # deg >= 3
NT_CURVE = 2  # deg=2, sharp turn (preserved shape point)
NT_ROUNDABOUT = 3  # part of a detected roundabout cycle
NT_ENDPOINT = 4  # deg <= 1

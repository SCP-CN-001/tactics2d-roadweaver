"""
BFS-based deterministic graph-to-sequence ordering for skeleton graphs.

Strategy:
  1. Build undirected adjacency from edge_connect.
  2. Compute degree for each node.
  3. Root selection: highest-degree node; ties broken by distance to map center, then index.
  4. BFS traversal from root, sorting neighbours by bearing angle (or distance/index).
  5. For disconnected components, pick the highest-degree unvisited node as next root.
"""

from typing import List, Tuple

import numpy as np


def _compute_degrees(adj: np.ndarray) -> np.ndarray:
    """Return degree per node (undirected, no self-loops)."""
    return np.sum(adj > 0.5, axis=1)


def _bearing_between(a: np.ndarray, b: np.ndarray) -> float:
    """Bearing (radians) of vector a→b relative to positive x axis."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return np.arctan2(dy, dx)


class BFSOrdering:
    """
    Deterministic BFS ordering for skeleton graphs.

    Usage:
        orderer = BFSOrdering()
        order, inv_order = orderer.build_order(coords, edge_connect)
    """

    def __init__(self, neighbor_sort_by: str = "bearing"):
        """
        Args:
            neighbor_sort_by: 'bearing' | 'distance' | 'index'
        """
        assert neighbor_sort_by in ("bearing", "distance", "index")
        self.neighbor_sort_by = neighbor_sort_by

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_order(
        self,
        coords: np.ndarray,        # (N, 2)
        edge_connect: np.ndarray,  # (N, N)  float or bool  — undirected
    ) -> Tuple[List[int], List[int]]:
        """
        Return:
            order:         List[int] of node indices in traversal order (length N)
            inverse_order: List[int] where inv[i] = position of original node i in order
        """
        N = coords.shape[0]
        adj = (edge_connect > 0.5).astype(bool)
        # Ensure undirected
        adj = adj | adj.T
        np.fill_diagonal(adj, False)

        degrees = _compute_degrees(adj.astype(float))
        map_center = coords.mean(axis=0)
        visited = np.zeros(N, dtype=bool)
        order = []

        while len(order) < N:
            # Pick root among unvisited: highest degree, tie→closest to center, tie→lowest index
            unvisited_mask = ~visited
            if not unvisited_mask.any():
                break
            candidates = np.where(unvisited_mask)[0]
            # Score: primary = degree, secondary = -distance_to_center (negate so higher is better)
            dists = np.linalg.norm(coords[candidates] - map_center, axis=1)
            # Composite score: degree + 1/(1+dist) * 0.01 as tiebreaker
            scores = degrees[candidates] + 0.001 / (1.0 + dists)
            root = candidates[np.argmax(scores)]

            self._bfs_from_root(root, coords, adj, visited, order)

        # Build inverse mapping
        inverse_order = [0] * N
        for pos, idx in enumerate(order):
            inverse_order[idx] = pos

        return order, inverse_order

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _bfs_from_root(
        self,
        root: int,
        coords: np.ndarray,
        adj: np.ndarray,
        visited: np.ndarray,
        order: List[int],
    ):
        """BFS starting from root, appending visited nodes to order."""
        from collections import deque

        queue = deque()
        queue.append(root)
        visited[root] = True

        while queue:
            current = queue.popleft()
            order.append(current)

            # Unvisited neighbours
            nbrs = np.where(adj[current] & ~visited)[0]
            if len(nbrs) == 0:
                continue

            # Sort neighbours
            nbrs_list = self._sort_neighbors(current, nbrs, coords)
            for nbr in nbrs_list:
                if not visited[nbr]:
                    visited[nbr] = True
                    queue.append(nbr)

    def _sort_neighbors(
        self,
        current: int,
        nbrs: np.ndarray,
        coords: np.ndarray,
    ) -> List[int]:
        """Sort neighbour indices by the configured strategy."""
        if self.neighbor_sort_by == "bearing":
            bearings = [_bearing_between(coords[current], coords[n]) for n in nbrs]
            pairs = sorted(zip(bearings, nbrs), key=lambda x: x[0])
            return [n for _, n in pairs]
        elif self.neighbor_sort_by == "distance":
            dists = [np.linalg.norm(coords[current] - coords[n]) for n in nbrs]
            pairs = sorted(zip(dists, nbrs), key=lambda x: x[0])
            return [n for _, n in pairs]
        else:  # index
            return sorted(nbrs.tolist())

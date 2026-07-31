"""Raster field to skeleton graph converter implementation."""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.ndimage import binary_closing, binary_opening
from skimage.morphology import (
    dilation,
    remove_small_holes,
    remove_small_objects,
    skeletonize,
    square,
)


def _prune_graph(graph: dict, min_edge_len: float = 0.005, min_degree1_chain: int = 2) -> dict:
    """Remove short dead-end branches from a skeleton graph."""
    coords = graph["coords"]
    edges = graph["edge_index"]
    types = graph["node_types"]
    N = len(coords)
    if N < 3 or len(edges) < 2:
        return graph

    adj = {i: set() for i in range(N)}
    for i, j in edges:
        if i < N and j < N:
            adj[i].add(j)
            adj[j].add(i)

    changed = True
    while changed:
        changed = False
        to_remove = set()
        for i in range(N):
            if i in to_remove:
                continue
            if len(adj[i]) == 1:
                chain = [i]
                prev = i
                cur = list(adj[i])[0]
                chain.append(cur)
                while len(adj[cur]) == 2 and cur not in to_remove:
                    nbrs = [n for n in adj[cur] if n != prev]
                    if len(nbrs) != 1:
                        break
                    prev, cur = cur, nbrs[0]
                    chain.append(cur)
                chain_len = sum(
                    np.linalg.norm(coords[chain[k]] - coords[chain[k + 1]])
                    for k in range(len(chain) - 1)
                )
                if chain_len < min_edge_len and len(chain) >= min_degree1_chain:
                    for n in chain:
                        if n not in to_remove:
                            to_remove.add(n)
                            for nb in adj[n]:
                                if nb in adj:
                                    adj[nb].discard(n)
                            adj[n] = set()
                    changed = True

    if not to_remove:
        return graph
    keep = [i for i in range(N) if i not in to_remove]
    id_map = {old: new for new, old in enumerate(keep)}
    return {
        "coords": coords[keep],
        "edge_index": (
            np.array(
                [
                    [id_map[i], id_map[j]]
                    for i, j in edges
                    if i not in to_remove and j not in to_remove
                ],
                dtype=np.int64,
            )
            if edges
            else np.zeros((0, 2), dtype=np.int64)
        ),
        "node_types": types[keep],
    }


def _cleanup_field(
    road_binary: np.ndarray,
    opening_radius: int = 2,
    closing_radius: int = 2,
    min_obj_size: int = 64,
) -> np.ndarray:
    """Morphological cleanup for generated road fields."""
    if opening_radius > 0:
        road_binary = binary_opening(road_binary, structure=square(opening_radius * 2 + 1))
    if closing_radius > 0:
        road_binary = binary_closing(road_binary, structure=square(closing_radius * 2 + 1))
    if min_obj_size > 0:
        road_binary = remove_small_objects(road_binary, min_size=min_obj_size)
    return road_binary


def field_to_graph(
    road_prob: np.ndarray,
    road_threshold: float = 0.5,
    binary_center: np.ndarray = None,
    resolution: int = 256,
    prune_short_branches: bool = True,
    min_edge_len: float = 0.008,
    cleanup: bool = True,
    opening_radius: int = 2,
    closing_radius: int = 2,
) -> dict:
    """Convert a binary road mask to a skeleton graph.

    Every skeleton pixel becomes a graph node; all 8-connected edges are kept.
    The caller (e.g. ``growth.grow``) handles simplification downstream.
    """
    H, W = road_prob.shape

    if binary_center is not None and binary_center.max() > 0:
        road_binary = binary_center > 0
        road_binary = dilation(road_binary, square(2))
    else:
        road_binary = (road_prob > road_threshold).astype(bool)
        road_binary = remove_small_holes(road_binary, area_threshold=64)
        road_binary = remove_small_objects(road_binary, min_size=32)
        road_binary = dilation(road_binary, square(2))

    if cleanup:
        road_binary = _cleanup_field(
            road_binary,
            opening_radius=opening_radius,
            closing_radius=closing_radius,
            min_obj_size=64,
        )

    skel = skeletonize(road_binary)
    skel_coords = np.column_stack(np.where(skel > 0))
    if len(skel_coords) < 5:
        return {
            "coords": np.zeros((0, 2)),
            "edge_index": np.zeros((0, 2), dtype=np.int64),
            "node_types": np.zeros(0, dtype=np.int64),
        }

    lookup = {(r, c): i for i, (r, c) in enumerate(skel_coords)}
    G = nx.Graph()
    G.add_nodes_from(range(len(skel_coords)))
    for i, (r, c) in enumerate(skel_coords):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                j = lookup.get((r + dr, c + dc))
                if j is not None:
                    G.add_edge(i, j)

    G.remove_nodes_from(list(nx.isolates(G)))
    if len(G) < 2:
        return {
            "coords": np.zeros((0, 2)),
            "edge_index": np.zeros((0, 2), dtype=np.int64),
            "node_types": np.zeros(0, dtype=np.int64),
        }

    # All skeleton pixels become graph nodes; let the refiner simplify later
    node_list = list(G.nodes())
    id_map = {old: new for new, old in enumerate(node_list)}
    deg_dict = dict(G.degree())
    out_nodes = []
    out_types = []
    for old_id in node_list:
        r, c = skel_coords[old_id]
        out_nodes.append([c / resolution, r / resolution])
        d = deg_dict.get(old_id, 2)
        if d == 1:
            out_types.append(4)  # dead-end
        elif d == 2:
            out_types.append(0)  # waypoint
        else:
            out_types.append(1)  # junction

    out_edges = [[id_map[u], id_map[v]] for u, v in G.edges()]

    result = {
        "coords": np.array(out_nodes, dtype=np.float32),
        "edge_index": (
            np.array(out_edges, dtype=np.int64) if out_edges else np.zeros((0, 2), dtype=np.int64)
        ),
        "node_types": np.array(out_types, dtype=np.int64),
    }

    if prune_short_branches:
        result = _prune_graph(result, min_edge_len=min_edge_len)

    return result

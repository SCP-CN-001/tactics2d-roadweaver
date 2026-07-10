"""Binary road field → skeleton graph extraction."""
from typing import Dict, List, Optional, Tuple
import numpy as np
from skimage.morphology import skeletonize, dilation, square
from skimage.morphology import remove_small_objects, remove_small_holes
import networkx as nx
from .graph_cleanup import prune_graph, cleanup_field

def field_to_graph(
    road_prob: np.ndarray,
    junction_hm: np.ndarray,
    endpoint_hm: np.ndarray,
    road_threshold: float = 0.5,
    binary_center: np.ndarray = None,
    merge_distance: float = 0.025,
    resolution: int = 256,
    node_detection_threshold: float = 0.3,
    prune_short_branches: bool = True,
    min_edge_len: float = 0.008,
    cleanup: bool = True,
    opening_radius: int = 2,
    closing_radius: int = 2,
    keep_all_nodes: bool = True,
) -> Dict:
    """
    Convert a binary road mask to a skeleton graph.

    For binary centerline fields (recommended): use binary_center directly.
    For soft fields: threshold road_prob.

    When *keep_all_nodes* is True (default), every skeleton pixel becomes a
    graph node and all 8-connected edges are kept; no degree-2 simplification
    is performed.  The caller (e.g. ``graph_refiner``) handles simplification.
    This gives the refiner full freedom to decide how to merge and reconnect.

    When *keep_all_nodes* is False, the function performs the legacy
    junction-only simplification (paths between junctions become single edges).
    """
    from skimage.morphology import dilation, square, remove_small_objects, remove_small_holes
    from scipy.ndimage import binary_opening, binary_closing

    H, W = road_prob.shape

    if binary_center is not None and binary_center.max() > 0:
        road_binary = (binary_center > 0)
        road_binary = dilation(road_binary, square(2))
    else:
        road_binary = (road_prob > road_threshold).astype(bool)
        road_binary = remove_small_holes(road_binary, area_threshold=64)
        road_binary = remove_small_objects(road_binary, min_size=32)
        road_binary = dilation(road_binary, square(2))

    # Morphological cleanup (critical for generated fields)
    if cleanup:
        road_binary = cleanup_field(
            road_binary, opening_radius=opening_radius,
            closing_radius=closing_radius, min_obj_size=64)

    skel = skeletonize(road_binary)
    skel_coords = np.column_stack(np.where(skel > 0))
    if len(skel_coords) < 5:
        return {"coords": np.zeros((0, 2)), "edge_index": np.zeros((0, 2), dtype=np.int64),
                "node_types": np.zeros(0, dtype=np.int64)}

    # Build 8-connected graph
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

    # Remove isolated pixels
    G.remove_nodes_from(list(nx.isolates(G)))
    if len(G) < 2:
        return {"coords": np.zeros((0, 2)), "edge_index": np.zeros((0, 2), dtype=np.int64),
                "node_types": np.zeros(0, dtype=np.int64)}

    # ── keep_all_nodes mode: output ALL skeleton CCs, let refiner connect them ──
    if keep_all_nodes:
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
                out_types.append(4)   # dead-end / endpoint
            elif d == 2:
                out_types.append(0)   # skeleton waypoint
            else:
                out_types.append(1)   # junction

        out_edges = []
        for u, v in G.edges():
            out_edges.append([id_map[u], id_map[v]])

        return {
            "coords": np.array(out_nodes, dtype=np.float32),
            "edge_index": np.array(out_edges, dtype=np.int64) if out_edges else np.zeros((0, 2), dtype=np.int64),
            "node_types": np.array(out_types, dtype=np.int64),
        }

    # ── Legacy mode: keep only largest connected component ──
    components = list(nx.connected_components(G))
    largest = max(components, key=len)
    G = G.subgraph(largest).copy()
    deg_dict = dict(G.degree())
    junction_set = {n for n, d in deg_dict.items() if d != 2}
    if not junction_set:
        endpoints = [n for n, d in deg_dict.items() if d == 1]
        if len(endpoints) >= 2:
            path = nx.shortest_path(G, endpoints[0], endpoints[1])
            junction_set = {path[0], path[-1]}
        else:
            nodes_list = list(G.nodes())
            junction_set = {nodes_list[0], nodes_list[-1]}

    # Simplify: paths between junctions become edges
    simple = nx.Graph()
    jlist = sorted(junction_set)
    for n in jlist:
        r, c = skel_coords[n]
        nt = 1 if deg_dict.get(n, 0) > 2 else 2
        simple.add_node(n, r=r, c=c, type=nt)

    for s in jlist:
        seen = {s}
        stack = [(s, [s])]
        while stack:
            cur, path = stack.pop()
            if cur != s and cur in junction_set:
                if len(path) >= 2:
                    simple.add_edge(s, cur)
                continue
            for nb in G.neighbors(cur):
                if nb not in seen:
                    seen.add(nb)
                    stack.append((nb, path + [nb]))

    # Extract
    id_map = {}
    out_nodes = []
    for new_id, (old_id, data) in enumerate(simple.nodes(data=True)):
        r = data.get('r', 0)
        c = data.get('c', 0)
        out_nodes.append([c / resolution, r / resolution])
        id_map[old_id] = new_id

    out_edges = []
    for u, v in simple.edges():
        if u in id_map and v in id_map:
            out_edges.append([id_map[u], id_map[v]])

    graph_out = {
        "coords": np.array(out_nodes, dtype=np.float32),
        "edge_index": np.array(out_edges, dtype=np.int64) if out_edges else np.zeros((0, 2), dtype=np.int64),
        "node_types": np.array([data.get('type', 0) for _, data in simple.nodes(data=True)], dtype=np.int64),
    }

    # Optional pruning
    if prune_short_branches:
        graph_out = prune_graph(graph_out, min_edge_len=min_edge_len)

    return graph_out



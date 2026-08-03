# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Topological graph metrics implementation."""

from __future__ import annotations

import re

import networkx as nx
import numpy as np


def compute_topological_metrics(G: nx.Graph, ref_avg_degree: float | None = None) -> dict:
    """LCC ratio, dead-end ratio, avg degree, Δd̄.

    Args:
        G: Road network graph.
        ref_avg_degree: OSM reference average degree. If None, Δd̄ not computed.

    Returns:
        dict with keys: lcc, dead_end_ratio, avg_degree, (optional) delta_avg_degree,
                         node_count, edge_count
    """
    if G.number_of_nodes() == 0:
        return {
            "lcc": 0.0,
            "dead_end_ratio": 0.0,
            "avg_degree": 0.0,
            "node_count": 0,
            "edge_count": 0,
        }

    # Largest connected component ratio
    components = list(nx.connected_components(G))
    lcc_size = max(len(c) for c in components) if components else 0
    lcc = lcc_size / G.number_of_nodes()

    # Dead-end ratio (degree-1 nodes)
    degrees = dict(G.degree())
    dead_ends = sum(1 for d in degrees.values() if d == 1)
    dead_end_ratio = dead_ends / G.number_of_nodes()

    # Average degree
    avg_deg = sum(degrees.values()) / G.number_of_nodes()

    result = {
        "lcc": lcc,
        "dead_end_ratio": dead_end_ratio,
        "avg_degree": avg_deg,
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
    }

    if ref_avg_degree is not None:
        result["delta_avg_degree"] = abs(avg_deg - ref_avg_degree)

    return result


def compute_cycle_ratio(G: nx.Graph) -> dict:
    """Percentage of nodes that belong to at least one cycle.

    A node belongs to a cycle iff it survives k-core decomposition at k=2
    (iteratively remove degree-1 nodes).  This reflects the existence of
    closed road blocks (blocks / loops) in the generated road network.

    Args:
        G: Road network graph.

    Returns:
        dict with keys: cycle_ratio (fraction of nodes in cycles),
                        n_cycle_nodes, n_nodes
    """
    if G.number_of_nodes() == 0:
        return {"cycle_ratio": 0.0, "n_cycle_nodes": 0, "n_nodes": 0}

    core = nx.k_core(G, k=2)
    n_cycle = core.number_of_nodes()
    return {
        "cycle_ratio": n_cycle / G.number_of_nodes(),
        "n_cycle_nodes": n_cycle,
        "n_nodes": G.number_of_nodes(),
    }


def contract_degree2_nodes(G_raw: nx.Graph) -> nx.Graph:
    """Contract all degree-2 nodes to produce an intersection-based graph.

    In many procedural road network representations, roads are subdivided into
    many degree-2 nodes (waypoints along a straight segment).  This function
    removes those intermediate nodes::

        Original:  A ── X ── Y ── B    (A,B=junctions; X,Y=degree-2)
        Contracted: A ──────────── B

    Rules:
      - Nodes with degree != 2 are kept (junctions, dead-ends, multi-way).
      - Nodes with degree == 2 are removed; their two neighbors are connected
        directly.
      - Degree-0 nodes (isolated) are dropped.
      - Parallel edges are merged into a single edge.

    Args:
        G_raw: Input graph (may contain degree-2 intermediate nodes).

    Returns:
        Contracted graph where every node has degree != 2, unless degree-2
        nodes form a cycle or chain that starts/ends at degree-2.
    """
    G = nx.Graph()

    # Identify kept nodes
    keep = set()
    for node, deg in G_raw.degree():
        if deg != 2:
            keep.add(node)

    # Add kept nodes with attributes
    for node in keep:
        for k, v in G_raw.nodes[node].items():
            G.add_node(node, **{k: v})
        if node not in G:
            G.add_node(node)

    # For each kept node, BFS through degree-2 nodes
    visited_edges = set()

    for start in keep:
        for neighbor in G_raw.neighbors(start):
            edge_key = tuple(sorted((start, neighbor)))
            if edge_key in visited_edges:
                continue

            # Walk through degree-2 nodes
            prev, curr = start, neighbor
            path = [start, curr]

            while curr in G_raw and G_raw.degree(curr) == 2:
                next_nodes = [n for n in G_raw.neighbors(curr) if n != prev]
                if not next_nodes:
                    break
                nxt = next_nodes[0]
                prev, curr = curr, nxt
                path.append(curr)

            end = curr

            if start != end:
                edge_key = tuple(sorted((start, end)))
                if edge_key not in visited_edges:
                    G.add_edge(start, end)
                    visited_edges.add(edge_key)

    return G


def merge_close_nodes(G: nx.Graph, distance_threshold: float = 30.0) -> nx.Graph:
    """Merge nodes that belong to the same MetaDrive block (intersection/roundabout).

    MetaDrive node names encode block information::

        {sign}{block_num}{type}{sub}_{lane}_     e.g. "4X0_0_", "-7O2_1_"

    Types:
      - **X** = intersection block — all nodes with same ``{num}X`` prefix
        are part of the same intersection and are merged.
      - **O** = roundabout block — all nodes with same ``{num}O`` prefix
        are part of the same roundabout and are merged.
      - **C** = straight/curve block — kept as-is (degree-2 contracted later).

    The function relies on the ``coords`` attribute for positioning metadata.

    Args:
        G: Input graph (raw MetaDrive graph with ``coords`` attribute on nodes).
        distance_threshold: Ignored when block prefix is used; kept for API
            compatibility.

    Returns:
        Graph with intersection/roundabout nodes merged.
    """
    G = G.copy()

    # Pattern: (-?)(\d+)(type)(sub)_(lane)_
    # Merge all nodes sharing the same (block_num, type) for JUNCTION types:
    #   X/x = intersection,  O/o = roundabout
    #   T/t = T-junction,    R/r = ramp/roundabout-entry
    #   C/c, S/s = NOT merged (handled by degree-2 contraction)
    pat = re.compile(r"^(-?)(\d+)([XOTRxotr])(\d+)_(\d+)_$")

    # Group by (block_num, type) — e.g. ("4", "X") or ("7", "O")
    groups = {}
    for n in G.nodes:
        m = pat.match(str(n))
        if m:
            sign, num, btype, sub, lane = m.groups()
            key = (num, btype)  # e.g. ("4", "X")
            groups.setdefault(key, []).append(n)

    # Merge each group with > 1 node
    for key, cluster in groups.items():
        if len(cluster) <= 1:
            continue

        survivor = cluster[0]

        # Collect all neighbors outside the cluster
        all_neighbors = set()
        for n in cluster:
            for nb in G.neighbors(n):
                if nb not in cluster:
                    all_neighbors.add(nb)

        # Remove all but survivor
        for n in cluster[1:]:
            G.remove_node(n)

        # Connect survivor to all external neighbors
        for nb in all_neighbors:
            G.add_edge(survivor, nb)

    # Clean self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    return G


# Pattern matching MetaDrive road node names:  {sign}{num}{type}{sub}_{lane}_
# e.g.  4X0_0_  (intersection),  -7O2_1_  (roundabout),  3r1_0_  (ramp)
# This excludes decoration/edge nodes (">", "->", ">>", etc.)
METADRIVE_NODE_PATTERN = re.compile(r"^-?\d+[A-Za-z]\d+_\d+_$")


def is_metadrive_road_node(name: str) -> bool:
    """Check if a MetaDrive node name represents a real road node (not decoration)."""
    return bool(METADRIVE_NODE_PATTERN.match(str(name)))


def extract_intersection_graph(env) -> nx.Graph:
    """Extract a clean intersection-based road graph from a MetaDrive env.

    Pipeline:
      1. Build raw graph from NodeRoadNetwork (filters out decoration nodes)
      2. Tag node positions from lane polyline midpoints
      3. Merge spatially close nodes into intersection clusters (**30m**)
      4. Collapse remaining degree-2 waypoint nodes

    Args:
        env: MetaDriveEnv instance (already reset).

    Returns:
        nx.Graph with nodes ≈ real-world junctions, edges ≈ road segments.
    """
    # Build raw graph — filter out decoration nodes by naming pattern
    G_raw = nx.Graph()
    for sn, td in env.engine.current_map.road_network.graph.items():
        for en in td:
            if is_metadrive_road_node(sn) and is_metadrive_road_node(en):
                G_raw.add_edge(sn, en)

    # Tag positions
    rn = env.engine.current_map.road_network
    for sn in rn.graph:
        for en, lanes in rn.graph[sn].items():
            if not is_metadrive_road_node(sn) or not is_metadrive_road_node(en):
                continue
            if lanes:
                try:
                    pts = lanes[0].get_polyline(interval=1)
                    if len(pts) >= 2:
                        mid = (pts[0] + pts[-1]) / 2
                        G_raw.nodes[sn].setdefault("_coords", []).append(mid)
                        G_raw.nodes[en].setdefault("_coords", []).append(mid)
                except Exception:
                    pass

    for n in G_raw.nodes:
        if "_coords" in G_raw.nodes[n]:
            G_raw.nodes[n]["coords"] = np.mean(G_raw.nodes[n]["_coords"], axis=0).tolist()

    # Step 1: merge close nodes (intersections, roundabouts)
    G = merge_close_nodes(G_raw, distance_threshold=30.0)

    # Step 2: collapse degree-2 waypoints (repeat until stable)
    prev_n = -1
    while G.number_of_nodes() != prev_n:
        prev_n = G.number_of_nodes()
        G = contract_degree2_nodes(G)

    return G


def classify_scale(node_count: int) -> str:
    """Classify map size by intersection-node count.

    Bins:
        - **small**   ≤ 20
        - **medium**  21 … 40
        - **large**   > 40
    """
    if node_count <= 20:
        return "small"
    if node_count <= 40:
        return "medium"
    return "large"

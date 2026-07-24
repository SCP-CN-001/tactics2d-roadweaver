"""
Shared evaluation metrics for road network generation baselines.

All metric functions follow the same pattern:
  Input:  networkx.Graph (topology/route)  OR  list[np.ndarray] (geometry)
  Output: dict of scalar metric values

Usage:
    from eval.metrics import (
        compute_topological_metrics,
        compute_route_coverage,
        compute_chamfer_distance,
        compute_endpoint_alignment,
        compute_edge_smoothness,
        compute_self_intersection_rate,
        compute_edge_length_distribution,
        compute_subnode_uniformity,
        compute_node_angle_distribution,
        compute_all_geometric_metrics,
        save_results,
        save_tex_row,
        load_osm_reference_degree,
    )
"""

from __future__ import annotations

import json
import warnings
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
#  1. Topological Validity   (Table 1)
# ═══════════════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════════════
#  2. Route-Level Coverage   (Table 3)
# ═══════════════════════════════════════════════════════════════════════════


def compute_route_coverage(G: nx.Graph, n_pairs: int = 100, seed: int = 42) -> dict:
    """Random OD sampling → reachable ratio + avg shortest path length.

    Args:
        G: Road network graph.
        n_pairs: Number of random origin-destination pairs to test.
        seed: Random seed for reproducibility.

    Returns:
        dict with keys: reachable_ratio, avg_route_length, n_pairs_tested, n_reachable
    """
    if G.number_of_nodes() < 2:
        return {
            "reachable_ratio": 0.0,
            "avg_route_length": 0.0,
            "n_pairs_tested": 0,
            "n_reachable": 0,
        }

    nodes = list(G.nodes())
    rng = np.random.default_rng(seed)
    reachable = 0
    lengths = []

    for _ in range(n_pairs):
        u, v = rng.choice(nodes, size=2, replace=False)
        try:
            path = nx.shortest_path(G, source=u, target=v)
            reachable += 1
            lengths.append(len(path) - 1)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

    return {
        "reachable_ratio": reachable / n_pairs,
        "avg_route_length": float(np.mean(lengths)) if lengths else 0.0,
        "n_pairs_tested": n_pairs,
        "n_reachable": reachable,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  3. Geometric Validity — Chamfer & Endpoint   (Table 2)
# ═══════════════════════════════════════════════════════════════════════════


def compute_chamfer_distance(polylines_a: list[np.ndarray], polylines_b: list[np.ndarray]) -> float:
    """Bidirectional Chamfer Distance between two sets of road polylines.

    Args:
        polylines_a, polylines_b: Each is a list of (N_i, 2) point arrays.

    Returns:
        Chamfer distance scalar, or -1.0 if insufficient points.
    """
    if not polylines_a or not polylines_b:
        return -1.0

    from scipy.spatial import cKDTree

    all_a = np.concatenate(polylines_a, axis=0)
    all_b = np.concatenate(polylines_b, axis=0)

    if len(all_a) < 2 or len(all_b) < 2:
        return -1.0

    tree_a = cKDTree(all_a)
    tree_b = cKDTree(all_b)

    d_ab = tree_b.query(all_a, k=1)[0].mean()
    d_ba = tree_a.query(all_b, k=1)[0].mean()
    return float((d_ab + d_ba) / 2)


def compute_endpoint_alignment(polylines: list[np.ndarray]) -> float:
    """Average distance between connected segment endpoints (lower = better aligned).

    For each pair of polylines, measures the minimum distance between any
    combination of their endpoints.

    Args:
        polylines: List of (N_i, 2) point arrays.

    Returns:
        Mean endpoint distance, or 0.0 if < 2 polylines.
    """
    if len(polylines) < 2:
        return 0.0

    ep_dists = []
    for i in range(len(polylines)):
        for j in range(i + 1, len(polylines)):
            pts_i = polylines[i]
            pts_j = polylines[j]
            if len(pts_i) < 2 or len(pts_j) < 2:
                continue
            d = min(
                np.linalg.norm(pts_i[0] - pts_j[0]),
                np.linalg.norm(pts_i[0] - pts_j[-1]),
                np.linalg.norm(pts_i[-1] - pts_j[0]),
                np.linalg.norm(pts_i[-1] - pts_j[-1]),
            )
            ep_dists.append(d)

    return float(np.mean(ep_dists)) if ep_dists else 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  4. Geometric Validity — Edge Smoothness (curvature)
# ═══════════════════════════════════════════════════════════════════════════


def compute_edge_smoothness(polylines: list[np.ndarray]) -> dict:
    """Average turning angle (curvature) across all edge polylines.

    For each polyline with ≥3 points, computes the angle between consecutive
    segments.  A perfectly straight edge has 0° mean turning; a jagged edge
    has high turning.

    Returns:
        dict: mean_turning_angle_deg, std_turning_angle_deg, max_turning_angle_deg
    """
    all_angles = []
    for pts in polylines:
        if len(pts) < 3:
            continue
        vecs = pts[1:] - pts[:-1]
        norms = np.linalg.norm(vecs, axis=1)
        for i in range(len(vecs) - 1):
            if norms[i] < 1e-8 or norms[i + 1] < 1e-8:
                continue
            cos_a = np.dot(vecs[i], vecs[i + 1]) / (norms[i] * norms[i + 1])
            cos_a = np.clip(cos_a, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_a))
            all_angles.append(angle)

    if not all_angles:
        return {
            "mean_turning_angle_deg": 0.0,
            "std_turning_angle_deg": 0.0,
            "max_turning_angle_deg": 0.0,
            "n_angles": 0,
        }

    return {
        "mean_turning_angle_deg": float(np.mean(all_angles)),
        "std_turning_angle_deg": float(np.std(all_angles)),
        "max_turning_angle_deg": float(np.max(all_angles)),
        "n_angles": len(all_angles),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  5. Geometric Validity — Self-Intersection Rate
# ═══════════════════════════════════════════════════════════════════════════


def compute_self_intersection_rate(polylines: list[np.ndarray]) -> dict:
    """Check whether road segments intersect each other (optimized with AABB).

    Uses axis-aligned bounding box pre-checks to skip far-apart edge pairs,
    then does full segment-segment intersection tests only on candidates.

    Returns:
        dict: n_intersections, intersection_rate (per-edge-pair),
              has_intersection (bool)
    """
    # Build flat segment list with (bbox, polyline_id, seg_id, p1, p2)
    segments = []  # (xmin, xmax, ymin, ymax, i, si, p1, p2)
    for i, pts in enumerate(polylines):
        if len(pts) < 2:
            continue
        for si in range(len(pts) - 1):
            p1, p2 = pts[si], pts[si + 1]
            segments.append(
                (
                    min(p1[0], p2[0]),
                    max(p1[0], p2[0]),
                    min(p1[1], p2[1]),
                    max(p1[1], p2[1]),
                    i,
                    si,
                    p1,
                    p2,
                )
            )

    def _bbox_overlap(a, b):
        """Check AABB overlap."""
        return a[0] <= b[1] and a[1] >= b[0] and a[2] <= b[3] and a[3] >= b[2]

    def _orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def _on_segment(p, q, r):
        """Check if q lies on segment pr (collinear case)."""
        return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(
            p[1], r[1]
        )

    def _segments_intersect(a1, a2, b1, b2):
        # Skip if segments share an endpoint (adjacent road edges)
        if (
            np.linalg.norm(a1 - b1) < 1e-10
            or np.linalg.norm(a1 - b2) < 1e-10
            or np.linalg.norm(a2 - b1) < 1e-10
            or np.linalg.norm(a2 - b2) < 1e-10
        ):
            return False
        o1 = _orient(a1, a2, b1)
        o2 = _orient(a1, a2, b2)
        o3 = _orient(b1, b2, a1)
        o4 = _orient(b1, b2, a2)

        if o1 == 0 and _on_segment(a1, b1, a2):
            return True
        if o2 == 0 and _on_segment(a1, b2, a2):
            return True
        if o3 == 0 and _on_segment(b1, a1, b2):
            return True
        if o4 == 0 and _on_segment(b1, a2, b2):
            return True

        return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)

    n_intersections = 0
    total_pairs = 0

    for si in range(len(segments)):
        for sj in range(si + 1, len(segments)):
            sa, sb = segments[si], segments[sj]
            # Skip if same polyline (edges within same road don't count)
            if sa[4] == sb[4]:
                continue
            total_pairs += 1
            # AABB quick reject
            if not _bbox_overlap(sa, sb):
                continue
            if _segments_intersect(sa[6], sa[7], sb[6], sb[7]):
                n_intersections += 1

    return {
        "n_intersections": n_intersections,
        "intersection_rate": n_intersections / max(total_pairs, 1),
        "has_intersection": n_intersections > 0,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  6. Geometric Validity — Edge Length Distribution
# ═══════════════════════════════════════════════════════════════════════════


def compute_edge_length_distribution(polylines: list[np.ndarray]) -> dict:
    """Statistics of road edge lengths.

    Edge length = sum of consecutive subnode distances (approximate curve length).
    Note: these are in the model's normalized coordinate space.

    Returns:
        dict: mean_edge_length, std_edge_length, cv_edge_length (std/mean),
              min_edge_length, max_edge_length, total_road_length, n_edges,
              edge_length_skewness
    """
    lengths = []
    for pts in polylines:
        if len(pts) < 2:
            continue
        d = float(np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1)))
        lengths.append(d)

    if not lengths:
        return {
            k: 0.0
            for k in (
                "mean_edge_length",
                "std_edge_length",
                "cv_edge_length",
                "min_edge_length",
                "max_edge_length",
                "total_road_length",
                "edge_length_skewness",
                "skewness",
            )
        }

    lengths_arr = np.array(lengths)
    mean_l = float(np.mean(lengths_arr))
    std_l = float(np.std(lengths_arr))
    cv = std_l / max(mean_l, 1e-8)

    # Skewness
    if std_l > 1e-8:
        skew = float(np.mean((lengths_arr - mean_l) ** 3) / (std_l**3))
    else:
        skew = 0.0

    return {
        "mean_edge_length": mean_l,
        "std_edge_length": std_l,
        "cv_edge_length": cv,
        "min_edge_length": float(np.min(lengths_arr)),
        "max_edge_length": float(np.max(lengths_arr)),
        "total_road_length": float(np.sum(lengths_arr)),
        "skewness": skew,
        "n_edges": len(lengths),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  7. Geometric Validity — Subnode Uniformity
# ═══════════════════════════════════════════════════════════════════════════


def compute_subnode_uniformity(polylines: list[np.ndarray]) -> dict:
    """How evenly spaced are consecutive subnode points along each edge.

    A uniform edge has CV of spacing ≈ 0; clustered points near endpoints
    yield high CV.

    Returns:
        dict: mean_spacing_cv, n_edges_with_subnodes
    """
    spacings_cv = []
    for pts in polylines:
        if len(pts) < 3:
            continue
        dists = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        if np.mean(dists) > 1e-8:
            cv = float(np.std(dists) / np.mean(dists))
            spacings_cv.append(cv)

    if not spacings_cv:
        return {"mean_spacing_cv": 0.0, "n_edges": 0}

    return {"mean_spacing_cv": float(np.mean(spacings_cv)), "n_edges": len(spacings_cv)}


# ═══════════════════════════════════════════════════════════════════════════
#  8. Geometric Validity — Node Angle Distribution
# ═══════════════════════════════════════════════════════════════════════════


def compute_node_angle_distribution(G: nx.Graph) -> dict:
    """Angle between incident edges at each junction node (degree ≥ 3).

    For each node with ≥2 incident edges, compute the angles between
    consecutive edges when sorted by direction.  Returns distribution stats.

    NOTE: Requires node coordinate attribute 'coords' (2D array-like).
    Without it, returns zeros.

    Returns:
        dict: mean_angle_deg, std_angle_deg, n_junctions
    """
    all_angles = []
    for node, deg in G.degree():
        if deg < 2:
            continue
        # Get node coordinates
        coords_data = G.nodes[node].get("coords")
        if coords_data is None:
            continue

        node_pt = np.array(coords_data, dtype=np.float64)
        if node_pt.shape != (2,):
            continue

        # Get neighbor coordinates and compute edge vectors
        vecs = []
        for nb in G.neighbors(node):
            nb_coords = G.nodes[nb].get("coords")
            if nb_coords is not None:
                nb_pt = np.array(nb_coords, dtype=np.float64)
                vec = nb_pt - node_pt
                norm = np.linalg.norm(vec)
                if norm > 1e-8:
                    vecs.append(vec / norm)

        if len(vecs) < 2:
            continue

        # Sort vectors by angle
        angles = np.arctan2([v[1] for v in vecs], [v[0] for v in vecs])
        idx = np.argsort(angles)
        sorted_vecs = [vecs[i] for i in idx]

        # Compute angles between consecutive vectors
        for i in range(len(sorted_vecs)):
            j = (i + 1) % len(sorted_vecs)
            cos_a = float(np.dot(sorted_vecs[i], sorted_vecs[j]))
            cos_a = np.clip(cos_a, -1.0, 1.0)
            deg_angle = np.degrees(np.arccos(cos_a))
            all_angles.append(deg_angle)

    if not all_angles:
        return {"mean_angle_deg": 0.0, "std_angle_deg": 0.0, "n_junctions": 0}

    return {
        "mean_angle_deg": float(np.mean(all_angles)),
        "std_angle_deg": float(np.std(all_angles)),
        "n_junctions": len(all_angles),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Composite: all geometric metrics at once
# ═══════════════════════════════════════════════════════════════════════════


def compute_all_geometric_metrics(polylines: list[np.ndarray], G: nx.Graph | None = None) -> dict:
    """Run all geometric metrics and return a merged dict.

    Args:
        polylines: List of (N_i, 2) arrays (edge road geometry).
        G: Optional NetworkX graph with node 'coords' for angle analysis.

    Returns:
        Dict with all geometric metric values.
    """
    result = {}

    # Chamfer self (split in half)
    if len(polylines) >= 4:
        mid = len(polylines) // 2
        result["chamfer_self"] = compute_chamfer_distance(polylines[:mid], polylines[mid:])
    else:
        result["chamfer_self"] = -1.0

    result["endpoint_alignment"] = compute_endpoint_alignment(polylines)
    result.update(compute_edge_smoothness(polylines))
    result.update(compute_self_intersection_rate(polylines))
    result.update(compute_edge_length_distribution(polylines))
    result.update(compute_subnode_uniformity(polylines))

    if G is not None:
        result.update(compute_node_angle_distribution(G))

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Graph utilities
# ═══════════════════════════════════════════════════════════════════════════


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
    import re

    G = G.copy()

    # Pattern: (-?)(\d+)(type)(sub)_(lane)_
    # Merge all nodes sharing the same (block_num, type):
    #   X = intersection,   O = roundabout
    #   T = T-junction,     R = ramp/roundabout-entry
    #   C = straight/curve (NOT merged — contracted via degree-2 later)
    #   S = lane-split     (NOT merged — handled by contraction)
    pat = re.compile(r"^(-?)(\d+)([XOTR])(\d+)_(\d+)_$")

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


def extract_intersection_graph(env) -> nx.Graph:
    """Extract a clean intersection-based road graph from a MetaDrive env.

    Pipeline:
      1. Build raw graph from NodeRoadNetwork
      2. Tag node positions from lane polyline midpoints
      3. Merge spatially close nodes into intersection clusters (**30m**)
      4. Collapse remaining degree-2 waypoint nodes

    Args:
        env: MetaDriveEnv instance (already reset).

    Returns:
        nx.Graph with nodes ≈ real-world junctions, edges ≈ road segments.
    """
    # Build raw graph
    G_raw = nx.Graph()
    for sn, td in env.engine.current_map.road_network.graph.items():
        for en in td:
            if sn != "decoration" and en != "decoration":
                G_raw.add_edge(sn, en)

    # Tag positions
    rn = env.engine.current_map.road_network
    for sn in rn.graph:
        for en, lanes in rn.graph[sn].items():
            if sn == "decoration" or en == "decoration":
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


_REF_DEG_CACHE: float | None = None


def load_osm_reference_degree(data_path: str | Path | None = None, use_cache: bool = True) -> float:
    """Load average node degree from OSM training data after degree-2 contraction.

    The skeleton graphs are contracted (degree-2 road midpoints removed) so that
    the reference operates at the same intersection-level granularity as
    MetaDrive / RoadGen / HDMapGen evaluation graphs.

    The first call computes the value and caches it; subsequent calls return the
    cached value instantly.  The cache also persists to a JSON sidecar file.

    Falls back to Boeing (2019) global average of 2.84 if data unavailable.
    """
    global _REF_DEG_CACHE

    if use_cache and _REF_DEG_CACHE is not None:
        return _REF_DEG_CACHE

    # Try persistent cache file
    repo_root = Path(__file__).resolve().parent.parent
    cache_path = repo_root / "runtimes" / ".osm_ref_degree_contracted.json"
    if use_cache and cache_path.exists():
        try:
            import json

            _REF_DEG_CACHE = json.loads(cache_path.read_text())["ref_deg"]
            return _REF_DEG_CACHE
        except Exception:
            pass

    if data_path is None:
        data_path = repo_root / "data" / "urban_prior" / "5km" / "splits" / "train.parquet"

    # Compute from scratch — subsample to first 1000 graphs for speed;
    # 1000 is enough for a stable average (std < 0.02 across OSM samples).
    try:
        import json as _json

        import pandas as pd

        df = pd.read_parquet(data_path)
        degs = []
        n_processed = 0
        for g in df["skeleton_graph_json"]:
            if n_processed >= 1000:
                break
            graph = _json.loads(g)
            nodes = graph.get("nodes", [])
            edges = graph.get("edges", [])
            if not nodes:
                continue

            # Build graph from skeleton JSON
            G = nx.Graph()
            for n in nodes:
                nid = n.get("id")
                if nid is not None:
                    G.add_node(nid)
            for e in edges:
                u, v = e.get("source"), e.get("target")
                if u is not None and v is not None:
                    G.add_edge(u, v)

            # Contract degree-2 nodes to match intersection-level granularity
            G = contract_degree2_nodes(G)

            if G.number_of_nodes() > 0:
                degs.append(sum(d for _, d in G.degree()) / G.number_of_nodes())
            n_processed += 1

        result = float(np.mean(degs)) if degs else 2.84

        # Persist cache
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(_json.dumps({"ref_deg": result}))
        except Exception:
            pass

        _REF_DEG_CACHE = result
        return result
    except Exception:
        return 2.84


def save_results(output_dir: Path, name: str, agg: dict, per_map: list | None = None):
    """Save evaluation results to JSON.

    Args:
        output_dir: Output directory.
        name: Base filename (without .json).
        agg: Aggregated metrics dict.
        per_map: Optional list of per-map metric dicts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    data = {"aggregate": agg}
    if per_map is not None:
        data["per_map"] = per_map
    with open(output_dir / f"{name}.json", "w") as f:
        json.dump(data, f, indent=2, cls=_NumpyEncoder)


def save_tex_row(
    output_dir: Path, table_name: str, method: str, agg: dict, fields: list[tuple[str, str]]
):
    """Save a LaTeX table row (mean ± std for given fields).

    Args:
        output_dir: Output directory for paper_tables/ subfolder.
        table_name: LaTeX filename (without .tex).
        method: Baseline name (e.g. "MetaDrive", "HDMapGen").
        agg: Dict with '{key}_mean' and '{key}_std' entries.
        fields: List of (key, format_spec) like ('lcc', '.3f').
    """
    cols = [method]
    for key, fmt in fields:
        mean = agg.get(f"{key}_mean", 0)
        std = agg.get(f"{key}_std", 0)
        cols.append(f"{mean:{fmt}} $\\pm$ {std:{fmt}}")

    row = " & ".join(cols) + " \\\\\n\\hline"

    tex_dir = output_dir / "paper_tables"
    tex_dir.mkdir(parents=True, exist_ok=True)
    with open(tex_dir / f"{table_name}.tex", "w") as f:
        f.write(row + "\n")


def print_results_table(title: str, results: dict, fields: list[tuple[str, str, str]]):
    """Pretty-print a results table.

    Args:
        title: Section title.
        results: Dict with '{key}_mean' and '{key}_std'.
        fields: List of (key, display_name, format_spec) like ('lcc', 'LCC', '.4f').
    """
    print(f"\n  Results:")
    for key, name, fmt in fields:
        mean = results.get(f"{key}_mean", 0)
        std = results.get(f"{key}_std", 0)
        print(f"    {name:20s}  {mean:{fmt}} ± {std:{fmt}}")


# ═══════════════════════════════════════════════════════════════════════════
#  Scale classification
# ═══════════════════════════════════════════════════════════════════════════


def classify_scale(node_count: int) -> str:
    """Classify map size by intersection-node count.

    Bins (matching the paper convention):
        - **small**   ≤ 15
        - **medium**  16 … 30
        - **large**   > 30
    """
    if node_count <= 15:
        return "small"
    if node_count <= 30:
        return "medium"
    return "large"


# ═══════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════════════


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

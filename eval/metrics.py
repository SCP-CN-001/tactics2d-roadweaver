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
        load_osm_reference_degree,
        monitor_resources,
        load_csv_keys,
        append_csv_row,
    )
"""

from __future__ import annotations

import csv
import json
import re
import warnings
from collections import defaultdict
from contextlib import contextmanager
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


def compute_chamfer_leave_one_out(polylines: list[np.ndarray]) -> dict:
    """Leave-one-out Chamfer in normalized [0,1]² coordinate space.

    For each polyline, computes the bidirectional Chamfer distance between
    that polyline's points and the point cloud of **all other** polylines.
    This replaces the old ``chamfer_self`` (random split in half) with a
    deterministic, more stable self-consistency measure.

    **Normalisation:** All coordinates are linearly scaled to [0,1]² based on
    the bounding box of the full polyline set.  This makes Chamfer values
    comparable across baselines regardless of native coordinate space
    (MetaDrive meters, RoadGen units, HDMapGen normalised, etc.).

    Returns:
        dict:
            chamfer_loo       — mean leave-one-out Chamfer over all polylines
            chamfer_loo_std   — standard deviation across polylines
            chamfer_loo_max   — worst-case (max) per-polyline Chamfer
            n_polylines       — number of polylines evaluated
    """
    if len(polylines) < 2:
        return {
            "chamfer_loo": 0.0,
            "chamfer_loo_std": 0.0,
            "chamfer_loo_max": 0.0,
            "n_polylines": len(polylines),
        }

    from scipy.spatial import cKDTree

    # ── Normalise all points to [0,1]² ─────────────────────────────────
    all_pts = np.vstack(polylines)
    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)
    size = max(x_max - x_min, y_max - y_min, 1.0)
    offset = np.array([x_min, y_min])

    # Pre-normalise every polyline
    normed = [(pts - offset) / size for pts in polylines]

    per_poly = []
    for i, pts_i in enumerate(normed):
        # Concatenate all OTHER polylines into one point cloud
        others = [normed[j] for j in range(len(normed)) if j != i]
        if not others:
            continue
        cloud = np.vstack(others)

        tree = cKDTree(cloud)
        d_i = tree.query(pts_i, k=1)[0].mean()
        per_poly.append(d_i)

    arr = np.array(per_poly)
    return {
        "chamfer_loo": float(np.mean(arr)),
        "chamfer_loo_std": float(np.std(arr)),
        "chamfer_loo_max": float(np.max(arr)),
        "n_polylines": len(per_poly),
    }


def compute_endpoint_alignment(
    polylines: list[np.ndarray], G: nx.Graph | None = None, merge_distance: float | None = None
) -> dict:
    """Road segment endpoint alignment at junctions (lower = better connected).

    Uses one of two strategies depending on whether *G* is provided:

    1. **Graph-guided (recommended)** — with G (``coords`` attribute required):
       For each graph node (junction), find all polyline endpoints that are close
       to that node (within *merge_distance*).  Measure how far those endpoints
       scatter around the node::
           per-node score = mean(|endpoint_i − node_coord|)
       This directly measures "do road segments actually meet at junctions".

    2. **Fallback (no graph)** — original all-pairs approach:
       For every pair of polylines, minimum distance between any combination of
       their endpoints.  This penalises even intentionally separate roads.

    Args:
        polylines: List of (N_i, 2) point arrays.
        G: Optional graph with node attribute ``coords`` (2D).  Uses graph
           connectivity to measure alignment only at junctions.
        merge_distance: Maximum distance from a node for an endpoint to be
                        considered "at that junction".  Auto-computed as 5 %
                        of the map extent when ``None``.

    Returns:
        dict with keys:
          alignment_mean — average endpoint→node distance across all junctions
          alignment_p50, alignment_p90 — percentiles for robust reporting
          n_junctions — number of junctions evaluated
          (falls back to a single scalar ``endpoint_alignment`` when no graph)
    """
    # ── Auto merge_distance ──────────────────────────────────────────────
    if merge_distance is None and polylines and G is not None:
        all_pts = np.vstack(polylines)
        extent = float(max(all_pts.max(axis=0) - all_pts.min(axis=0)))
        merge_distance = max(extent * 0.05, 1.0)  # 5 %, floor 1 unit

    # ── Graph-guided: measure endpoint scatter at each junction node ─────
    if G is not None and merge_distance is not None and merge_distance > 0:
        # Build list of all polyline endpoints
        endpoints = []  # (pos, is_start, poly_id)
        for pi, pts in enumerate(polylines):
            if len(pts) < 2:
                continue
            endpoints.append((np.array(pts[0], dtype=np.float64), True, pi))
            endpoints.append((np.array(pts[-1], dtype=np.float64), False, pi))

        if not endpoints:
            return {
                "alignment_mean": 0.0,
                "alignment_p50": 0.0,
                "alignment_p90": 0.0,
                "n_junctions": 0,
            }

        per_node_dists = []
        for node, deg in G.degree():
            if deg < 2:
                continue
            nc = G.nodes[node].get("coords")
            if nc is None:
                continue
            node_pt = np.array(nc, dtype=np.float64)

            # Find endpoints near this node
            near_dists = []
            for ep_pos, _is_start, _pi in endpoints:
                d = float(np.linalg.norm(ep_pos - node_pt))
                if d <= merge_distance:
                    near_dists.append(d)

            if len(near_dists) >= 2:
                per_node_dists.extend(near_dists)

        if not per_node_dists:
            return {
                "alignment_mean": 0.0,
                "alignment_p50": 0.0,
                "alignment_p90": 0.0,
                "n_junctions": 0,
            }

        arr = np.array(per_node_dists)
        return {
            "alignment_mean": float(np.mean(arr)),
            "alignment_p50": float(np.median(arr)),
            "alignment_p90": float(np.percentile(arr, 90)),
            "n_junctions": len(per_node_dists),
        }

    # ── Fallback: original all-pairs behaviour (kept for compatibility) ───
    if len(polylines) < 2:
        return {"endpoint_alignment": 0.0}

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

    mean_d = float(np.mean(ep_dists)) if ep_dists else 0.0
    return {"endpoint_alignment": mean_d}


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


def _segments_min_distance(p1, p2, q1, q2, n_samples: int = 10) -> float:
    """Approximate minimum Euclidean distance between two 2D line segments.

    Samples *n_samples* equally-spaced points along each segment and returns
    the minimum pairwise distance.  This is a cheap approximation to the true
    segment-segment distance but is sufficient for near-miss detection in
    road networks.
    """
    ts = np.linspace(0.0, 1.0, n_samples)
    pts_a = p1[np.newaxis, :] + ts[:, np.newaxis] * (p2 - p1)[np.newaxis, :]
    pts_b = q1[np.newaxis, :] + ts[:, np.newaxis] * (q2 - q1)[np.newaxis, :]
    diffs = pts_a[:, np.newaxis, :] - pts_b[np.newaxis, :, :]  # (n, n, 2)
    dists = np.sqrt(np.sum(diffs**2, axis=-1))
    return float(dists.min())


def compute_self_intersection_rate(polylines: list[np.ndarray], epsilon: float = 0.0) -> dict:
    """Check whether road segments intersect or nearly-miss each other.

    Uses axis-aligned bounding box pre-checks to skip far-apart edge pairs,
    then does full segment-segment intersection tests only on candidates.

    Args:
        polylines: List of (N_i, 2) point arrays.
        epsilon: Distance tolerance.  When > 0, any two segments whose minimum
                 separation is below *epsilon* count as intersecting (near-miss
                 detection).  When 0.0 (default), only exact geometric
                 intersections are counted.

    Returns:
        dict with keys:
            n_intersections      — number of crossing segment pairs
            intersection_rate    — n_intersections / n_cross_pairs
            n_cross_pairs        — total cross-polyline segment pairs checked
            n_collinear_overlaps — collinear overlapping segment pairs
            has_intersection     — bool, whether any intersection exists
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
        return a[0] <= b[1] and a[1] >= b[0] and a[2] <= b[3] and a[3] >= b[2]

    def _orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def _on_segment(p, q, r):
        return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(
            p[1], r[1]
        )

    def _segments_cross(a1, a2, b1, b2):
        """Check if two segments geometrically cross (strict intersection)."""
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

    def _is_collinear_overlap(p1, p2, q1, q2) -> bool:
        """Check if two segments are collinear and overlap."""
        v1 = p2 - p1
        v2 = q2 - q1
        # Cross product near zero → collinear
        if abs(np.cross(v1, v2)) > 1e-6:
            return False
        # Project both segments onto the direction of v1
        axis = v1 / (np.linalg.norm(v1) + 1e-12)
        p_proj = np.array([0.0, np.dot(p2 - p1, axis)])
        q_proj = np.array([np.dot(q1 - p1, axis), np.dot(q2 - p1, axis)])
        lo = max(p_proj.min(), q_proj.min())
        hi = min(p_proj.max(), q_proj.max())
        return hi > lo + 1e-6

    n_intersections = 0
    n_collinear = 0
    total_pairs = 0
    use_epsilon = epsilon > 0.0

    for si in range(len(segments)):
        for sj in range(si + 1, len(segments)):
            sa, sb = segments[si], segments[sj]
            if sa[4] == sb[4]:  # same polyline
                continue
            total_pairs += 1
            if not _bbox_overlap(sa, sb) and not use_epsilon:
                continue

            p1, p2 = sa[6], sa[7]
            q1, q2 = sb[6], sb[7]

            # Shared-endpoint check (adjacent edges at a junction)
            if _segments_min_distance(p1, p2, q1, q2, n_samples=10) < 1e-10:
                continue

            if use_epsilon:
                # Soft near-miss detection
                d = _segments_min_distance(p1, p2, q1, q2, n_samples=20)
                if d < epsilon:
                    if _is_collinear_overlap(p1, p2, q1, q2):
                        n_collinear += 1
                    else:
                        n_intersections += 1
            else:
                # Exact geometric intersection (backward-compatible)
                if _segments_cross(p1, p2, q1, q2):
                    n_intersections += 1

    return {
        "n_intersections": n_intersections,
        "intersection_rate": n_intersections / max(total_pairs, 1),
        "n_cross_pairs": total_pairs,
        "n_collinear_overlaps": n_collinear,
        "has_intersection": n_intersections > 0,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  6. Geometric Validity — Edge Length Distribution
# ═══════════════════════════════════════════════════════════════════════════


def compute_edge_length_distribution(polylines: list[np.ndarray]) -> dict:
    """Statistics of road edge lengths.

    Edge length = sum of consecutive subnode distances (approximate curve length).

    Returns:
        dict: mean_edge_length, std_edge_length, cv_edge_length (std/mean),
              min_edge_length, max_edge_length, total_road_length, n_edges,
              skewness
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
                "skewness",
                "n_edges",
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

    For each node with ≥3 incident edges, sorts the edge vectors by direction
    and computes angles between consecutive pairs.  Results are grouped by
    junction type (3-way, 4-way, 5+) so the distribution reflects road network
    design rather than mixing fundamentally different junction geometries.

    NOTE: Requires node coordinate attribute ``coords`` (2D array-like).
    Nodes without it are silently skipped (count reported in
    ``n_missing_coords``).

    Returns:
        dict with keys:
            mean_angle_deg    — overall mean angle across all junctions
            std_angle_deg     — overall standard deviation
            n_junctions       — total number of angle measurements
            n_missing_coords  — graph nodes skipped for missing coordinates
            angle_3way_mean   — mean angle at 3-way junctions
            angle_3way_std    — std of angles at 3-way junctions
            n_3way            — number of angle measurements at 3-way junctions
            angle_4way_mean   — mean angle at 4-way junctions
            angle_4way_std
            n_4way
            angle_5plus_mean  — mean angle at 5+ way junctions
            angle_5plus_std
            n_5plus
    """
    all_angles: list[float] = []
    angles_by_type: dict[str, list[float]] = {"3way": [], "4way": [], "5plus": []}

    missing_coords = 0

    for node, deg in G.degree():
        if deg < 3:  # only real junctions
            continue
        coords_data = G.nodes[node].get("coords")
        if coords_data is None:
            missing_coords += 1
            continue

        node_pt = np.array(coords_data, dtype=np.float64)
        if node_pt.shape != (2,):
            continue

        # Collect normalized edge vectors
        vecs: list[np.ndarray] = []
        for nb in G.neighbors(node):
            nb_coords = G.nodes[nb].get("coords")
            if nb_coords is not None:
                nb_pt = np.array(nb_coords, dtype=np.float64)
                vec = nb_pt - node_pt
                nrm = np.linalg.norm(vec)
                if nrm > 1e-8:
                    vecs.append(vec / nrm)

        if len(vecs) < 2:
            continue

        # Sort by direction angle
        angles_rad = np.arctan2([v[1] for v in vecs], [v[0] for v in vecs])
        idx = np.argsort(angles_rad)
        sorted_vecs = [vecs[i] for i in idx]

        # Angles between consecutive pairs (wrap-around)
        node_angles: list[float] = []
        for i in range(len(sorted_vecs)):
            j = (i + 1) % len(sorted_vecs)
            cos_a = float(np.dot(sorted_vecs[i], sorted_vecs[j]))
            cos_a = np.clip(cos_a, -1.0, 1.0)
            node_angles.append(float(np.degrees(np.arccos(cos_a))))

        all_angles.extend(node_angles)

        # Group by junction type
        type_key: str = "3way"
        if deg >= 5:
            type_key = "5plus"
        elif deg == 4:
            type_key = "4way"
        angles_by_type[type_key].extend(node_angles)

    if not all_angles:
        return {
            "mean_angle_deg": 0.0,
            "std_angle_deg": 0.0,
            "n_junctions": 0,
            "n_missing_coords": missing_coords,
            "angle_3way_mean": 0.0,
            "angle_3way_std": 0.0,
            "n_3way": 0,
            "angle_4way_mean": 0.0,
            "angle_4way_std": 0.0,
            "n_4way": 0,
            "angle_5plus_mean": 0.0,
            "angle_5plus_std": 0.0,
            "n_5plus": 0,
        }

    result: dict[str, float | int] = {
        "mean_angle_deg": float(np.mean(all_angles)),
        "std_angle_deg": float(np.std(all_angles)),
        "n_junctions": len(all_angles),
        "n_missing_coords": missing_coords,
    }

    for key, vals in angles_by_type.items():
        if vals:
            result[f"angle_{key}_mean"] = float(np.mean(vals))
            result[f"angle_{key}_std"] = float(np.std(vals))
            result[f"n_{key}"] = len(vals)
        else:
            result[f"angle_{key}_mean"] = 0.0
            result[f"angle_{key}_std"] = 0.0
            result[f"n_{key}"] = 0

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Composite: all geometric metrics at once
# ═══════════════════════════════════════════════════════════════════════════


def compute_all_geometric_metrics(
    polylines: list[np.ndarray], G: nx.Graph | None = None, self_intersect_epsilon: float = 0.0
) -> dict:
    """Run all geometric metrics and return a merged dict.

    Args:
        polylines: List of (N_i, 2) arrays (edge road geometry).
        G: Optional NetworkX graph with node 'coords' for angle analysis and
           graph-guided endpoint alignment.
        self_intersect_epsilon: Distance tolerance for near-miss intersection
            detection.  Pass > 0 (e.g. 1.0 in meter-scale coordinates) to
            detect near-misses as well as exact crossings.

    Returns:
        Dict with all geometric metric values.
    """
    result: dict = {}

    # Leave-one-out Chamfer in normalised [0,1]² space
    # (replaces old chamfer_self that split polylines arbitrarily in half)
    result.update(compute_chamfer_leave_one_out(polylines))

    # Endpoint alignment — graph-guided when G available
    ep_result = compute_endpoint_alignment(polylines, G=G)
    if "alignment_mean" in ep_result:
        result["endpoint_alignment"] = ep_result["alignment_mean"]
        result["endpoint_alignment_p50"] = ep_result["alignment_p50"]
        result["endpoint_alignment_p90"] = ep_result["alignment_p90"]
        result["n_endpoint_junctions"] = ep_result["n_junctions"]
    else:
        result["endpoint_alignment"] = ep_result.get("endpoint_alignment", 0.0)

    result.update(compute_edge_smoothness(polylines))
    result.update(compute_self_intersection_rate(polylines, epsilon=self_intersect_epsilon))
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
    """Save per-map evaluation results to CSV + aggregate summary.

    Writes two files:
      * ``{name}.csv``        — one row per map
      * ``{name}_summary.csv`` — aggregate (mean ± std per metric)

    Args:
        output_dir: Output directory.
        name: Base filename.
        agg: Aggregated metrics dict (keys like ``lcc_mean``, ``lcc_std``, …).
        per_map: List of per-map metric dicts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if per_map:
        # Collect all keys across all maps
        all_keys: list[str] = []
        seen = set()
        for m in per_map:
            for k in m:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

        # Normalise: node_count/num_nodes → n_nodes, edge_count → n_edges
        col_map = {"node_count": "n_nodes", "num_nodes": "n_nodes", "edge_count": "n_edges"}
        cols = [col_map.get(k, k) for k in all_keys]

        with open(output_dir / f"{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for m in per_map:
                row = []
                for k in all_keys:
                    v = m.get(k, "")
                    if isinstance(v, (np.floating,)):
                        v = float(v)
                    elif isinstance(v, (np.integer,)):
                        v = int(v)
                    row.append(v)
                w.writerow(row)

    # Summary (aggregate)
    if agg:
        with open(output_dir / f"{name}_summary.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "mean", "std"])
            for k, v in agg.items():
                if k.endswith("_mean"):
                    base = k[:-5]
                    std_val = agg.get(f"{base}_std", "")
                    w.writerow([base, v, std_val])
                elif k in ("n_maps", "avg_generation_time", "osm_reference_avg_degree"):
                    w.writerow([k, v, ""])


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
#  System resource tracking
# ═══════════════════════════════════════════════════════════════════════════


def get_resource_stats() -> dict[str, float]:
    """Return current CPU% (0-100), memory MB, and GPU memory MB."""
    cpu = mem = gpu = 0.0
    try:
        import psutil

        p = psutil.Process()
        cpu = p.cpu_percent(interval=0)
        mem = p.memory_info().rss / 1024**2
    except Exception:
        pass
    try:
        import subprocess

        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        gpu = float(r.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return {"cpu_percent": cpu, "mem_mb": mem, "gpu_mem_mb": gpu}


@contextmanager
def monitor_resources(interval: float = 0.5):
    """Context manager tracking peak CPU%, memory MB, GPU memory MB.

    Spawns a daemon thread that samples every *interval* seconds during
    the wrapped block and returns a dict with peak values on exit.

    Usage::

        with monitor_resources() as peaks:
            generate_map(...)
        print(peaks["cpu_peak"], peaks["mem_peak_mb"], peaks["gpu_peak_mb"])
    """
    import threading
    import time as _time

    peaks = {"cpu_peak": 0.0, "mem_peak_mb": 0.0, "gpu_peak_mb": 0.0}
    lock = threading.Lock()
    running = True

    def _sample():
        proc = None
        try:
            import psutil

            proc = psutil.Process()
            proc.cpu_percent(interval=0)  # warm-up
        except ImportError:
            pass

        while running:
            _time.sleep(interval)
            cpu = mem = 0.0
            if proc is not None:
                try:
                    cpu = proc.cpu_percent(interval=0)
                    mem = proc.memory_info().rss / 1024**2
                except Exception:
                    pass
            gpu = 0.0
            try:
                import subprocess

                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                gpu = float(r.stdout.strip().split("\n")[0])
            except Exception:
                pass
            with lock:
                peaks["cpu_peak"] = max(peaks["cpu_peak"], cpu)
                peaks["mem_peak_mb"] = max(peaks["mem_peak_mb"], mem)
                peaks["gpu_peak_mb"] = max(peaks["gpu_peak_mb"], gpu)

    t = threading.Thread(target=_sample, daemon=True)
    t.start()
    try:
        yield peaks
    finally:
        running = False
        t.join(timeout=3)


# ═══════════════════════════════════════════════════════════════════════════
#  CSV resume helpers
# ═══════════════════════════════════════════════════════════════════════════


def load_csv_keys(csv_path: Path, key_cols: list[str]) -> set[str]:
    """Load existing CSV and return set of ``'|'.join(key_cols)`` seen keys.

    Used for resume: before generating a map, check if
    ``f"{row[col1]}|{row[col2]}"`` is already in the CSV.  If yes, skip it.
    """
    if not csv_path.exists():
        return set()
    seen: set[str] = set()
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parts = [str(row.get(c, "")).strip() for c in key_cols]
                seen.add("|".join(parts))
    except Exception:
        pass
    return seen


def append_csv_row(csv_path: Path, columns: list[str], row: dict):
    """Append a single row to CSV.  Writes header only if file is new.

    Args:
        csv_path: Path to the CSV file (append mode).
        columns: Ordered list of column names.
        row: Dict of column → value for this row.
    """
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(columns)
        w.writerow([row.get(c, "") for c in columns])


def save_system_info(output_dir: Path, label: str, init_stats: dict, peak_stats: dict, n_maps: int):
    """Save system resource info to ``system.csv``."""
    out = output_dir / "system.csv"
    exists = out.exists()
    with open(out, "a" if exists else "w", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(
                [
                    "baseline",
                    "n_maps",
                    "cpu_init",
                    "cpu_peak",
                    "mem_init_mb",
                    "mem_peak_mb",
                    "gpu_init_mb",
                    "gpu_peak_mb",
                ]
            )
        w.writerow(
            [
                label,
                n_maps,
                init_stats.get("cpu_percent", ""),
                peak_stats.get("cpu_percent", ""),
                init_stats.get("mem_mb", ""),
                peak_stats.get("mem_mb", ""),
                init_stats.get("gpu_mem_mb", ""),
                peak_stats.get("gpu_mem_mb", ""),
            ]
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Scale classification
# ═══════════════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

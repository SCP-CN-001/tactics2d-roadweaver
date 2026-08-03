# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Geometric road network metrics implementation."""

from __future__ import annotations

import networkx as nx
import numpy as np


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

    For each polyline, computes the **bidirectional** Chamfer distance between
    that polyline's points and the point cloud of **all other** polylines::

        d_i = ( d(polyline_i → rest) + d(rest → polyline_i) ) / 2

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

        # True bidirectional Chamfer: polyline i → rest, and rest → polyline i
        tree_cloud = cKDTree(cloud)
        tree_i = cKDTree(pts_i)
        d_fwd = float(tree_cloud.query(pts_i, k=1)[0].mean())  # polyline i → rest
        d_bwd = float(tree_i.query(cloud, k=1)[0].mean())  # rest → polyline i
        per_poly.append((d_fwd + d_bwd) / 2.0)

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
        # 1 % of extent, clamped to [10, 30] m.  Tighter than the old 5 % so
        # an endpoint matches only its own junction — 5 % (~100 m on a 2 km
        # map) cross-matched endpoints to every nearby junction and inflated
        # the error on dense intersection graphs.
        merge_distance = min(max(extent * 0.01, 10.0), 30.0)

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
        "n_cross_pairs": total_pairs,
        "n_collinear_overlaps": n_collinear,
        "has_intersection": n_intersections > 0,
    }


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

    # Crossings normalised per km of road (native coords assumed metre-scale).
    # This replaces the old near-zero `intersection_rate` (count / all pairs).
    result["crossings_per_km"] = result["n_intersections"] / max(
        result.get("total_road_length", 0.0) / 1000.0, 1e-9
    )

    return result

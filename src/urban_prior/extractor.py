"""Urban Structural Prior Extraction from OSM road networks.

Computes all structural priors (global, skeleton graph, block) from a
clipped OSM road network GeoDataFrame for a 2km x 2km urban patch.

Public entry point: extract_all_priors()
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from shapely.geometry import Polygon, box

from src.urban_prior.block_utils import compute_block_prior, extract_blocks
from src.urban_prior.graph_utils import (
    HIGHWAY_HIERARCHY,
    MAJOR_HIGHWAYS,
    METERS_PER_DEG_LAT,
    SKELETON_HIGHWAYS,
    build_graph_from_gdf,
    detect_roundabouts,
    estimate_lanes,
    estimate_width_m,
    find_boundary_nodes,
    meters_per_deg_lon,
    project_to_local_meters,
    total_road_length_m,
)

logger = logging.getLogger(__name__)


def compute_global_prior(
    gdf,
    graph: nx.Graph,
    center_lat: float,
    center_lon: float,
    context_size_m: float,
) -> Dict[str, Any]:
    """Compute all global prior metrics from a clipped road network.

    Args:
        gdf: Clipped road GeoDataFrame.
        graph: NetworkX graph built from the clipped GDF.
        center_lat, center_lon: Patch center.
        context_size_m: Patch width/height in meters.

    Returns:
        Dictionary of global prior metrics.
    """
    area_km2 = (context_size_m / 1000.0) ** 2

    # ── Road densities ──────────────────────────────────────────────
    total_length_m = total_road_length_m(gdf, center_lat)
    total_length_km = total_length_m / 1000.0
    road_density = total_length_km / area_km2 if area_km2 > 0 else 0.0

    # Major road density (motorway, trunk, primary, secondary)
    major_mask = gdf["highway"].isin(MAJOR_HIGHWAYS)
    major_gdf = gdf[major_mask]
    major_length_m = total_road_length_m(major_gdf, center_lat)
    major_density = (major_length_m / 1000.0) / area_km2 if area_km2 > 0 else 0.0

    # Minor road density
    minor_gdf = gdf[~major_mask]
    minor_length_m = total_road_length_m(minor_gdf, center_lat)
    minor_density = (minor_length_m / 1000.0) / area_km2 if area_km2 > 0 else 0.0

    # ── Node and edge counts ───────────────────────────────────────
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()

    # ── Node degree statistics ────────────────────────────────────
    degrees = [d for _, d in graph.degree()] if node_count > 0 else []
    avg_degree = float(np.mean(degrees)) if degrees else 0.0

    dead_end_count = sum(1 for d in degrees if d == 1)
    dead_end_ratio = dead_end_count / node_count if node_count > 0 else 0.0

    three_way_count = sum(1 for d in degrees if d == 3)
    four_way_count = sum(1 for d in degrees if d == 4)

    # Filter to intersection nodes (degree >= 3)
    intersection_count = sum(1 for d in degrees if d >= 3)

    three_way_ratio = three_way_count / intersection_count if intersection_count > 0 else 0.0
    four_way_ratio = four_way_count / intersection_count if intersection_count > 0 else 0.0

    # ── Major node count (degree >= 3 on major roads) ─────────────
    major_nodes = set()
    for u, v, data in graph.edges(data=True):
        hw = data.get("highway", "")
        if hw in MAJOR_HIGHWAYS:
            if graph.degree(u) >= 3:
                major_nodes.add(u)
            if graph.degree(v) >= 3:
                major_nodes.add(v)
    major_node_count = len(major_nodes)

    # ── Boundary entry count ──────────────────────────────────────
    boundary_nodes = find_boundary_nodes(graph, center_lat, center_lon, context_size_m)
    boundary_entry_count = len(boundary_nodes)

    # ── Road length statistics ────────────────────────────────────
    m_per_lon = meters_per_deg_lon(center_lat)
    road_lengths = []
    for _, _, data in graph.edges(data=True):
        geom = data.get("geometry")
        if geom is not None and not geom.is_empty:
            # Compute length in meters
            coords = list(geom.coords)
            seg_len = 0.0
            for i in range(len(coords) - 1):
                dx = (coords[i + 1][0] - coords[i][0]) * m_per_lon
                dy = (coords[i + 1][1] - coords[i][1]) * METERS_PER_DEG_LAT
                seg_len += math.sqrt(dx * dx + dy * dy)
            road_lengths.append(seg_len)

    if road_lengths:
        road_length_mean = float(np.mean(road_lengths))
        road_length_std = float(np.std(road_lengths))
        road_length_median = float(np.median(road_lengths))
    else:
        road_length_mean = 0.0
        road_length_std = 0.0
        road_length_median = 0.0

    # ── Bearing entropy ───────────────────────────────────────────
    bearing_entropy_val = compute_bearing_entropy(graph)

    # ── Orientation entropy (segment-based) ───────────────────────
    orientation_entropy_val = compute_orientation_entropy(gdf, center_lat)

    # ── Pattern scores ────────────────────────────────────────────
    gridness = compute_gridness_score(graph, gdf)
    radialness = compute_radialness_score(
        graph, center_lat, center_lon, context_size_m
    )
    organic = compute_organic_score(graph, gdf, center_lat)

    # ── Lane / width estimates ────────────────────────────────────
    lane_sum = 0.0
    width_sum = 0.0
    lane_count = 0
    for _, _, data in graph.edges(data=True):
        hw = data.get("highway", "unclassified")
        lane_sum += estimate_lanes(hw)
        width_sum += estimate_width_m(hw)
        lane_count += 1
    estimated_lane_mean = lane_sum / lane_count if lane_count > 0 else 0.0
    estimated_road_width_mean = width_sum / lane_count if lane_count > 0 else 0.0

    return {
        "road_density_km_per_km2": road_density,
        "major_road_density_km_per_km2": major_density,
        "minor_road_density_km_per_km2": minor_density,
        "node_count": node_count,
        "edge_count": edge_count,
        "intersection_count": intersection_count,
        "major_node_count": major_node_count,
        "boundary_entry_count": boundary_entry_count,
        "avg_degree": avg_degree,
        "dead_end_ratio": dead_end_ratio,
        "three_way_ratio": three_way_ratio,
        "four_way_ratio": four_way_ratio,
        "road_length_mean_m": road_length_mean,
        "road_length_std_m": road_length_std,
        "road_length_median_m": road_length_median,
        "orientation_entropy": orientation_entropy_val,
        "bearing_entropy": bearing_entropy_val,
        "gridness_score": gridness,
        "radialness_score": radialness,
        "organic_score": organic,
        "estimated_lane_mean": estimated_lane_mean,
        "estimated_road_width_mean_m": estimated_road_width_mean,
    }


# ─── Bearing entropy ────────────────────────────────────────────────


def compute_bearing_entropy(graph: nx.Graph, n_bins: int = 18) -> float:
    """Compute entropy of edge bearing distribution.

    Bearings are binned into n_bins equal-width bins over [0, 180) degrees.
    Uses Shannon entropy: H = -sum(p_i * log(p_i)), normalized to [0, 1].

    Args:
        graph: Road graph with 'bearing_deg' edge attribute.
        n_bins: Number of bins (18 = 10-degree bins for 0-180).

    Returns:
        Normalized bearing entropy in [0, 1].
    """
    bearings = []
    for _, _, data in graph.edges(data=True):
        bearing = data.get("bearing_deg")
        if bearing is not None:
            bearings.append(bearing)

    if not bearings:
        return 0.0

    # Bin bearings
    bins = np.linspace(0, 180, n_bins + 1)
    hist, _ = np.histogram(bearings, bins=bins, density=True)

    # Normalize to probabilities
    probs = hist / (hist.sum() + 1e-10)

    # Shannon entropy (natural log)
    entropy = -np.sum(probs * np.log(probs + 1e-10))

    # Normalize to [0, 1]: max entropy is log(n_bins)
    max_entropy = math.log(n_bins)
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


# ─── Orientation entropy (segment-based) ────────────────────────────


def compute_orientation_entropy(gdf, center_lat: float, n_bins: int = 18) -> float:
    """Compute entropy of road segment orientation distribution.

    Each road LineString is decomposed into straight segments (between
    consecutive coordinates), and each segment's bearing is computed.
    Bearings are binned over [0, 180).

    Args:
        gdf: Road GeoDataFrame.
        center_lat: Center latitude for meter-per-degree scaling.
        n_bins: Number of bins.

    Returns:
        Normalized orientation entropy in [0, 1].
    """
    bearings = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "MultiLineString":
            lines = list(geom.geoms)
        else:
            lines = [geom]

        for line in lines:
            coords = list(line.coords)
            if len(coords) < 2:
                continue
            for i in range(len(coords) - 1):
                dx = coords[i + 1][0] - coords[i][0]
                dy = coords[i + 1][1] - coords[i][1]

                # Only consider segments longer than ~1m to avoid noise
                d_m = math.sqrt(
                    (dx * meters_per_deg_lon(center_lat)) ** 2
                    + (dy * METERS_PER_DEG_LAT) ** 2
                )
                if d_m < 1.0:
                    continue

                bearing_rad = math.atan2(dx, dy)
                bearing_deg = math.degrees(bearing_rad) % 180
                bearings.append(bearing_deg)

    if not bearings:
        return 0.0

    bins = np.linspace(0, 180, n_bins + 1)
    hist, _ = np.histogram(bearings, bins=bins, density=True)
    probs = hist / (hist.sum() + 1e-10)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = math.log(n_bins)
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


# ─── Gridness score heuristic ───────────────────────────────────────


def compute_gridness_score(graph: nx.Graph, gdf) -> float:
    """Estimate gridness of the road network.

    Heuristic formula combining:
    - Gridness increases with four-way intersection ratio.
    - Gridness decreases with bearing entropy (strong orientation = more grid-like).
    - Gridness increases with proportion of orthogonal bearings.
    - Gridness decreases with dead-end ratio.

    Score range: [0, 1] where 1 = perfectly grid-like.

    Formula:
        gridness = w1 * (1 - bearing_entropy) +
                   w2 * four_way_ratio +
                   w3 * orthogonality_score +
                   w4 * (1 - dead_end_ratio)

    Weights: w1=0.3, w2=0.25, w3=0.3, w4=0.15
    """
    bearing_entropy = compute_bearing_entropy(graph)
    dead_end_ratio = _compute_dead_end_ratio(graph)

    # Four-way ratio among intersections
    four_way_ratio = _compute_four_way_ratio(graph)

    # Orthogonality score: proportion of edge pairs that are near 90°
    orthogonality_score = _compute_orthogonality_score(graph)

    w1, w2, w3, w4 = 0.3, 0.25, 0.3, 0.15

    gridness = (
        w1 * (1.0 - bearing_entropy)
        + w2 * four_way_ratio
        + w3 * orthogonality_score
        + w4 * (1.0 - dead_end_ratio)
    )

    return float(np.clip(gridness, 0.0, 1.0))


def _compute_dead_end_ratio(graph: nx.Graph) -> float:
    """Compute proportion of degree-1 nodes."""
    if graph.number_of_nodes() == 0:
        return 0.0
    dead_ends = sum(1 for _, d in graph.degree() if d == 1)
    return dead_ends / graph.number_of_nodes()


def _compute_four_way_ratio(graph: nx.Graph) -> float:
    """Compute ratio of degree-4 nodes among all intersection nodes (degree >= 3)."""
    intersections = [d for _, d in graph.degree() if d >= 3]
    if not intersections:
        return 0.0
    four_ways = sum(1 for d in intersections if d == 4)
    return four_ways / len(intersections)


def _compute_orthogonality_score(graph: nx.Graph, tol_deg: float = 15.0) -> float:
    """Compute proportion of edge pairs at a node that are near-orthogonal.

    At each node, for each pair of incident edges, check if the bearing
    difference is within tol_deg of 90°. Returns the fraction of near-
    orthogonal pairs across all nodes.
    """
    total_pairs = 0
    ortho_pairs = 0

    for node in graph.nodes():
        edges = list(graph.edges(node, data=True))
        if len(edges) < 2:
            continue

        bearings = []
        for u, v, data in edges:
            bearing = data.get("bearing_deg")
            if bearing is not None:
                bearings.append(bearing)

        for i in range(len(bearings)):
            for j in range(i + 1, len(bearings)):
                total_pairs += 1
                diff = abs(bearings[i] - bearings[j])
                diff = min(diff, 180 - diff)  # circular diff on [0, 180)
                if abs(diff - 90) <= tol_deg:
                    ortho_pairs += 1

    if total_pairs == 0:
        return 0.0
    return ortho_pairs / total_pairs


# ─── Radialness score heuristic ─────────────────────────────────────


def compute_radialness_score(
    graph: nx.Graph,
    center_lat: float,
    center_lon: float,
    context_size_m: float,
    n_radial_bins: int = 36,
) -> float:
    """Estimate radialness of the road network.

    Heuristic: a radial network has roads emanating from a center point
    toward the boundary.

    Algorithm:
    1. Find the "center" node as the one closest to the patch center.
    2. For each edge on a major road, compute:
       - Whether it points outward from center (bearing away from center).
       - Whether it connects the center to the boundary.
    3. Compute radial alignment: fraction of major edges whose bearing
       points radially (within tolerance of the center-to-edge direction).

    Formula:
        radialness = w1 * radial_alignment + w2 * hub_connectivity + w3 * boundary_connectivity

    Score range: [0, 1].
    """
    if graph.number_of_nodes() == 0:
        return 0.0

    m_per_lon = meters_per_deg_lon(center_lat)

    # Find the hub node (closest to center)
    hub_node = None
    hub_dist = float("inf")
    for node in graph.nodes():
        lon, lat = node
        x_m = (lon - center_lon) * m_per_lon
        y_m = (lat - center_lat) * METERS_PER_DEG_LAT
        dist = math.sqrt(x_m * x_m + y_m * y_m)
        if dist < hub_dist:
            hub_dist = dist
            hub_node = node

    if hub_node is None:
        return 0.0

    half = context_size_m / 2.0

    # Count major edges and how many point radially
    total_major_edges = 0
    radial_edges = 0
    hub_connected = 0
    boundary_connected = 0
    boundary_entries = set(
        find_boundary_nodes(graph, center_lat, center_lon, context_size_m)
    )

    for u, v, data in graph.edges(data=True):
        hw = data.get("highway", "")
        if hw not in {"motorway", "trunk", "primary", "secondary"}:
            continue

        total_major_edges += 1

        # Check if either endpoint is near center
        for n in [u, v]:
            if n == hub_node:
                hub_connected += 1
                break

        # Check if connects to boundary
        if u in boundary_entries or v in boundary_entries:
            boundary_connected += 1

        # Check radial alignment
        lon_u, lat_u = u
        lon_v, lat_v = v
        x_u = (lon_u - center_lon) * m_per_lon
        y_u = (lat_u - center_lat) * METERS_PER_DEG_LAT
        x_v = (lon_v - center_lon) * m_per_lon
        y_v = (lat_v - center_lat) * METERS_PER_DEG_LAT

        mid_x = (x_u + x_v) / 2.0
        mid_y = (y_u + y_v) / 2.0

        # Direction from center to edge midpoint
        center_to_edge_angle = math.atan2(mid_x, mid_y)

        # Edge direction (from one endpoint to the other)
        edge_angle = math.atan2(x_v - x_u, y_v - y_u)

        # Check alignment: are the edge direction and center-to-edge
        # direction within 30 degrees?
        angle_diff = abs(center_to_edge_angle - edge_angle)
        angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
        if angle_diff < math.radians(30):
            radial_edges += 1

    radial_alignment = radial_edges / total_major_edges if total_major_edges > 0 else 0.0
    hub_connectivity = hub_connected / total_major_edges if total_major_edges > 0 else 0.0
    boundary_connectivity = boundary_connected / total_major_edges if total_major_edges > 0 else 0.0

    # Also consider: is there a node near the center at all?
    center_concentration = 1.0 if hub_dist < half * 0.3 else 0.3

    w1, w2, w3, w4 = 0.3, 0.2, 0.3, 0.2
    radialness = (
        w1 * radial_alignment
        + w2 * hub_connectivity
        + w3 * boundary_connectivity
        + w4 * center_concentration
    )

    return float(np.clip(radialness, 0.0, 1.0))


# ─── Organic score heuristic ─────────────────────────────────────────


def compute_organic_score(graph: nx.Graph, gdf, center_lat: float) -> float:
    """Estimate organic (irregular, unplanned) nature of the road network.

    Organic networks are characterized by:
    - High orientation entropy (no dominant direction).
    - High dead-end ratio (cul-de-sacs).
    - High three-way intersection ratio (T-junctions).
    - High edge curvature (winding roads).

    Formula:
        organic = w1 * orientation_entropy +
                  w2 * dead_end_ratio +
                  w3 * three_way_ratio +
                  w4 * mean_curvature

    Score range: [0, 1].
    """
    orientation_entropy = compute_orientation_entropy(gdf, center_lat)
    dead_end_ratio = _compute_dead_end_ratio(graph)
    three_way_ratio = _compute_three_way_ratio(graph)
    mean_curvature = _compute_mean_curvature(graph)

    w1, w2, w3, w4 = 0.3, 0.25, 0.25, 0.2

    organic = (
        w1 * orientation_entropy
        + w2 * dead_end_ratio
        + w3 * three_way_ratio
        + w4 * mean_curvature
    )

    return float(np.clip(organic, 0.0, 1.0))


def _compute_three_way_ratio(graph: nx.Graph) -> float:
    """Compute ratio of degree-3 nodes among all intersection nodes (degree >= 3)."""
    intersections = [d for _, d in graph.degree() if d >= 3]
    if not intersections:
        return 0.0
    three_ways = sum(1 for d in intersections if d == 3)
    return three_ways / len(intersections)


def _compute_mean_curvature(graph: nx.Graph) -> float:
    """Compute mean edge curvature, normalized to [0, 1].

    Curvature = path_length / straight_line_distance.
    Normalization: 1 - exp(-(curvature - 1)), so curvature=1 gives 0,
    curvature=2 gives ~0.63, curvature=3 gives ~0.86.
    """
    curvatures = []
    for _, _, data in graph.edges(data=True):
        curvature = data.get("curvature", 1.0)
        if curvature > 1.0:
            curvatures.append(curvature)

    if not curvatures:
        return 0.0

    mean_curv = float(np.mean(curvatures))
    # Normalize: map curvature-1 to [0, 1] using 1 - exp(-(curv-1))
    normalized = 1.0 - math.exp(-(mean_curv - 1.0))
    return normalized


# ─── Urban skeleton graph extraction ─────────────────────────────────


def extract_skeleton_graph(
    graph: nx.Graph,
    gdf,
    center_lat: float,
    center_lon: float,
    context_size_m: float,
    include_tertiary: bool = False,
) -> Dict[str, Any]:
    """Extract the urban skeleton graph (major roads only).

    Returns serializable dict with 'nodes' and 'edges' lists.

    Args:
        graph: Complete road graph.
        gdf: Complete clipped GeoDataFrame.
        center_lat, center_lon: Patch center.
        context_size_m: Patch size in meters.
        include_tertiary: Whether to include tertiary roads in skeleton.

    Returns:
        Dict with 'nodes' (list) and 'edges' (list), plus metadata.
    """
    skeleton_highways = SKELETON_HIGHWAYS.copy()
    if include_tertiary:
        skeleton_highways = skeleton_highways | {"tertiary"}

    half = context_size_m / 2.0

    # Find boundary nodes for labeling
    boundary_nodes = set(
        find_boundary_nodes(graph, center_lat, center_lon, context_size_m)
    )

    # Collect all skeleton edge data
    skeleton_edges = []
    skeleton_nodes = set()

    for u, v, data in graph.edges(data=True):
        hw = data.get("highway", "")
        if hw not in skeleton_highways:
            continue
        skeleton_edges.append((u, v, data))
        skeleton_nodes.add(u)
        skeleton_nodes.add(v)

    # Build node entries
    m_per_lon = meters_per_deg_lon(center_lat)
    node_list = []
    node_id_map = {}  # (lon, lat) -> integer ID

    for idx, node in enumerate(sorted(skeleton_nodes, key=lambda n: (n[1], n[0]))):
        lon, lat = node
        x_m = (lon - center_lon) * m_per_lon
        y_m = (lat - center_lat) * METERS_PER_DEG_LAT
        x_norm = (x_m + half) / context_size_m
        y_norm = (y_m + half) / context_size_m

        node_type = _classify_node_type(graph, node)

        node_list.append({
            "id": idx,
            "x": round(x_norm, 6),
            "y": round(y_norm, 6),
            "x_m": round(x_m, 2),
            "y_m": round(y_m, 2),
            "degree": graph.degree(node),
            "node_type": node_type,
            "is_boundary": node in boundary_nodes,
        })
        node_id_map[node] = idx

    # Build edge entries
    edge_list = []
    skeleton_includes_tertiary = include_tertiary
    for u, v, data in skeleton_edges:
        hw = data.get("highway", "")
        geom = data.get("geometry")
        if geom is not None and not geom.is_empty:
            # More accurate length from geometry
            coords = list(geom.coords)
            seg_len = 0.0
            for i in range(len(coords) - 1):
                dx = (coords[i + 1][0] - coords[i][0]) * m_per_lon
                dy = (coords[i + 1][1] - coords[i][1]) * METERS_PER_DEG_LAT
                seg_len += math.sqrt(dx * dx + dy * dy)
        else:
            seg_len = 0.0

        if u not in node_id_map or v not in node_id_map:
            continue

        edge_list.append({
            "source": node_id_map[u],
            "target": node_id_map[v],
            "highway": hw,
            "hierarchy_level": HIGHWAY_HIERARCHY.get(hw, 1),
            "length_m": round(seg_len, 2),
            "bearing_deg": round(data.get("bearing_deg", 0.0), 2),
            "curvature": round(data.get("curvature", 1.0), 4),
            "estimated_lanes": estimate_lanes(hw),
            "estimated_width_m": estimate_width_m(hw),
        })

    # Detect roundabouts and relabel nodes
    roundabout_ids = detect_roundabouts(node_list, edge_list)
    for n in node_list:
        if n["id"] in roundabout_ids:
            n["node_type"] = "roundabout"

    return {
        "nodes": node_list,
        "edges": edge_list,
        "skeleton_includes_tertiary": skeleton_includes_tertiary,
        "skeleton_edge_count": len(edge_list),
        "skeleton_node_count": len(node_list),
    }


def _classify_node_type(graph: nx.Graph, node) -> str:
    """Classify a node's type based on incident edge highways and degree."""
    deg = graph.degree(node)

    if deg >= 4:
        return "major_intersection"
    elif deg == 3:
        return "intersection"
    elif deg == 2:
        # Could be a through-point or a minor bend
        # Check if incident edges are on major highways
        incident_highways = set()
        for neighbor in graph.neighbors(node):
            edge_data = graph.get_edge_data(node, neighbor)
            if edge_data and isinstance(edge_data, dict):
                incident_highways.add(edge_data.get("highway", ""))
            elif edge_data:
                # Multiple edges between same nodes
                for _, data in edge_data.items() if isinstance(edge_data, dict) else []:
                    incident_highways.add(data.get("highway", ""))
                    break

        if any(hw in MAJOR_HIGHWAYS for hw in incident_highways):
            return "skeleton_waypoint"
        return "waypoint"
    elif deg == 1:
        return "dead_end"
    return "isolated"


# ─── Block prior wrapper ────────────────────────────────────────────


def compute_block_prior_wrapper(
    gdf,
    bbox_poly: Polygon,
    center_lat: float,
) -> Tuple[Dict, bool, Optional[str]]:
    """Extract block prior from a clipped GeoDataFrame.

    Args:
        gdf: Clipped road GeoDataFrame.
        bbox_poly: Bounding box polygon.
        center_lat: Center latitude for area computation.

    Returns:
        (block_prior_dict, was_successful, error_msg)
    """
    blocks, error = extract_blocks(gdf, bbox_poly)
    if error or not blocks:
        return {}, False, error or "No blocks extracted"

    try:
        block_prior = compute_block_prior(blocks, center_lat)
        if not block_prior:
            return {}, False, "Block prior computation returned empty"
        return block_prior, True, None
    except Exception as e:
        return {}, False, f"Block prior computation failed: {e}"


# ─── Main entry point ───────────────────────────────────────────────


def extract_all_priors(
    gdf,
    center_lat: float,
    center_lon: float,
    context_size_m: float,
) -> Dict[str, Any]:
    """Extract all urban structural priors from a clipped road network.

    This is the main entry point for prior extraction.

    Args:
        gdf: Clipped road GeoDataFrame (LineStrings with 'highway' column).
        center_lat, center_lon: Patch center in degrees.
        context_size_m: Patch width/height in meters.

    Returns:
        Dictionary with keys:
            - global_prior: Dict of global metrics (may be empty on failure).
            - urban_skeleton_graph: Dict with nodes/edges (may be empty on failure).
            - block_prior: Dict of block metrics (may be empty on failure).
            - quality: Dict with 'valid_graph' and 'error' fields.
    """
    result = {
        "global_prior": {},
        "urban_skeleton_graph": {"nodes": [], "edges": []},
        "block_prior": {},
        "quality": {"valid_graph": True, "error": None},
    }

    if gdf is None or gdf.empty:
        result["quality"] = {"valid_graph": False, "error": "Empty road GDF"}
        return result

    try:
        # Build graph from GDF
        graph = build_graph_from_gdf(gdf)
    except Exception as e:
        result["quality"] = {
            "valid_graph": False,
            "error": f"Graph building failed: {e}",
        }
        return result

    if graph.number_of_nodes() == 0:
        result["quality"] = {"valid_graph": False, "error": "Graph has zero nodes"}
        return result

    # ── Global prior ───────────────────────────────────────────────
    try:
        result["global_prior"] = compute_global_prior(
            gdf, graph, center_lat, center_lon, context_size_m
        )
    except Exception as e:
        logger.warning("Global prior computation failed: %s", e)
        result["quality"]["error"] = f"Global prior failed: {e}"

    # ── Urban skeleton graph ───────────────────────────────────────
    try:
        # First try without tertiary
        skeleton = extract_skeleton_graph(
            graph, gdf, center_lat, center_lon, context_size_m,
            include_tertiary=False,
        )
        # If too sparse, try with tertiary
        if skeleton["skeleton_edge_count"] < 3:
            skeleton = extract_skeleton_graph(
                graph, gdf, center_lat, center_lon, context_size_m,
                include_tertiary=True,
            )
        result["urban_skeleton_graph"] = skeleton
    except Exception as e:
        logger.warning("Skeleton graph extraction failed: %s", e)
        if result["quality"]["error"] is None:
            result["quality"]["error"] = f"Skeleton extraction failed: {e}"

    # ── Block prior ────────────────────────────────────────────────
    try:
        half = context_size_m / 2.0
        m_per_lon = meters_per_deg_lon(center_lat)
        dlat = half / METERS_PER_DEG_LAT
        dlon = half / m_per_lon
        bbox_poly = box(
            center_lon - dlon, center_lat - dlat,
            center_lon + dlon, center_lat + dlat,
        )
        block_prior_dict, block_ok, block_error = compute_block_prior_wrapper(
            gdf, bbox_poly, center_lat
        )
        result["block_prior"] = block_prior_dict
        if not block_ok:
            result["block_prior"]["block_prior_available"] = False
            if block_error:
                if result["quality"]["error"] is None:
                    result["quality"]["error"] = block_error
        else:
            result["block_prior"]["block_prior_available"] = True
    except Exception as e:
        logger.warning("Block prior extraction failed: %s", e)
        result["block_prior"] = {"block_prior_available": False}
        if result["quality"]["error"] is None:
            result["quality"]["error"] = f"Block prior failed: {e}"

    return result

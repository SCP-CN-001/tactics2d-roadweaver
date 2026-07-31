"""Urban structural prior extraction from OSM networks."""

from __future__ import annotations

import logging
import math
from typing import Any

import networkx as nx
import numpy as np
from shapely.geometry import Polygon, box

from .block_analysis import compute_block_prior, extract_blocks
from .osm_graph import (
    MAJOR_ROAD_TYPES,
    METERS_PER_DEG_LAT,
    ROAD_HIERARCHY,
    SKELETON_ROAD_TYPES,
    build_graph_from_gdf,
    detect_roundabouts,
    estimate_lanes,
    estimate_width_m,
    find_boundary_nodes,
    meters_per_deg_lon,
    total_road_length_m,
)

logger = logging.getLogger(__name__)


def compute_global_prior(
    gdf, graph: nx.Graph, center_lat: float, center_lon: float, context_size_m: float
) -> dict[str, Any]:
    """Compute global prior metrics from a clipped road network."""
    area_km2 = (context_size_m / 1000.0) ** 2

    total_length_m = total_road_length_m(gdf, center_lat)
    total_length_km = total_length_m / 1000.0
    road_density = total_length_km / area_km2 if area_km2 > 0 else 0.0

    major_mask = gdf["highway"].isin(MAJOR_ROAD_TYPES)
    major_density = (
        (total_road_length_m(gdf[major_mask], center_lat) / 1000.0) / area_km2
        if area_km2 > 0
        else 0.0
    )
    minor_density = (
        (total_road_length_m(gdf[~major_mask], center_lat) / 1000.0) / area_km2
        if area_km2 > 0
        else 0.0
    )

    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    degrees = [d for _, d in graph.degree()] if node_count > 0 else []
    avg_degree = float(np.mean(degrees)) if degrees else 0.0
    dead_end_count = sum(1 for d in degrees if d == 1)
    dead_end_ratio = dead_end_count / node_count if node_count > 0 else 0.0
    intersection_count = sum(1 for d in degrees if d >= 3)
    three_way_count = sum(1 for d in degrees if d == 3)
    four_way_count = sum(1 for d in degrees if d == 4)
    three_way_ratio = three_way_count / intersection_count if intersection_count > 0 else 0.0
    four_way_ratio = four_way_count / intersection_count if intersection_count > 0 else 0.0

    major_nodes = set()
    for u, v, data in graph.edges(data=True):
        hw = data.get("highway", "")
        if hw in MAJOR_ROAD_TYPES:
            if graph.degree(u) >= 3:
                major_nodes.add(u)
            if graph.degree(v) >= 3:
                major_nodes.add(v)
    major_node_count = len(major_nodes)

    boundary_nodes = find_boundary_nodes(graph, center_lat, center_lon, context_size_m)
    boundary_entry_count = len(boundary_nodes)

    m_per_lon = meters_per_deg_lon(center_lat)
    road_lengths = []
    for _, _, data in graph.edges(data=True):
        geom = data.get("geometry")
        if geom is not None and not geom.is_empty:
            coords = list(geom.coords)
            seg_len = 0.0
            for i in range(len(coords) - 1):
                dx = (coords[i + 1][0] - coords[i][0]) * m_per_lon
                dy = (coords[i + 1][1] - coords[i][1]) * METERS_PER_DEG_LAT
                seg_len += math.sqrt(dx * dx + dy * dy)
            road_lengths.append(seg_len)

    road_length_mean = float(np.mean(road_lengths)) if road_lengths else 0.0
    road_length_std = float(np.std(road_lengths)) if road_lengths else 0.0
    road_length_median = float(np.median(road_lengths)) if road_lengths else 0.0

    bearing_entropy_val = compute_bearing_entropy(graph)
    orientation_entropy_val = compute_orientation_entropy(gdf, center_lat)
    gridness = compute_gridness_score(graph, gdf)
    radialness = compute_radialness_score(graph, center_lat, center_lon, context_size_m)
    organic = compute_organic_score(graph, gdf, center_lat)

    lane_sum = width_sum = lane_count = 0
    for _, _, data in graph.edges(data=True):
        hw = data.get("highway", "unclassified")
        lane_sum += estimate_lanes(hw)
        width_sum += estimate_width_m(hw)
        lane_count += 1

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
        "estimated_lane_mean": lane_sum / lane_count if lane_count > 0 else 0.0,
        "estimated_road_width_mean_m": width_sum / lane_count if lane_count > 0 else 0.0,
    }


# ── Bearing entropy ───────────────────────────────────────────────────


def compute_bearing_entropy(graph: nx.Graph, n_bins: int = 18) -> float:
    """Shannon entropy of edge bearing distribution, normalised to [0, 1]."""
    bearings = []
    for _, _, data in graph.edges(data=True):
        bearing = data.get("bearing_deg")
        if bearing is not None:
            bearings.append(bearing)

    if not bearings:
        return 0.0

    hist, _ = np.histogram(bearings, bins=np.linspace(0, 180, n_bins + 1), density=True)
    probs = hist / (hist.sum() + 1e-10)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return float(entropy / math.log(n_bins)) if n_bins > 1 else 0.0


def compute_orientation_entropy(gdf, center_lat: float, n_bins: int = 18) -> float:
    """Segment-based orientation entropy, normalised to [0, 1]."""
    bearings = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        lines = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
        for line in lines:
            coords = list(line.coords)
            if len(coords) < 2:
                continue
            for i in range(len(coords) - 1):
                dx = coords[i + 1][0] - coords[i][0]
                dy = coords[i + 1][1] - coords[i][1]
                d_m = math.sqrt(
                    (dx * meters_per_deg_lon(center_lat)) ** 2 + (dy * METERS_PER_DEG_LAT) ** 2
                )
                if d_m < 1.0:
                    continue
                bearing_deg = math.degrees(math.atan2(dx, dy)) % 180
                bearings.append(bearing_deg)

    if not bearings:
        return 0.0

    hist, _ = np.histogram(bearings, bins=np.linspace(0, 180, n_bins + 1), density=True)
    probs = hist / (hist.sum() + 1e-10)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return float(entropy / math.log(n_bins)) if n_bins > 1 else 0.0


# ── Gridness score ────────────────────────────────────────────────────


def compute_gridness_score(graph: nx.Graph, gdf) -> float:
    """Heuristic gridness score in [0, 1]."""
    bearing_entropy = compute_bearing_entropy(graph)
    dead_end_ratio = _compute_dead_end_ratio(graph)
    four_way_ratio = _compute_four_way_ratio(graph)
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
    """Compute the dead-end ratio of a graph."""
    if graph.number_of_nodes() == 0:
        return 0.0
    return sum(1 for _, d in graph.degree() if d == 1) / graph.number_of_nodes()


def _compute_four_way_ratio(graph: nx.Graph) -> float:
    """Compute the four-way intersection ratio."""
    intersections = [d for _, d in graph.degree() if d >= 3]
    if not intersections:
        return 0.0
    return sum(1 for d in intersections if d == 4) / len(intersections)


def _compute_orthogonality_score(graph: nx.Graph, tol_deg: float = 15.0) -> float:
    """Fraction of edge pairs at nodes that are within *tol_deg* of orthogonal."""
    total_pairs = ortho_pairs = 0
    for node in graph.nodes():
        edges = list(graph.edges(node, data=True))
        if len(edges) < 2:
            continue
        bearings = [
            data.get("bearing_deg") for _, _, data in edges if data.get("bearing_deg") is not None
        ]
        for i in range(len(bearings)):
            for j in range(i + 1, len(bearings)):
                total_pairs += 1
                diff = min(abs(bearings[i] - bearings[j]), 180 - abs(bearings[i] - bearings[j]))
                if abs(diff - 90) <= tol_deg:
                    ortho_pairs += 1
    return ortho_pairs / total_pairs if total_pairs > 0 else 0.0


# ── Radialness score ──────────────────────────────────────────────────


def compute_radialness_score(
    graph: nx.Graph,
    center_lat: float,
    center_lon: float,
    context_size_m: float,
    n_radial_bins: int = 36,
) -> float:
    """Heuristic radialness score in [0, 1]."""
    if graph.number_of_nodes() == 0:
        return 0.0

    m_per_lon = meters_per_deg_lon(center_lat)
    hub_node = min(
        graph.nodes(),
        key=lambda n: math.sqrt(
            ((n[0] - center_lon) * m_per_lon) ** 2 + ((n[1] - center_lat) * METERS_PER_DEG_LAT) ** 2
        ),
    )
    hub_dist = math.sqrt(
        ((hub_node[0] - center_lon) * m_per_lon) ** 2
        + ((hub_node[1] - center_lat) * METERS_PER_DEG_LAT) ** 2
    )
    half = context_size_m / 2.0
    boundary_entries = set(find_boundary_nodes(graph, center_lat, center_lon, context_size_m))

    total_major = radial = hub_conn = boundary_conn = 0
    for u, v, data in graph.edges(data=True):
        hw = data.get("highway", "")
        if hw not in MAJOR_ROAD_TYPES:
            continue
        total_major += 1
        if u == hub_node or v == hub_node:
            hub_conn += 1
        if u in boundary_entries or v in boundary_entries:
            boundary_conn += 1

        lon_u, lat_u = u
        lon_v, lat_v = v
        x_u, y_u = (lon_u - center_lon) * m_per_lon, (lat_u - center_lat) * METERS_PER_DEG_LAT
        x_v, y_v = (lon_v - center_lon) * m_per_lon, (lat_v - center_lat) * METERS_PER_DEG_LAT
        mid_x, mid_y = (x_u + x_v) / 2.0, (y_u + y_v) / 2.0
        c2e_angle = math.atan2(mid_x, mid_y)
        edge_angle = math.atan2(x_v - x_u, y_v - y_u)
        angle_diff = min(abs(c2e_angle - edge_angle), 2 * math.pi - abs(c2e_angle - edge_angle))
        if angle_diff < math.radians(30):
            radial += 1

    radial_alignment = radial / total_major if total_major > 0 else 0.0
    hub_connectivity = hub_conn / total_major if total_major > 0 else 0.0
    boundary_connectivity = boundary_conn / total_major if total_major > 0 else 0.0
    center_concentration = 1.0 if hub_dist < half * 0.3 else 0.3

    w1, w2, w3, w4 = 0.3, 0.2, 0.3, 0.2
    return float(
        np.clip(
            w1 * radial_alignment
            + w2 * hub_connectivity
            + w3 * boundary_connectivity
            + w4 * center_concentration,
            0.0,
            1.0,
        )
    )


# ── Organic score ─────────────────────────────────────────────────────


def compute_organic_score(graph: nx.Graph, gdf, center_lat: float) -> float:
    """Heuristic organic score in [0, 1]."""
    w1, w2, w3, w4 = 0.3, 0.25, 0.25, 0.2
    return float(
        np.clip(
            w1 * compute_orientation_entropy(gdf, center_lat)
            + w2 * _compute_dead_end_ratio(graph)
            + w3 * _compute_three_way_ratio(graph)
            + w4 * _compute_mean_curvature(graph),
            0.0,
            1.0,
        )
    )


def _compute_three_way_ratio(graph: nx.Graph) -> float:
    """Compute the three-way intersection ratio."""
    intersections = [d for _, d in graph.degree() if d >= 3]
    if not intersections:
        return 0.0
    return sum(1 for d in intersections if d == 3) / len(intersections)


def _compute_mean_curvature(graph: nx.Graph) -> float:
    """Mean edge curvature normalised to [0, 1]."""
    curvatures = [
        data.get("curvature", 1.0)
        for _, _, data in graph.edges(data=True)
        if data.get("curvature", 1.0) > 1.0
    ]
    if not curvatures:
        return 0.0
    return 1.0 - math.exp(-(float(np.mean(curvatures)) - 1.0))


# ── Urban skeleton graph extraction ───────────────────────────────────


def extract_skeleton_graph(
    graph: nx.Graph,
    gdf,
    center_lat: float,
    center_lon: float,
    context_size_m: float,
    include_tertiary: bool = False,
    road_types: set | None = None,
) -> dict[str, Any]:
    """Extract the urban skeleton graph from specified road types.

    Args:
        road_types: Set of road types to include.  If None, uses
            ``SKELETON_ROAD_TYPES`` (motorway, trunk, primary, secondary).
    """
    skeleton_road_types = road_types or SKELETON_ROAD_TYPES
    if include_tertiary:
        skeleton_road_types = skeleton_road_types | {"tertiary"}
    half = context_size_m / 2.0
    boundary_nodes = set(find_boundary_nodes(graph, center_lat, center_lon, context_size_m))
    m_per_lon = meters_per_deg_lon(center_lat)

    skeleton_edges = [
        (u, v, data)
        for u, v, data in graph.edges(data=True)
        if data.get("highway", "") in skeleton_road_types
    ]
    skeleton_nodes = {n for u, v, _ in skeleton_edges for n in (u, v)}

    node_list = []
    node_id_map = {}
    for idx, node in enumerate(sorted(skeleton_nodes, key=lambda n: (n[1], n[0]))):
        lon, lat = node
        x_m = (lon - center_lon) * m_per_lon
        y_m = (lat - center_lat) * METERS_PER_DEG_LAT
        node_list.append(
            {
                "id": idx,
                "x": round((x_m + half) / context_size_m, 6),
                "y": round((y_m + half) / context_size_m, 6),
                "x_m": round(x_m, 2),
                "y_m": round(y_m, 2),
                "degree": graph.degree(node),
                "node_type": _classify_node_type(graph, node),
                "is_boundary": node in boundary_nodes,
            }
        )
        node_id_map[node] = idx

    edge_list = []
    for u, v, data in skeleton_edges:
        geom = data.get("geometry")
        if geom is not None and not geom.is_empty:
            coords = list(geom.coords)
            seg_len = sum(
                math.sqrt(
                    ((coords[i + 1][0] - coords[i][0]) * m_per_lon) ** 2
                    + ((coords[i + 1][1] - coords[i][1]) * METERS_PER_DEG_LAT) ** 2
                )
                for i in range(len(coords) - 1)
            )
        else:
            seg_len = 0.0

        if u not in node_id_map or v not in node_id_map:
            continue

        hw = data.get("highway", "")
        edge_list.append(
            {
                "source": node_id_map[u],
                "target": node_id_map[v],
                "highway": hw,
                "hierarchy_level": ROAD_HIERARCHY.get(hw, 1),
                "length_m": round(seg_len, 2),
                "bearing_deg": round(data.get("bearing_deg", 0.0), 2),
                "curvature": round(data.get("curvature", 1.0), 4),
                "estimated_lanes": estimate_lanes(hw),
                "estimated_width_m": estimate_width_m(hw),
            }
        )

    roundabout_ids = detect_roundabouts(node_list, edge_list)
    for n in node_list:
        if n["id"] in roundabout_ids:
            n["node_type"] = "roundabout"

    return {
        "nodes": node_list,
        "edges": edge_list,
        "skeleton_includes_tertiary": include_tertiary,
        "skeleton_edge_count": len(edge_list),
        "skeleton_node_count": len(node_list),
    }


def _classify_node_type(graph: nx.Graph, node) -> str:
    """Classify a node based on degree and incident highway types."""
    deg = graph.degree(node)
    if deg >= 4:
        return "major_intersection"
    elif deg == 3:
        return "intersection"
    elif deg == 2:
        incident_highways = set()
        for neighbor in graph.neighbors(node):
            edge_data = graph.get_edge_data(node, neighbor)
            if edge_data and isinstance(edge_data, dict):
                incident_highways.add(edge_data.get("highway", ""))
            elif edge_data:
                for _, data in edge_data.items() if isinstance(edge_data, dict) else []:
                    incident_highways.add(data.get("highway", ""))
                    break
        return (
            "skeleton_waypoint"
            if any(hw in MAJOR_ROAD_TYPES for hw in incident_highways)
            else "waypoint"
        )
    elif deg == 1:
        return "dead_end"
    return "isolated"


# ── Block prior wrapper ───────────────────────────────────────────────


def compute_block_prior_wrapper(
    gdf, bbox_poly: Polygon, center_lat: float
) -> tuple[dict, bool, str | None]:
    """Extract block prior from a clipped GeoDataFrame."""
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


# ── Main entry point ──────────────────────────────────────────────────


def extract_all_priors(
    gdf,
    center_lat: float,
    center_lon: float,
    context_size_m: float,
    include_tertiary: bool = False,
    road_types: set | None = None,
) -> dict[str, Any]:
    """Extract all urban structural priors from a clipped road network.

    Args:
        road_types: Set of road types for skeleton graph extraction.
            If None, uses SKELETON_ROAD_TYPES (motorway, trunk, primary, secondary).
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
        graph = build_graph_from_gdf(gdf)
    except Exception as e:
        result["quality"] = {"valid_graph": False, "error": f"Graph building failed: {e}"}
        return result

    if graph.number_of_nodes() == 0:
        result["quality"] = {"valid_graph": False, "error": "Graph has zero nodes"}
        return result

    try:
        result["global_prior"] = compute_global_prior(
            gdf, graph, center_lat, center_lon, context_size_m
        )
    except Exception as e:
        logger.warning("Global prior failed: %s", e)
        result["quality"]["error"] = f"Global prior failed: {e}"

    try:
        skeleton = extract_skeleton_graph(
            graph,
            gdf,
            center_lat,
            center_lon,
            context_size_m,
            include_tertiary=include_tertiary,
            road_types=road_types,
        )
        if not include_tertiary and skeleton["skeleton_edge_count"] < 3:
            skeleton = extract_skeleton_graph(
                graph,
                gdf,
                center_lat,
                center_lon,
                context_size_m,
                include_tertiary=True,
                road_types=road_types,
            )
        result["urban_skeleton_graph"] = skeleton
    except Exception as e:
        logger.warning("Skeleton extraction failed: %s", e)
        if result["quality"]["error"] is None:
            result["quality"]["error"] = f"Skeleton extraction failed: {e}"

    try:
        half = context_size_m / 2.0
        dlat = half / METERS_PER_DEG_LAT
        dlon = half / meters_per_deg_lon(center_lat)
        bbox_poly = box(center_lon - dlon, center_lat - dlat, center_lon + dlon, center_lat + dlat)
        block_prior_dict, block_ok, block_error = compute_block_prior_wrapper(
            gdf, bbox_poly, center_lat
        )
        result["block_prior"] = block_prior_dict
        if not block_ok:
            result["block_prior"]["block_prior_available"] = False
        else:
            result["block_prior"]["block_prior_available"] = True
    except Exception as e:
        logger.warning("Block prior failed: %s", e)
        result["block_prior"] = {"block_prior_available": False}

    return result

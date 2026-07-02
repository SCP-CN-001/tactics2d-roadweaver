"""Utilities for loading, building, and analyzing OSM road graphs.

Provides functions for:
- City name matching between CRHD images and OSM GeoJSON files.
- Loading and spatially clipping OSM road network GeoDataFrames.
- Building simplified NetworkX graphs from LineString GeoDataFrames.
- Coordinate projection (lat/lon -> local cartesian meters).
- Edge bearing and curvature computation.
"""

import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, box

logger = logging.getLogger(__name__)

# ─── Highway hierarchy levels ─────────────────────────────────────────

HIGHWAY_HIERARCHY = {
    'motorway': 5,
    'motorway_link': 4,
    'trunk': 5,
    'trunk_link': 4,
    'primary': 4,
    'primary_link': 3,
    'secondary': 3,
    'secondary_link': 2,
    'tertiary': 2,
    'tertiary_link': 1,
    'residential': 1,
    'service': 1,
    'living_street': 1,
    'unclassified': 1,
}

MAJOR_HIGHWAYS = {'motorway', 'trunk', 'primary', 'secondary'}
SKELETON_HIGHWAYS = {'motorway', 'trunk', 'primary', 'secondary'}

# Approximate lane counts by highway type (heuristic)
ESTIMATED_LANES = {
    'motorway': 4,
    'motorway_link': 2,
    'trunk': 4,
    'trunk_link': 2,
    'primary': 3,
    'primary_link': 2,
    'secondary': 2,
    'secondary_link': 1,
    'tertiary': 2,
    'tertiary_link': 1,
    'residential': 1,
    'service': 1,
    'living_street': 1,
    'unclassified': 1,
}

# Approximate road width in meters by highway type
ESTIMATED_WIDTH_M = {
    'motorway': 25.0,
    'motorway_link': 10.0,
    'trunk': 20.0,
    'trunk_link': 10.0,
    'primary': 15.0,
    'primary_link': 8.0,
    'secondary': 10.0,
    'secondary_link': 6.0,
    'tertiary': 8.0,
    'tertiary_link': 5.0,
    'residential': 6.0,
    'service': 4.0,
    'living_street': 4.0,
    'unclassified': 5.0,
}

# Meters per degree latitude (constant)
METERS_PER_DEG_LAT = 111320.0


def meters_per_deg_lon(lat: float) -> float:
    """Compute meters per degree longitude at given latitude."""
    return METERS_PER_DEG_LAT * abs(math.cos(math.radians(lat))) + 1e-12


# ─── City name matching ──────────────────────────────────────────────


def build_city_name_map(
    osm_dir: str = "data/osm",
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build a bidirectional mapping between display city names and GeoJSON file names.

    Returns:
        (display_to_file, file_to_display) dictionaries.
        display_to_file["Addis Ababa"] -> "Addis_Ababa.geojson"
        file_to_display["Addis_Ababa.geojson"] -> "Addis Ababa"
    """
    if not os.path.isdir(osm_dir):
        logger.warning("OSM directory not found: %s", osm_dir)
        return {}, {}

    display_to_file: Dict[str, str] = {}
    file_to_display: Dict[str, str] = {}

    for fname in os.listdir(osm_dir):
        if not fname.endswith(".geojson"):
            continue
        file_stem = fname[: -len(".geojson")]
        # The file stem uses underscores; convert to spaces for display name
        display_name = file_stem.replace("_", " ")
        display_to_file[display_name] = os.path.join(osm_dir, fname)
        file_to_display[file_stem] = display_name

    logger.info(
        "Built city name map: %d OSM GeoJSON files in %s",
        len(display_to_file),
        osm_dir,
    )
    return display_to_file, file_to_display


def patch_id_to_city(patch_id: str) -> str:
    """Extract city name from a patch ID like 'Abidjan_0' or 'Addis Ababa_7'.

    The city name is everything before the last underscore-digit suffix.
    """
    # Handle edge case: city name might itself contain digits
    # Strategy: split on last underscore; if the last part is numeric,
    # it's the index; everything before is the city name.
    parts = patch_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    # If the patch_id has no trailing number, return as-is
    return patch_id


# ─── Loading OSM GeoJSON ─────────────────────────────────────────────


def load_city_geojson(geojson_path: str) -> Optional[gpd.GeoDataFrame]:
    """Load a city's OSM road network from a GeoJSON file.

    Filters to only include road LineStrings with a 'highway' property.

    Returns:
        GeoDataFrame with columns: geometry, highway, osm_id.
        None on failure.
    """
    if not os.path.isfile(geojson_path):
        logger.error("GeoJSON not found: %s", geojson_path)
        return None

    try:
        gdf = gpd.read_file(geojson_path)
    except Exception as e:
        logger.error("Failed to read GeoJSON %s: %s", geojson_path, e)
        return None

    if gdf.empty:
        logger.warning("Empty GeoDataFrame from %s", geojson_path)
        return gdf

    # Ensure highway column exists
    if "highway" not in gdf.columns:
        logger.error("No 'highway' column in %s", geojson_path)
        return None

    # Filter to LineString geometry
    mask = gdf.geometry.type.isin(["LineString", "MultiLineString"])
    road_gdf = gdf[mask].copy()

    # Ensure CRS is WGS84
    if road_gdf.crs is None:
        road_gdf.set_crs("EPSG:4326", inplace=True)

    # Simplify highway to scalar
    def _resolve_highway(hw):
        if isinstance(hw, (list, tuple, np.ndarray)):
            return hw[0] if len(hw) > 0 else "unclassified"
        return str(hw) if hw is not None else "unclassified"

    road_gdf["highway"] = road_gdf["highway"].apply(_resolve_highway)
    road_gdf["osm_id"] = road_gdf.get("osm_id", pd.Series(index=road_gdf.index, dtype=str))

    logger.debug(
        "Loaded %s: %d road features",
        os.path.basename(geojson_path),
        len(road_gdf),
    )
    return road_gdf


# ─── Bounding box computation ────────────────────────────────────────


def compute_bbox_latlon(
    center_lat: float, center_lon: float, context_size_m: float
) -> Tuple[float, float, float, float]:
    """Compute a lat/lon bounding box centered on (center_lat, center_lon).

    Args:
        center_lat: Center latitude in degrees.
        center_lon: Center longitude in degrees.
        context_size_m: Width/height of the square bounding box in meters.

    Returns:
        (minx, miny, maxx, maxy) in degrees.
    """
    half_side = context_size_m / 2.0
    dlat = half_side / METERS_PER_DEG_LAT
    dlon = half_side / meters_per_deg_lon(center_lat)
    return (center_lon - dlon, center_lat - dlat,
            center_lon + dlon, center_lat + dlat)


# ─── Spatial clipping ────────────────────────────────────────────────


def clip_gdf_to_bbox(
    gdf: gpd.GeoDataFrame, bbox: Tuple[float, float, float, float]
) -> gpd.GeoDataFrame:
    """Spatially clip a road GeoDataFrame to a lat/lon bounding box.

    Roads are clipped at the boundary so they don't extend outside.

    Args:
        gdf: Road GeoDataFrame with 'geometry' column.
        bbox: (minx, miny, maxx, maxy) in degrees.

    Returns:
        Clipped GeoDataFrame (may be empty).
    """
    minx, miny, maxx, maxy = bbox
    bbox_poly = box(minx, miny, maxx, maxy)

    if gdf.empty:
        return gdf

    # Use spatial index for fast filtering
    sindex = gdf.sindex
    possible = list(sindex.intersection(bbox_poly.bounds))
    if not possible:
        return gdf.iloc[:0]

    candidates = gdf.iloc[possible]
    clipped = candidates[candidates.intersects(bbox_poly)].copy()
    clipped.loc[:, "geometry"] = clipped.geometry.intersection(bbox_poly)
    # Remove empty geometries after clip
    clipped = clipped[~clipped.geometry.is_empty & clipped.geometry.is_valid]
    return clipped


# ─── Graph building from clipped LineStrings ─────────────────────────


def _round_coord(coord: Tuple[float, float], precision: int = 6) -> Tuple[float, float]:
    """Round a coordinate tuple to given decimal precision."""
    return (round(coord[0], precision), round(coord[1], precision))


def build_graph_from_gdf(
    gdf: gpd.GeoDataFrame,
    snap_precision: int = 6,
) -> nx.Graph:
    """Build a simplified undirected graph from a GeoDataFrame of road LineStrings.

    Nodes are created at start/end points of each LineString, snapped
    together via coordinate rounding (approx 0.1m at precision=6 near equator).

    Edge attributes:
        highway: Road type string.
        length_deg: Length in degrees (approximate).
        geometry: Shapely LineString of the edge.
        bearing_deg: Compass bearing (0-180°) of the straight-line connection.

    Args:
        gdf: GeoDataFrame with LineString geometries and 'highway' column.
        snap_precision: Decimal places for coordinate rounding (6 ≈ 0.1m).

    Returns:
        networkx.Graph (undirected).
    """
    G = nx.Graph()

    if gdf.empty:
        return G

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        # Handle MultiLineString by iterating over parts
        if geom.geom_type == "MultiLineString":
            lines = list(geom.geoms)
        else:
            lines = [geom]

        for line in lines:
            coords = list(line.coords)
            if len(coords) < 2:
                continue

            start = _round_coord(coords[0], snap_precision)
            end = _round_coord(coords[-1], snap_precision)

            highway = row.get("highway", "unclassified")

            # Add / update nodes
            for node, node_coord in [(start, coords[0]), (end, coords[-1])]:
                if node not in G:
                    G.add_node(node, x=node_coord[0], y=node_coord[1])

            # Compute bearing of the straight line between endpoints
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            bearing_rad = math.atan2(dx, dy)
            bearing_deg = math.degrees(bearing_rad) % 360
            # Normalize to [0, 180)
            if bearing_deg >= 180:
                bearing_deg -= 180

            # Compute curvature as ratio of path length to straight-line distance
            path_length_deg = line.length
            straight_dist = math.sqrt(dx ** 2 + dy ** 2)
            curvature = (path_length_deg / straight_dist) if straight_dist > 1e-10 else 1.0

            # Avoid duplicate edges (same start/end with same highway)
            edge_data = G.get_edge_data(start, end)
            if edge_data is not None:
                # Keep the shorter edge if we have duplicates
                existing_geom = edge_data.get("geometry", None)
                existing_length = existing_geom.length if existing_geom is not None else float("inf")
                if path_length_deg < existing_length:
                    G.remove_edge(start, end)
                    G.add_edge(start, end, highway=highway, length_deg=path_length_deg,
                               geometry=line, bearing_deg=bearing_deg, curvature=curvature)
            else:
                G.add_edge(start, end, highway=highway, length_deg=path_length_deg,
                           geometry=line, bearing_deg=bearing_deg, curvature=curvature)

    return G


# ─── Coordinate projection ───────────────────────────────────────────


def project_to_local_meters(
    lon: float, lat: float, center_lon: float, center_lat: float
) -> Tuple[float, float]:
    """Convert (lon, lat) to local Cartesian meters relative to a center point.

    Uses simple equirectangular approximation, accurate to ~0.1% within 2km.

    Returns:
        (x_m, y_m) in meters, with origin at (center_lon, center_lat).
    """
    m_per_lon = meters_per_deg_lon(center_lat)
    x_m = (lon - center_lon) * m_per_lon
    y_m = (lat - center_lat) * METERS_PER_DEG_LAT
    return x_m, y_m


def normalize_coords_to_01(
    x_m: float, y_m: float, context_size_m: float
) -> Tuple[float, float]:
    """Normalize local meter coordinates to [0, 1] within the context patch.

    Args:
        x_m: X coordinate in meters (origin at center).
        y_m: Y coordinate in meters (origin at center).
        context_size_m: Total width/height of the patch in meters.

    Returns:
        (x_norm, y_norm) in [0, 1].
    """
    half = context_size_m / 2.0
    x_n = (x_m + half) / context_size_m
    y_n = (y_m + half) / context_size_m
    return x_n, y_n


# ─── Bearing and curvature computation ──────────────────────────────


def compute_line_bearing_deg(line: LineString) -> Optional[float]:
    """Compute the compass bearing (0-180°) of a LineString.

    Returns bearing of the straight line from start to end, normalized to [0, 180).
    """
    if line is None or line.is_empty or len(line.coords) < 2:
        return None
    start = line.coords[0]
    end = line.coords[-1]
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    bearing_rad = math.atan2(dx, dy)
    bearing_deg = math.degrees(bearing_rad) % 180
    return bearing_deg


def estimate_lanes(highway: str) -> int:
    """Estimate number of lanes from highway type."""
    return ESTIMATED_LANES.get(highway, 1)


def estimate_width_m(highway: str) -> float:
    """Estimate road width in meters from highway type."""
    return ESTIMATED_WIDTH_M.get(highway, 5.0)


# ─── Metric: total road length in meters ────────────────────────────


def total_road_length_m(gdf: gpd.GeoDataFrame, center_lat: float) -> float:
    """Compute total road length in meters from a GeoDataFrame.

    Converts from degrees to meters using latitude approximation.
    """
    if gdf.empty:
        return 0.0
    m_per_deg = meters_per_deg_lon(center_lat)

    total = 0.0
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "MultiLineString":
            for line in geom.geoms:
                total += _line_length_m(line, m_per_deg, METERS_PER_DEG_LAT)
        else:
            total += _line_length_m(geom, m_per_deg, METERS_PER_DEG_LAT)
    return total


def _line_length_m(line: LineString, m_per_deg_lon: float, m_per_deg_lat: float) -> float:
    """Compute length of a LineString in meters using degree-to-meter scaling."""
    coords = list(line.coords)
    if len(coords) < 2:
        return 0.0
    total = 0.0
    for i in range(len(coords) - 1):
        dx = (coords[i + 1][0] - coords[i][0]) * m_per_deg_lon
        dy = (coords[i + 1][1] - coords[i][1]) * m_per_deg_lat
        total += math.sqrt(dx * dx + dy * dy)
    return total


# ─── Boundary detection ──────────────────────────────────────────────


def find_boundary_nodes(
    G: nx.Graph,
    center_lat: float,
    center_lon: float,
    context_size_m: float,
    boundary_tol_m: float = 30.0,
) -> List[Tuple[float, float]]:
    """Find nodes in the graph that are near the patch boundary.

    Args:
        G: Road graph.
        center_lat, center_lon: Center of the patch.
        context_size_m: Width/height of the patch.
        boundary_tol_m: Distance threshold from boundary in meters.

    Returns:
        List of boundary node coordinates (lon, lat).
    """
    half = context_size_m / 2.0
    m_per_lon = meters_per_deg_lon(center_lat)
    boundary_nodes = []

    for node, data in G.nodes(data=True):
        if data is None:
            continue
        lon, lat = node
        x_m = (lon - center_lon) * m_per_lon
        y_m = (lat - center_lat) * METERS_PER_DEG_LAT

        dist_to_boundary = min(
            abs(x_m - half),
            abs(x_m + half),
            abs(y_m - half),
            abs(y_m + half),
        )
        if dist_to_boundary <= boundary_tol_m:
            boundary_nodes.append(node)

    return boundary_nodes


# ─── Roundabout detection ──────────────────────────────────────────


def detect_roundabouts(skeleton_nodes: List[Dict], skeleton_edges: List[Dict]) -> set:
    """Detect roundabout nodes via topological cycle analysis.

    A roundabout is a small cycle where most nodes are degree-2 (the ring)
    with some degree-3/4 nodes at entry/exit points, forming a compact
    roughly-circular shape.

    Args:
        skeleton_nodes: List of node dicts from skeleton_graph_json.
                        Each must have 'id', 'degree', 'x_m', 'y_m'.
        skeleton_edges: List of edge dicts with 'source', 'target'.

    Returns:
        Set of node IDs belonging to detected roundabout cycles.
    """
    G = nx.Graph()
    node_deg: Dict[int, int] = {}
    node_xy: Dict[int, Tuple[float, float]] = {}

    for n in skeleton_nodes:
        nid = n["id"]
        G.add_node(nid)
        node_deg[nid] = n.get("degree", 0)
        node_xy[nid] = (n.get("x_m", 0.0), n.get("y_m", 0.0))

    for e in skeleton_edges:
        G.add_edge(e["source"], e["target"])

    try:
        cycles = nx.cycle_basis(G)
    except nx.NetworkXNoCycle:
        return set()

    roundabout_ids: set = set()

    for cycle in cycles:
        # Roundabouts are small cycles (3–12 nodes)
        if len(cycle) < 3 or len(cycle) > 12:
            continue

        # Degree check: at least 40 % of nodes should be degree-2
        degs = [node_deg.get(n, 0) for n in cycle]
        deg2_count = sum(1 for d in degs if d == 2)
        if deg2_count < len(cycle) * 0.4:
            continue

        # Spatial compactness: compute centroid and CV of distances
        xs = [node_xy[n][0] for n in cycle]
        ys = [node_xy[n][1] for n in cycle]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        dists = [math.sqrt((x - cx) ** 2 + (y - cy) ** 2) for x, y in zip(xs, ys)]
        mean_d = sum(dists) / len(dists)
        if mean_d < 0.1:  # degenerate: all nodes at the same point
            continue
        var_d = sum((d - mean_d) ** 2 for d in dists) / len(dists)
        std_d = math.sqrt(var_d)
        cv = std_d / mean_d

        # CV < 0.5 means nodes are roughly circular (not elongated / block-like)
        if cv < 0.5:
            roundabout_ids.update(cycle)

    return roundabout_ids

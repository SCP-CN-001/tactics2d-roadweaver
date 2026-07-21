"""
OSM road graph construction and analysis utilities.

Functions for loading, building, clipping, and analyzing
OpenStreetMap road networks for structural prior computation.
"""

from __future__ import annotations

import logging
import math
import os

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, box

logger = logging.getLogger(__name__)

# ── Road hierarchy ────────────────────────────────────────────────────

ROAD_HIERARCHY = {
    "motorway": 5,
    "motorway_link": 4,
    "trunk": 5,
    "trunk_link": 4,
    "primary": 4,
    "primary_link": 3,
    "secondary": 3,
    "secondary_link": 2,
    "tertiary": 2,
    "tertiary_link": 1,
    "residential": 1,
    "service": 1,
    "living_street": 1,
    "unclassified": 1,
}

MAJOR_ROAD_TYPES = {"motorway", "trunk", "primary", "secondary"}
SKELETON_ROAD_TYPES = {"motorway", "trunk", "primary", "secondary"}

ESTIMATED_LANES = {
    "motorway": 4,
    "motorway_link": 2,
    "trunk": 4,
    "trunk_link": 2,
    "primary": 3,
    "primary_link": 2,
    "secondary": 2,
    "secondary_link": 1,
    "tertiary": 2,
    "tertiary_link": 1,
    "residential": 1,
    "service": 1,
    "living_street": 1,
    "unclassified": 1,
}

ESTIMATED_WIDTH_M = {
    "motorway": 25.0,
    "motorway_link": 10.0,
    "trunk": 20.0,
    "trunk_link": 10.0,
    "primary": 15.0,
    "primary_link": 8.0,
    "secondary": 10.0,
    "secondary_link": 6.0,
    "tertiary": 8.0,
    "tertiary_link": 5.0,
    "residential": 6.0,
    "service": 4.0,
    "living_street": 4.0,
    "unclassified": 5.0,
}

METERS_PER_DEG_LAT = 111320.0


def meters_per_deg_lon(lat: float) -> float:
    """Meters per degree longitude at a given latitude."""
    return METERS_PER_DEG_LAT * abs(math.cos(math.radians(lat))) + 1e-12


# ── City name matching ────────────────────────────────────────────────


def build_city_name_map(osm_dir: str = "data/osm") -> tuple[dict[str, str], dict[str, str]]:
    """Build bidirectional mapping between city names and GeoJSON file names.

    Returns:
        (display_to_file, file_to_display) e.g.
        display_to_file["Addis Ababa"] -> "Addis_Ababa.geojson"
    """
    if not os.path.isdir(osm_dir):
        logger.warning("OSM directory not found: %s", osm_dir)
        return {}, {}

    display_to_file: dict[str, str] = {}
    file_to_display: dict[str, str] = {}

    for fname in os.listdir(osm_dir):
        if not fname.endswith(".geojson"):
            continue
        file_stem = fname[: -len(".geojson")]
        display_name = file_stem.replace("_", " ")
        display_to_file[display_name] = os.path.join(osm_dir, fname)
        file_to_display[file_stem] = display_name

    logger.info("Built city name map: %d OSM files in %s", len(display_to_file), osm_dir)
    return display_to_file, file_to_display


def patch_id_to_city(patch_id: str) -> str:
    """Extract city name from a patch ID like 'Abidjan_0' or 'Addis Ababa_7'.

    The city name is everything before the last underscore-digit suffix.
    """
    parts = patch_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return patch_id


# ── OSM GeoJSON loading ───────────────────────────────────────────────


def load_city_geojson(geojson_path: str) -> gpd.GeoDataFrame | None:
    """Load a city's OSM road network from a GeoJSON file.

    Filters to LineString road features with a 'highway' property.
    Returns None on failure.
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

    if "highway" not in gdf.columns:
        logger.error("No 'highway' column in %s", geojson_path)
        return None

    mask = gdf.geometry.type.isin(["LineString", "MultiLineString"])
    road_gdf = gdf[mask].copy()

    if road_gdf.crs is None:
        road_gdf.set_crs("EPSG:4326", inplace=True)

    def _resolve_highway(hw):
        if isinstance(hw, (list, tuple, np.ndarray)):
            return hw[0] if len(hw) > 0 else "unclassified"
        return str(hw) if hw is not None else "unclassified"

    road_gdf["highway"] = road_gdf["highway"].apply(_resolve_highway)
    road_gdf["osm_id"] = road_gdf.get("osm_id", pd.Series(index=road_gdf.index, dtype=str))

    logger.debug("Loaded %s: %d road features", os.path.basename(geojson_path), len(road_gdf))
    return road_gdf


# ── Bounding box ──────────────────────────────────────────────────────


def compute_bbox_latlon(
    center_lat: float, center_lon: float, context_size_m: float
) -> tuple[float, float, float, float]:
    """Lat/lon bounding box centered on (center_lat, center_lon)."""
    half_side = context_size_m / 2.0
    dlat = half_side / METERS_PER_DEG_LAT
    dlon = half_side / meters_per_deg_lon(center_lat)
    return (center_lon - dlon, center_lat - dlat, center_lon + dlon, center_lat + dlat)


# ── Spatial clipping ──────────────────────────────────────────────────


def clip_gdf_to_bbox(
    gdf: gpd.GeoDataFrame, bbox: tuple[float, float, float, float]
) -> gpd.GeoDataFrame:
    """Clip road GeoDataFrame to a lat/lon bounding box.

    Roads are clipped at the boundary so they don't extend outside.
    """
    minx, miny, maxx, maxy = bbox
    bbox_poly = box(minx, miny, maxx, maxy)

    if gdf.empty:
        return gdf

    sindex = gdf.sindex
    possible = list(sindex.intersection(bbox_poly.bounds))
    if not possible:
        return gdf.iloc[:0]

    candidates = gdf.iloc[possible]
    clipped = candidates[candidates.intersects(bbox_poly)].copy()
    clipped.loc[:, "geometry"] = clipped.geometry.intersection(bbox_poly)
    clipped = clipped[~clipped.geometry.is_empty & clipped.geometry.is_valid]
    return clipped


# ── Graph building from clipped LineStrings ───────────────────────────


def _round_coord(coord: tuple[float, float], precision: int = 6) -> tuple[float, float]:
    """Round a coordinate tuple to given decimal precision (6 ≈ 0.1 m)."""
    return (round(coord[0], precision), round(coord[1], precision))


def build_graph_from_gdf(gdf: gpd.GeoDataFrame, snap_precision: int = 6) -> nx.Graph:
    """Build an undirected road graph from a GeoDataFrame of LineStrings.

    Nodes are created at start/end points of each LineString, snapped
    together via coordinate rounding.  Edge attributes include highway
    type, length, geometry, bearing, and curvature.
    """
    G = nx.Graph()

    if gdf.empty:
        return G

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        lines = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]

        for line in lines:
            coords = list(line.coords)
            if len(coords) < 2:
                continue

            start = _round_coord(coords[0], snap_precision)
            end = _round_coord(coords[-1], snap_precision)
            highway = row.get("highway", "unclassified")

            for node, node_coord in [(start, coords[0]), (end, coords[-1])]:
                if node not in G:
                    G.add_node(node, x=node_coord[0], y=node_coord[1])

            dx = end[0] - start[0]
            dy = end[1] - start[1]
            bearing_rad = math.atan2(dx, dy)
            bearing_deg = math.degrees(bearing_rad) % 360
            if bearing_deg >= 180:
                bearing_deg -= 180

            path_length = line.length
            straight_dist = math.sqrt(dx**2 + dy**2)
            curvature = (path_length / straight_dist) if straight_dist > 1e-10 else 1.0

            edge_data = G.get_edge_data(start, end)
            if edge_data is not None:
                existing_geom = edge_data.get("geometry")
                existing_length = (
                    existing_geom.length if existing_geom is not None else float("inf")
                )
                if path_length < existing_length:
                    G.remove_edge(start, end)
                    G.add_edge(
                        start,
                        end,
                        highway=highway,
                        length_deg=path_length,
                        geometry=line,
                        bearing_deg=bearing_deg,
                        curvature=curvature,
                    )
            else:
                G.add_edge(
                    start,
                    end,
                    highway=highway,
                    length_deg=path_length,
                    geometry=line,
                    bearing_deg=bearing_deg,
                    curvature=curvature,
                )

    return G


# ── Coordinate projection ─────────────────────────────────────────────


def project_to_local_meters(
    lon: float, lat: float, center_lon: float, center_lat: float
) -> tuple[float, float]:
    """Convert (lon, lat) to local Cartesian metres relative to a centre point.

    Equirectangular approximation, accurate to ~0.1 % within 2 km.
    """
    return (
        (lon - center_lon) * meters_per_deg_lon(center_lat),
        (lat - center_lat) * METERS_PER_DEG_LAT,
    )


def normalize_coords_to_01(x_m: float, y_m: float, context_size_m: float) -> tuple[float, float]:
    """Normalise local metre coordinates to [0, 1] within the patch."""
    half = context_size_m / 2.0
    return (x_m + half) / context_size_m, (y_m + half) / context_size_m


# ── Bearing and curvature ─────────────────────────────────────────────


def compute_line_bearing_deg(line: LineString) -> float | None:
    """Compass bearing (0-180°) of a LineString's straight-line connection."""
    if line is None or line.is_empty or len(line.coords) < 2:
        return None
    start = line.coords[0]
    end = line.coords[-1]
    bearing_rad = math.atan2(end[0] - start[0], end[1] - start[1])
    return math.degrees(bearing_rad) % 180


def estimate_lanes(road_type: str) -> int:
    """Estimate number of lanes from road type."""
    return ESTIMATED_LANES.get(road_type, 1)


def estimate_width_m(road_type: str) -> float:
    """Estimate road width in metres from road type."""
    return ESTIMATED_WIDTH_M.get(road_type, 5.0)


# ── Road length ───────────────────────────────────────────────────────


def _line_length_m(line: LineString, m_per_deg_lon: float, m_per_deg_lat: float) -> float:
    """Length of a LineString in metres using degree-to-metre scaling."""
    coords = list(line.coords)
    if len(coords) < 2:
        return 0.0
    total = 0.0
    for i in range(len(coords) - 1):
        dx = (coords[i + 1][0] - coords[i][0]) * m_per_deg_lon
        dy = (coords[i + 1][1] - coords[i][1]) * m_per_deg_lat
        total += math.sqrt(dx * dx + dy * dy)
    return total


def total_road_length_m(gdf: gpd.GeoDataFrame, center_lat: float) -> float:
    """Total road length in metres for a GeoDataFrame."""
    if gdf.empty:
        return 0.0
    m_per_lon = meters_per_deg_lon(center_lat)
    total = 0.0
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "MultiLineString":
            for line in geom.geoms:
                total += _line_length_m(line, m_per_lon, METERS_PER_DEG_LAT)
        else:
            total += _line_length_m(geom, m_per_lon, METERS_PER_DEG_LAT)
    return total


# ── Boundary detection ────────────────────────────────────────────────


def find_boundary_nodes(
    G: nx.Graph,
    center_lat: float,
    center_lon: float,
    context_size_m: float,
    boundary_tol_m: float = 30.0,
) -> list[tuple[float, float]]:
    """Find graph nodes near the patch boundary (within *boundary_tol_m*)."""
    half = context_size_m / 2.0
    m_per_lon = meters_per_deg_lon(center_lat)
    boundary_nodes = []

    for node, data in G.nodes(data=True):
        if data is None:
            continue
        lon, lat = node
        x_m = (lon - center_lon) * m_per_lon
        y_m = (lat - center_lat) * METERS_PER_DEG_LAT

        dist_to_boundary = min(abs(x_m - half), abs(x_m + half), abs(y_m - half), abs(y_m + half))
        if dist_to_boundary <= boundary_tol_m:
            boundary_nodes.append(node)

    return boundary_nodes


def sanitize_city_name(name: str) -> str:
    """Replace unsafe filesystem characters in a city name."""
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_")


# ── Roundabout detection (skeleton JSON) ──────────────────────────────


def detect_roundabouts(skeleton_nodes: list[dict], skeleton_edges: list[dict]) -> set:
    """Detect roundabout nodes from skeleton graph JSON via cycle analysis.

    A roundabout is a compact cycle (3-12 nodes, CV < 0.5) where
    at least 40 % of nodes have degree 2.
    """
    G = nx.Graph()
    node_deg: dict[int, int] = {}
    node_xy: dict[int, tuple[float, float]] = {}

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
        if len(cycle) < 3 or len(cycle) > 12:
            continue

        degs = [node_deg.get(n, 0) for n in cycle]
        if sum(1 for d in degs if d == 2) < len(cycle) * 0.4:
            continue

        xs = [node_xy[n][0] for n in cycle]
        ys = [node_xy[n][1] for n in cycle]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        dists = [math.sqrt((x - cx) ** 2 + (y - cy) ** 2) for x, y in zip(xs, ys)]
        mean_d = sum(dists) / len(dists)
        if mean_d < 0.1:
            continue
        std_d = math.sqrt(sum((d - mean_d) ** 2 for d in dists) / len(dists))

        if std_d / mean_d < 0.5:
            roundabout_ids.update(cycle)

    return roundabout_ids

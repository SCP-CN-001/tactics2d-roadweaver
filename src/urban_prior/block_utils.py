"""Block (urban parcel) extraction from road networks.

Pipelines a road network GeoDataFrame and bounding box into block
polygons using shapely's polygonize operation.

Block: a contiguous area enclosed by roads and/or the patch boundary.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import polygonize, unary_union

logger = logging.getLogger(__name__)


def extract_blocks(
    road_gdf,
    bbox_poly: Polygon,
    simplify_tol_deg: float = 1e-5,
) -> Tuple[List[Polygon], Optional[str]]:
    """Extract block polygons from a clipped road network.

    The algorithm:
    1. Merge all road geometries into a single geometry collection.
    2. Union the roads with the bounding box boundary.
    3. Run shapely polygonize on the combined linework.
    4. Filter resulting polygons to those inside the bounding box.

    Args:
        road_gdf: GeoDataFrame of road LineStrings (already clipped to bbox).
        bbox_poly: Shapely Polygon of the bounding box.
        simplify_tol_deg: Simplification tolerance in degrees (default ~1m).

    Returns:
        (blocks, error_msg): List of block polygons, or None error message.
    """
    if road_gdf is None or road_gdf.empty:
        return [], "No road data for block extraction"

    try:
        # Collect all road geometries
        geoms = []
        for geom in road_gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type == "MultiLineString":
                for g in geom.geoms:
                    if not g.is_empty:
                        geoms.append(g)
            else:
                geoms.append(geom)

        if not geoms:
            return [], "No valid road geometries"

        # Simplify roads slightly to avoid topology errors
        if simplify_tol_deg > 0:
            geoms = [g.simplify(simplify_tol_deg, preserve_topology=True) for g in geoms]

        # Merge all roads into a single line collection
        merged_roads = unary_union(geoms)

        # Add the bounding box boundary so blocks are bounded at patch edges
        bbox_boundary = bbox_poly.boundary

        # Combine roads and boundary for polygonization
        all_lines = unary_union([merged_roads, bbox_boundary])

        # Polygonize
        polygons = list(polygonize(all_lines))

        if not polygons:
            return [], "Polygonize produced no polygons"

        # Filter to only polygons inside/on the bbox (exclude the outer "border" polygon)
        blocks = []
        for poly in polygons:
            if poly.is_valid and not poly.is_empty:
                centroid = poly.centroid
                if bbox_poly.contains(centroid) or bbox_poly.touches(centroid):
                    # Exclude very small sliver polygons (area < 1 m^2)
                    blocks.append(poly)

        if not blocks:
            return [], "No valid blocks after filtering"

        logger.debug("Extracted %d blocks", len(blocks))
        return blocks, None

    except Exception as e:
        error_msg = f"Block polygonization failed: {e}"
        logger.warning(error_msg)
        return [], error_msg


def compute_block_prior(blocks: List[Polygon], center_lat: float) -> Dict:
    """Compute block prior statistics from block polygons.

    Uses equirectangular approximation to compute area in square meters.

    Args:
        blocks: List of Shapely Polygon objects.
        center_lat: Center latitude for meter-per-degree scaling.

    Returns:
        Dictionary with block prior metrics, or empty dict if no blocks.
    """
    if not blocks:
        return {}

    from src.urban_prior.graph_utils import meters_per_deg_lon, METERS_PER_DEG_LAT

    m_per_lon = meters_per_deg_lon(center_lat)
    m_per_lat = METERS_PER_DEG_LAT
    scale_factor = m_per_lon * m_per_lat

    areas_m2 = []
    aspect_ratios = []
    compactnesses = []

    for poly in blocks:
        area_deg = poly.area
        area_m2 = area_deg * scale_factor
        if area_m2 < 1.0:
            continue  # skip slivers
        areas_m2.append(area_m2)

        # Aspect ratio: ratio of minimum rotated rectangle sides
        min_rect = poly.minimum_rotated_rectangle
        if min_rect is not None and not min_rect.is_empty:
            try:
                coords = list(min_rect.exterior.coords)[:-1]
                if len(coords) >= 4:
                    # Compute side lengths of the rotated rectangle
                    side_lengths = []
                    for i in range(len(coords)):
                        p1 = coords[i]
                        p2 = coords[(i + 1) % len(coords)]
                        dx = (p2[0] - p1[0]) * m_per_lon
                        dy = (p2[1] - p1[1]) * m_per_lat
                        side_lengths.append(math.sqrt(dx * dx + dy * dy))
                    # Group into two unique side lengths
                    if len(side_lengths) >= 2:
                        s1 = min(side_lengths[0], side_lengths[2]) if len(side_lengths) > 2 else side_lengths[0]
                        s2 = min(side_lengths[1], side_lengths[3]) if len(side_lengths) > 3 else side_lengths[1]
                        if s1 > 0 and s2 > 0:
                            aspect = max(s1, s2) / min(s1, s2)
                            aspect_ratios.append(aspect)
            except Exception:
                pass

        # Compactness: 4*pi*area / perimeter^2 (circle = 1)
        perimeter_m = poly.length * math.sqrt(scale_factor)
        if perimeter_m > 0:
            compactness = (4 * math.pi * area_m2) / (perimeter_m * perimeter_m)
            compactnesses.append(compactness)

    if not areas_m2:
        return {}

    areas_arr = np.array(areas_m2)
    block_prior = {
        "block_count": len(blocks),
        "block_area_mean_m2": float(np.mean(areas_arr)),
        "block_area_std_m2": float(np.std(areas_arr)),
        "block_area_median_m2": float(np.median(areas_arr)),
        "block_scale_m": float(math.sqrt(np.median(areas_arr))),
    }

    if aspect_ratios:
        block_prior["block_aspect_ratio_mean"] = float(np.mean(aspect_ratios))
    else:
        block_prior["block_aspect_ratio_mean"] = None

    if compactnesses:
        block_prior["block_compactness_mean"] = float(np.mean(compactnesses))
    else:
        block_prior["block_compactness_mean"] = None

    return block_prior

"""CRHD (Color Road Hierarchy Diagram) Generator.

Works with osmnx 2.x and shapely 2.x by avoiding the problematic
polygon projection path in osmnx's graph_from_point.
"""

import math
import os
import sys
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import osmnx as ox
from shapely.geometry import box

# CRHD rendering config (from original crhd_generator_v2)
STREET_TYPES = [
    'service', 'residential', 'tertiary_link', 'tertiary',
    'secondary_link', 'primary_link', 'motorway_link',
    'secondary', 'trunk', 'primary', 'motorway',
]

STREET_WIDTHS = {
    'service': 1, 'residential': 1, 'tertiary_link': 1,
    'tertiary': 2, 'secondary_link': 2, 'primary_link': 2,
    'motorway_link': 2, 'secondary': 3, 'trunk': 4,
    'primary': 4, 'motorway': 2.5,
}

# Colors for white background
STREET_COLORS = {
    'service': 'skyblue', 'residential': 'skyblue',
    'tertiary_link': 'skyblue', 'tertiary': 'cornflowerblue',
    'secondary_link': 'cornflowerblue', 'primary_link': 'cornflowerblue',
    'trunk_link': 'cornflowerblue', 'motorway_link': 'darkred',
    'secondary': 'darkblue', 'trunk': 'black',
    'primary': 'black', 'motorway': 'darkred',
}


def _make_bbox_polygon(lon: float, lat: float, dist_m: float) -> box:
    """Create a valid WGS84 bounding box polygon around a point.

    This avoids osmnx's internal projection+buffer which can fail
    with shapely 2.x + GEOS 3.12 (especially near the equator).
    """
    deg_per_m_lat = 1.0 / 111320.0
    deg_per_m_lon = 1.0 / (111320.0 * abs(math.cos(math.radians(lat))) + 1e-12)
    half_lon = dist_m * deg_per_m_lon
    half_lat = dist_m * deg_per_m_lat
    return box(lon - half_lon, lat - half_lat,
               lon + half_lon, lat + half_lat)


def graph_from_point(lon: float, lat: float, dist: int = 1000,
                     network_type: str = 'all'):
    """Download OSMnx graph around a point, with shapely 2.x compat."""
    poly = _make_bbox_polygon(lon, lat, dist)
    return ox.graph_from_polygon(poly, network_type=network_type)


def graph_to_crhd(
    graph,
    output_path: str,
    figsize: Tuple[int, int] = (6, 6),
    dpi: int = 100,
    format: str = 'png',
):
    """Convert an OSMnx road graph into a CRHD image file.

    Args:
        graph: An OSMnx MultiDiGraph.
        output_path: Where to save the CRHD image.
        figsize: Matplotlib figure size (inches).
        dpi: Image resolution.
        format: Image format (png, jpg, etc.).
    """
    gdf = ox.graph_to_gdfs(graph, nodes=False, edges=True)
    gdf.highway = gdf.highway.map(lambda x: x[0] if isinstance(x, list) else x)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    gdf.plot(ax=ax, linewidth=0.5, edgecolor='lightgreen')
    for stype in STREET_TYPES:
        if (gdf.highway == stype).any():
            gdf[gdf.highway == stype].plot(
                ax=ax, linewidth=STREET_WIDTHS[stype],
                edgecolor=STREET_COLORS[stype],
            )
    plt.axis('off')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                pad_inches=0, format=format)
    plt.close()


def load_osmnx_graph(input_path: str):
    """Load an OSMnx graph from various file formats.

    Supports: .graphml, .gpkg, .pkl/.pickle, .osm/.xml.
    """
    ext = os.path.splitext(input_path)[1].lower()
    if ext == '.graphml':
        return ox.load_graphml(input_path)
    elif ext == '.gpkg':
        G = ox.graph_from_xml(input_path)
        return G
    elif ext in ('.pkl', '.pickle'):
        import pickle
        with open(input_path, 'rb') as f:
            return pickle.load(f)
    elif ext in ('.osm', '.xml'):
        return ox.graph_from_xml(input_path)
    else:
        raise ValueError(f'Unsupported graph format: {ext}')


def generate_centroid_crhd(
    lon: float,
    lat: float,
    output_path: str,
    dist: int = 1000,
    figsize: Tuple[int, int] = (6, 6),
    dpi: int = 100,
    network_type: str = 'all',
) -> bool:
    """Download OSM data around a point and save a CRHD image.

    Returns:
        True on success, False on failure.
    """
    try:
        graph = graph_from_point(lon, lat, dist, network_type)
        graph_to_crhd(graph, output_path, figsize, dpi)
        return True
    except Exception as e:
        print(f'  [warn] Failed for ({lat:.4f}, {lon:.4f}): {e}')
        return False


def batch_generate_crhd(
    input_dir: str,
    output_dir: str,
    manifest_path: Optional[str] = None,
    image_size: int = 512,
    dpi: int = 100,
    limit: Optional[int] = None,
):
    """Batch-generate CRHDs from classified grid shapefiles.

    Args:
        input_dir: Directory containing city grid shapefiles (e.g., data/original_grids/classified_grids_of_cities).
        output_dir: Where to save CRHD images.
        manifest_path: Optional path to save a JSON manifest.
        image_size: Image size in pixels (used for figsize calculation).
        dpi: Image DPI.
        limit: Maximum number of CRHDs to generate (for testing).
    """
    import geopandas as gpd
    import json

    os.makedirs(output_dir, exist_ok=True)
    figsize = (image_size / dpi, image_size / dpi)
    city_dirs = sorted([
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ])

    manifest = []
    total_generated = 0
    total_failed = 0

    for city in city_dirs:
        shp_files = [
            f for f in os.listdir(os.path.join(input_dir, city))
            if f.endswith('.shp')
        ]
        if not shp_files:
            continue

        gdf = gpd.read_file(os.path.join(input_dir, city, shp_files[0]))
        print(f'[{city}] {len(gdf)} grid cells')

        for idx, row in gdf.iterrows():
            if limit and total_generated + total_failed >= limit:
                break

            centroid = row.geometry.centroid
            lat, lon = centroid.y, centroid.x
            label = row.get('mid_cls', 'Unknown')
            img_name = f'{row["id"]}.png'
            img_path = os.path.join(output_dir, img_name)
            rel_path = os.path.relpath(img_path)

            success = generate_centroid_crhd(
                lon, lat, img_path,
                dist=1000, figsize=figsize, dpi=dpi,
            )

            manifest.append({
                'source_path': row['id'],
                'crhd_path': rel_path if success else None,
                'city': city,
                'lat': lat,
                'lon': lon,
                'label': label,
                'status': 'success' if success else 'failed',
            })

            if success:
                total_generated += 1
            else:
                total_failed += 1

            if (total_generated + total_failed) % 10 == 0:
                print(f'  ... {total_generated} OK, {total_failed} failed')

        if limit and total_generated + total_failed >= limit:
            break

    print(f'\nTotal generated: {total_generated}')
    print(f'Total failed: {total_failed}')

    if manifest_path:
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f'Manifest saved to: {manifest_path}')

    return manifest, total_generated, total_failed


# ─── Local clipping mode (download OSM once per city, clip per grid cell) ───


def _clip_gdf_to_bbox(gdf_edges, bbox_polygon):
    """Spatially clip an edge GeoDataFrame to a bounding box polygon.

    Uses R-tree spatial index for fast filtering, then clips geometries
    to the bbox boundary so roads don't extend outside the cell.
    """
    sindex = gdf_edges.sindex
    possible = list(sindex.intersection(bbox_polygon.bounds))
    if not possible:
        return gdf_edges.iloc[:0]
    candidates = gdf_edges.iloc[possible]
    clipped = candidates[candidates.intersects(bbox_polygon)].copy()
    clipped.loc[:, 'geometry'] = clipped.geometry.intersection(bbox_polygon)
    return clipped[~clipped.geometry.is_empty & ~clipped.geometry.isna()]


# ─── Direct Overpass queries (bypassing osmnx's urllib3 connection pool) ───


OVERPASS_ENDPOINTS = [
    'https://overpass-api.de/api/interpreter',
]

OVERPASS_HEADERS = {
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
    'User-Agent': 'RoadWeaver/1.0',
}

import requests as _requests
import time as _time

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def _overpass_roads_gdf(bbox, endpoint=None, timeout=60):
    """Query Overpass API via direct requests.post() and return road GDF.

    Handles 429 (rate limit) with automatic backoff up to 5 retries.

    Args:
        bbox: (west, south, east, north) in degrees.
        endpoint: Overpass API URL (default: OVERPASS_ENDPOINTS[0]).
        timeout: Request timeout in seconds.

    Returns:
        GeoDataFrame of road edges with 'highway' column, or None on failure.
    """
    import geopandas as _gpd
    from shapely.geometry import LineString as _LineString

    if endpoint is None:
        endpoint = OVERPASS_ENDPOINTS[0]

    west, south, east, north = bbox
    query = (
        f'[out:json][timeout:{min(timeout, 60)}];'
        f'(way["highway"]({south},{west},{north},{east}););'
        f'out geom;'
    )

    for attempt in range(5):
        try:
            r = _requests.post(
                endpoint, data=query, timeout=timeout,
                verify=False, headers=OVERPASS_HEADERS,
            )
            if r.status_code == 429:
                wait = (attempt + 1) * 10
                print(f'  429 rate limit, waiting {wait}s ...', end=' ', flush=True)
                _time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            if attempt < 4:
                _time.sleep(5)
                continue
            print(f'  [warn] Overpass request failed: {e}')
            return None
    else:
        return None

    elements = data.get('elements', [])
    rows = []
    for el in elements:
        if el.get('type') != 'way':
            continue
        geom_coords = el.get('geometry')
        if not geom_coords:
            continue
        coords = [(p['lon'], p['lat']) for p in geom_coords]
        if len(coords) < 2:
            continue
        highway = el.get('tags', {}).get('highway')
        if highway is None:
            continue
        rows.append({
            'geometry': _LineString(coords),
            'highway': highway,
            'osm_id': el['id'],
        })

    if not rows:
        return _gpd.GeoDataFrame(
            [], columns=['geometry', 'highway', 'osm_id'],
            geometry='geometry', crs='EPSG:4326',
        )

    gdf = _gpd.GeoDataFrame(rows, geometry='geometry', crs='EPSG:4326')
    return gdf


def _download_city_roads_adaptive(
    bounds,
    pad: float = 0.01,
    timeout: int = 60,
    max_splits: int = 4,
    tile_delay: float = 3.0,
):
    """Download OSM roads with adaptive tiling.

    Tries the full bounding box first. If the query fails (too large),
    progressively splits into 4, 9, 16, … tiles until all succeed or
    *max_splits* is reached.  Adds *tile_delay* seconds between tiles to
    respect Overpass API rate limits.

    Args:
        bounds: (minx, miny, maxx, maxy) in degrees.
        pad: Extra padding (degrees).
        timeout: Per-query timeout in seconds.
        max_splits: Maximum tile splits (1=full bbox, 2=2x2, …).
        tile_delay: Seconds to wait between consecutive tiles.

    Returns:
        Combined GeoDataFrame, or None if all attempts failed.
    """
    minx, miny, maxx, maxy = bounds
    minx -= pad
    miny -= pad
    maxx += pad
    maxy += pad

    for splits in range(1, max_splits + 1):
        cols = splits
        rows = splits
        tile_w = (maxx - minx) / cols
        tile_h = (maxy - miny) / rows

        tile_bboxes = []
        for r in range(rows):
            for c in range(cols):
                tile_bboxes.append((
                    minx + c * tile_w,
                    miny + r * tile_h,
                    minx + (c + 1) * tile_w,
                    miny + (r + 1) * tile_h,
                ))

        ntiles = len(tile_bboxes)
        print(f'  splitting {splits}x{splits} ({ntiles} tiles) ...',
              end=' ', flush=True)

        all_gdfs = []
        failed = 0
        for ti, bbox in enumerate(tile_bboxes):
            if ti > 0:
                _time.sleep(tile_delay)
            gdf = _overpass_roads_gdf(bbox, timeout=timeout)
            if gdf is not None and not gdf.empty:
                all_gdfs.append(gdf)
            else:
                failed += 1

        if failed == 0:
            import pandas as _pd
            combined = _pd.concat(all_gdfs, ignore_index=True)
            combined = combined.drop_duplicates(subset='geometry')
            print(f'{len(combined)} roads')
            return combined

        print(f'{failed}/{ntiles} tiles failed')
        if splits >= max_splits:
            return None

    return None


def _render_gdf(
    gdf,
    output_path: str,
    figsize: Tuple[float, float] = (5.12, 5.12),
    dpi: int = 100,
    format: str = 'png',
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
):
    """Render a road GeoDataFrame as a CRHD image file (no OSM download).

    Unlike graph_to_crhd(), this takes a pre-clipped GeoDataFrame directly.
    Use xlim/ylim to lock axes to the cell bounding box for consistent scale.
    """
    gdf.highway = gdf.highway.map(lambda x: x[0] if isinstance(x, list) else x)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if not gdf.empty:
        gdf.plot(ax=ax, linewidth=0.5, edgecolor='lightgreen')
        for stype in STREET_TYPES:
            if (gdf.highway == stype).any():
                gdf[gdf.highway == stype].plot(
                    ax=ax, linewidth=STREET_WIDTHS[stype],
                    edgecolor=STREET_COLORS[stype],
                )
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    plt.axis('off')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                pad_inches=0, format=format)
    plt.close()


def _sanitize_city_name(name: str) -> str:
    """Replace unsafe filesystem characters in a city name."""
    return name.replace(' ', '_').replace('/', '_').replace('\\', '_')


def _verify_geojson(filepath: str) -> bool:
    """Verify a GeoJSON file is valid and contains road data."""
    try:
        import geopandas as _gpd
        gdf = _gpd.read_file(filepath)
        return len(gdf) > 0
    except Exception:
        return False


def download_osm_cache(
    input_dir: str,
    cache_dir: str,
    osm_dir: Optional[str] = None,
    city_limit: Optional[int] = None,
):
    """Phase 1: download OSM road networks and cache as GeoJSON.

    Downloads to *cache_dir* first.  If *osm_dir* is given, verifies each
    cached file and moves it to *osm_dir* on success.  Skips cities that
    already have a file in *osm_dir* (or in *cache_dir* if *osm_dir* is
    not set).

    Args:
        input_dir: Directory containing classified grid shapefiles.
        cache_dir: Temporary directory for GeoJSON cache.
        osm_dir: Permanent directory for verified GeoJSON files.  When set,
            cached files are moved here after successful verification.
        city_limit: Max cities to process (default: all).
    """
    import geopandas as gpd
    import shutil

    os.makedirs(cache_dir, exist_ok=True)
    if osm_dir:
        os.makedirs(osm_dir, exist_ok=True)

    city_dirs = sorted([
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ])

    # When osm_dir is set, pre-filter to only cities not already there.
    # This way city_limit only counts cities that actually need downloading.
    if osm_dir:
        filtered = []
        for city in city_dirs:
            safe = _sanitize_city_name(city)
            osm_path = os.path.join(osm_dir, f'{safe}.geojson')
            if not os.path.exists(osm_path):
                filtered.append(city)
        city_dirs = filtered
        if city_limit:
            city_dirs = city_dirs[:city_limit]
    else:
        if city_limit:
            city_dirs = city_dirs[:city_limit]

    for city_idx, city in enumerate(city_dirs):

        shp_files = [
            f for f in os.listdir(os.path.join(input_dir, city))
            if f.endswith('.shp')
        ]
        if not shp_files:
            continue

        safe = _sanitize_city_name(city)

        # Skip if already in the permanent osm dir
        if osm_dir:
            osm_path = os.path.join(osm_dir, f'{safe}.geojson')
            if os.path.exists(osm_path):
                print(f'[{city}] already in osm/, skip')
                continue

        cache_path = os.path.join(cache_dir, f'{safe}.geojson')
        if os.path.exists(cache_path):
            # Cache exists but hasn't been moved yet — verify and move
            print(f'[{city}] cached, verifying ...', end=' ', flush=True)
            if _verify_geojson(cache_path):
                if osm_dir:
                    shutil.move(cache_path, osm_path)
                    print(f'moved to osm/')
                else:
                    print('ok')
            else:
                print('invalid, re-downloading')
                os.remove(cache_path)
            continue

        gdf_grid = gpd.read_file(os.path.join(input_dir, city, shp_files[0]))
        bounds = gdf_grid.total_bounds
        print(f'[{city}] {len(gdf_grid)} cells, bounds={bounds.round(3).tolist()}',
              end=' ', flush=True)

        gdf_roads = _download_city_roads_adaptive(bounds)
        if gdf_roads is None:
            print(' -> SKIP (all tiles failed)')
            continue

        gdf_roads.to_file(cache_path, driver='GeoJSON')
        print(f' -> cached ({len(gdf_roads)} roads)', end='')

        # Verify and move to permanent osm dir
        if osm_dir:
            if _verify_geojson(cache_path):
                shutil.move(cache_path, osm_path)
                print(', verified, moved to osm/')
            else:
                print(', but verification FAILED')
                os.remove(cache_path)
        else:
            print()


def render_crhd_from_cache(
    input_dir: str,
    cache_dir: str,
    output_dir: str,
    manifest_path: Optional[str] = None,
    image_size: int = 512,
    dpi: int = 100,
    limit: Optional[int] = None,
    city_limit: Optional[int] = None,
    skip_existing: bool = False,
    context_size: int = 0,
):
    """Phase 2: render CRHD images from cached OSM road networks.

    Two rendering modes:

    1. **Cell-clip mode** (context_size=0, default):
       Clips roads to each grid cell's exact bounding box.  Output size
       is ``image_size``×``image_size``.

    2. **Context mode** (context_size > 0, e.g. 1000):
       For each grid cell centroid, renders a ``2*context_size`` ×
       ``2*context_size`` metre area (matching the original paper's
       ``dist=1000`` CRHDs).  Output size is **600×600** (figsize 6,6
       at 100 dpi) regardless of ``image_size``.

    When *skip_existing* is True, cells with an existing output file are
    not re-rendered but are still included in the manifest so it remains
    a complete index of the dataset.

    Args:
        input_dir: Directory containing classified grid shapefiles.
        cache_dir: Directory with GeoJSON cache files (from Phase 1).
        output_dir: Where to save CRHD PNG images.
        manifest_path: Path for manifest JSON.
        image_size: Output CRHD size in pixels (only used in cell-clip mode).
        dpi: Image DPI.
        limit: Max total CRHDs to render.
        city_limit: Max cities to process.
        skip_existing: Skip cells whose output already exists.
        context_size: Context radius in metres (0 = cell-clip mode).
    """
    import geopandas as gpd
    import json
    from shapely.geometry import box as shp_box

    os.makedirs(output_dir, exist_ok=True)

    if context_size > 0:
        # Original paper mode: 600×600, centroid-centred 2*context_size area
        render_figsize = (6, 6)
        render_dpi = 100
        print(f'Context mode: {context_size}m radius ({2*context_size}m × {2*context_size}m area, '
              f'{int(render_figsize[0]*render_dpi)}×{int(render_figsize[1]*render_dpi)} px)')
    else:
        render_figsize = (image_size / dpi, image_size / dpi)
        render_dpi = dpi

    city_dirs = sorted([
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ])

    manifest = []
    total_ok = 0
    total_fail = 0
    total_skipped = 0

    for city_idx, city in enumerate(city_dirs):

        shp_files = [
            f for f in os.listdir(os.path.join(input_dir, city))
            if f.endswith('.shp')
        ]
        if not shp_files:
            continue

        safe = _sanitize_city_name(city)
        cache_path = os.path.join(cache_dir, f'{safe}.geojson')
        if not os.path.exists(cache_path):
            print(f'[{city}] no cache, skip')
            continue

        gdf_grid = gpd.read_file(os.path.join(input_dir, city, shp_files[0]))
        gdf_roads = gpd.read_file(cache_path)
        print(f'[{city}] {len(gdf_grid)} cells, {len(gdf_roads)} cached roads',
              end=' ', flush=True)

        cell_ok = 0
        cell_fail = 0
        cell_skipped = 0
        for _, row in gdf_grid.iterrows():
            if limit and total_ok + total_fail >= limit:
                break

            cell_id = row.get('id', f'{city}_{total_ok}')
            img_name = f'{cell_id}.png'
            img_path = os.path.join(output_dir, img_name)
            rel_path = os.path.relpath(img_path)
            label = row.get('mid_cls', 'Unknown')

            centroid = row.geometry.centroid
            clat, clon = centroid.y, centroid.x

            if skip_existing and os.path.exists(img_path):
                manifest.append({
                    'source_path': str(cell_id),
                    'crhd_path': rel_path,
                    'city': city,
                    'lat': clat,
                    'lon': clon,
                    'label': label,
                    'status': 'success',
                })
                total_skipped += 1
                cell_skipped += 1
                continue

            try:
                if context_size > 0:
                    # Centroid-based context bbox (matching original paper)
                    deg_per_m_lat = 1.0 / 111320.0
                    deg_per_m_lon = 1.0 / (111320.0 * abs(math.cos(math.radians(clat))) + 1e-12)
                    half_lon = context_size * deg_per_m_lon
                    half_lat = context_size * deg_per_m_lat
                    ctx_bbox = shp_box(
                        clon - half_lon, clat - half_lat,
                        clon + half_lon, clat + half_lat,
                    )
                    clipped = _clip_gdf_to_bbox(gdf_roads, ctx_bbox)
                    _render_gdf(
                        clipped, img_path, render_figsize, render_dpi,
                        xlim=(clon - half_lon, clon + half_lon),
                        ylim=(clat - half_lat, clat + half_lat),
                    )
                else:
                    # Cell-clip mode (original default)
                    cell_bounds = row.geometry.bounds
                    clipped = _clip_gdf_to_bbox(gdf_roads, shp_box(*cell_bounds))
                    _render_gdf(
                        clipped, img_path, render_figsize, render_dpi,
                        xlim=(cell_bounds[0], cell_bounds[2]),
                        ylim=(cell_bounds[1], cell_bounds[3]),
                    )
                manifest.append({
                    'source_path': str(cell_id),
                    'crhd_path': rel_path,
                    'city': city,
                    'lat': clat,
                    'lon': clon,
                    'label': label,
                    'status': 'success',
                })
                total_ok += 1
                cell_ok += 1
            except Exception as e:
                print(f'  [warn] Cell {cell_id} failed: {e}')
                manifest.append({
                    'source_path': str(cell_id),
                    'crhd_path': None,
                    'city': city,
                    'lat': clat,
                    'lon': clon,
                    'label': label,
                    'status': 'failed',
                })
                total_fail += 1
                cell_fail += 1

        print(f'-> {cell_ok} OK, {cell_fail} failed'
              f'{", " + str(cell_skipped) + " skipped" if cell_skipped else ""}')
        if limit and total_ok + total_fail >= limit:
            break

    print(f'\nTotal rendered: {total_ok}')
    print(f'Total failed: {total_fail}')
    if total_skipped:
        print(f'Total skipped (already exist): {total_skipped}')

    if manifest_path:
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f'Manifest saved to: {manifest_path}')

    return manifest, total_ok, total_fail


def _count_cities(input_dir: str) -> int:
    """Count the number of city directories in the input directory."""
    return len([
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ])


def _count_completed_cities(osm_dir: str, city_names: list) -> int:
    """Count how many cities have a verified GeoJSON in osm_dir."""
    if not os.path.isdir(osm_dir):
        return 0
    existing = {os.path.splitext(f)[0] for f in os.listdir(osm_dir)
                if f.endswith('.geojson')}
    return sum(1 for c in city_names if _sanitize_city_name(c) in existing)


def main():
    """CLI entry point for batch CRHD operations.

    Pipeline mode (default) — downloads OSM, verifies, moves to ``osm/``,
    renders CRHDs, and loops until all cities are processed:

      python -m src.utils.crhd_generator \\
          --input ./data/original_grids/classified_grids_of_cities

    Download only:
      python -m src.utils.crhd_generator \\
          --input ./data/original_grids/classified_grids_of_cities \\
          --download-only

    Render only (from existing OSM GeoJSON):
      python -m src.utils.crhd_generator \\
          --input ./data/original_grids/classified_grids_of_cities \\
          --render-only
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate CRHD images from grid shapefiles')
    parser.add_argument('--download-only', action='store_true',
                        help='Download OSM roads only (no render)')
    parser.add_argument('--render-only', action='store_true',
                        help='Render CRHDs from existing OSM GeoJSON only')
    parser.add_argument('--input', type=str, required=True,
                        help='Directory of classified grid shapefiles')
    parser.add_argument('--cache', type=str, default='./data/osm_cache',
                        help='Temp dir for OSM downloads (default: ./data/osm_cache)')
    parser.add_argument('--osm-dir', type=str, default='./data/osm',
                        help='Permanent dir for verified OSM GeoJSON (default: ./data/osm)')
    parser.add_argument('--output', type=str, default='./data/crhd',
                        help='Output directory for CRHD images (default: ./data/crhd)')
    parser.add_argument('--manifest', type=str, default=None,
                        help='Path for manifest JSON (default: <output>/manifest.json)')
    parser.add_argument('--image-size', type=int, default=512,
                        help='CRHD image size in pixels (default: 512)')
    parser.add_argument('--dpi', type=int, default=100,
                        help='Image DPI (default: 100)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Max CRHDs to generate (default: no limit)')
    parser.add_argument('--city-limit', type=int, default=None,
                        help='Max cities to process (default: all)')
    parser.add_argument('--download-limit', type=int, default=None,
                        help='Max downloads per pipeline iteration (default: no limit)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip CRHD cells whose output already exists')
    parser.add_argument('--context-size', type=int, default=0,
                        help='Context radius in metres for centroid-centred CRHDs '
                             '(e.g. 1000 = 2km×2km area, 600×600 px). '
                             '0 = clip to cell boundary (default).')
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f'[error] Input directory not found: {args.input}', file=sys.stderr)
        sys.exit(1)

    manifest_path = args.manifest or os.path.join(args.output, 'manifest.json')

    # ── Download-only mode ──────────────────────────────────────────────
    if args.download_only:
        download_osm_cache(
            input_dir=args.input,
            cache_dir=args.cache,
            osm_dir=args.osm_dir,
            city_limit=args.city_limit,
        )
        return

    # ── Render-only mode ────────────────────────────────────────────────
    if args.render_only:
        _, ok, failed = render_crhd_from_cache(
            input_dir=args.input,
            cache_dir=args.osm_dir,
            output_dir=args.output,
            manifest_path=manifest_path,
            image_size=args.image_size,
            dpi=args.dpi,
            limit=args.limit,
            city_limit=args.city_limit,
            skip_existing=args.skip_existing,
            context_size=args.context_size,
        )
        print(f'\nDone. Rendered: {ok}, Failed: {failed}')
        return

    # ── Pipeline mode (default): monitor loop until all cities done ────
    total_cities = _count_cities(args.input)
    print(f'Pipeline started: {total_cities} cities total')
    print(f'  Cache temp:  {args.cache}')
    print(f'  OSM storage: {args.osm_dir}')
    print(f'  CRHD output: {args.output}')
    print()

    city_dirs = sorted([
        d for d in os.listdir(args.input)
        if os.path.isdir(os.path.join(args.input, d))
    ])
    city_names = [
        c for c in city_dirs
        if any(f.endswith('.shp') for f in os.listdir(os.path.join(args.input, c)))
    ]

    iteration = 0
    while True:
        iteration += 1
        completed = _count_completed_cities(args.osm_dir, city_names)
        pending = len(city_names) - completed

        print(f'--- Iteration {iteration}: {completed}/{len(city_names)} cities done, '
              f'{pending} pending ---')

        if pending == 0:
            print(f'\n{"="*60}')
            print(f'ALL {total_cities} CITIES COMPLETE!')
            print(f'{"="*60}')
            break

        # Phase 1: download a batch of pending cities
        dl_limit = args.download_limit if args.download_limit else pending
        print(f'  Downloading up to {dl_limit} cities ...')
        download_osm_cache(
            input_dir=args.input,
            cache_dir=args.cache,
            osm_dir=args.osm_dir,
            city_limit=dl_limit,
        )

        # Phase 2: render CRHDs for all completed cities
        print(f'  Rendering CRHDs from osm/ ...')
        _, ok, failed = render_crhd_from_cache(
            input_dir=args.input,
            cache_dir=args.osm_dir,
            output_dir=args.output,
            manifest_path=manifest_path,
            image_size=args.image_size,
            dpi=args.dpi,
            limit=args.limit,
            skip_existing=True,
            context_size=args.context_size,
        )

        if ok + failed > 0:
            print(f'  Rendered this pass: {ok} OK, {failed} failed')
        else:
            print(f'  (no new CRHDs to render)')

        # Brief pause before next iteration
        if pending > 0:
            print(f'  Sleeping 5s before next iteration ...\n')
            _time.sleep(5)

    print(f'\nFinal manifest: {manifest_path}')
    print('Pipeline finished.')


if __name__ == '__main__':
    main()

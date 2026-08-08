"""OSM road network downloader via Overpass API."""

from __future__ import annotations

import logging
import os
import shutil
import time

import geopandas as gpd
import osmnx as ox
import pandas as pd
import requests
import urllib3
from shapely.geometry import LineString

from .osm_graph import sanitize_city_name

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

OVERPASS_ENDPOINTS = ["https://overpass-api.de/api/interpreter"]

OVERPASS_HEADERS = {
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "User-Agent": "RoadWeaver/1.0",
}


def _overpass_roads_gdf(bbox, endpoint=None, timeout=60):
    """Query Overpass API and return road GeoDataFrame.

    Handles 429 (rate limit) with automatic backoff up to 5 retries.
    """
    if endpoint is None:
        endpoint = OVERPASS_ENDPOINTS[0]

    west, south, east, north = bbox
    query = (
        f"[out:json][timeout:{min(timeout, 60)}];"
        f'(way["highway"]({south},{west},{north},{east}););'
        f"out geom;"
    )

    for attempt in range(5):
        try:
            r = requests.post(
                endpoint, data=query, timeout=timeout, verify=False, headers=OVERPASS_HEADERS
            )
            if r.status_code == 429:
                wait = (attempt + 1) * 10
                print(f"  429 rate limit, waiting {wait}s ...", end=" ", flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            if attempt < 4:
                time.sleep(5)
                continue
            logger.warning("Overpass request failed: %s", e)
            return None
    else:
        return None

    elements = data.get("elements", [])
    rows = []
    for el in elements:
        if el.get("type") != "way":
            continue
        geom_coords = el.get("geometry")
        if not geom_coords:
            continue
        coords = [(p["lon"], p["lat"]) for p in geom_coords]
        if len(coords) < 2:
            continue
        highway = el.get("tags", {}).get("highway")
        if highway is None:
            continue
        rows.append({"geometry": LineString(coords), "highway": highway, "osm_id": el["id"]})

    if not rows:
        return gpd.GeoDataFrame(
            [], columns=["geometry", "highway", "osm_id"], geometry="geometry", crs="EPSG:4326"
        )

    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def _download_city_roads_adaptive(
    bounds, pad: float = 0.01, timeout: int = 60, max_splits: int = 4, tile_delay: float = 3.0
):
    """Download OSM roads with adaptive tiling.

    Tries the full bounding box first.  If the query fails, progressively
    splits into 4, 9, 16, ... tiles until all succeed or *max_splits*.
    """
    minx, miny, maxx, maxy = bounds
    minx -= pad
    miny -= pad
    maxx += pad
    maxy += pad

    for splits in range(1, max_splits + 1):
        tile_w = (maxx - minx) / splits
        tile_h = (maxy - miny) / splits

        tile_bboxes = []
        for r in range(splits):
            for c in range(splits):
                tile_bboxes.append(
                    (
                        minx + c * tile_w,
                        miny + r * tile_h,
                        minx + (c + 1) * tile_w,
                        miny + (r + 1) * tile_h,
                    )
                )

        print(f"  splitting {splits}x{splits} ({len(tile_bboxes)} tiles) ...", end=" ", flush=True)

        all_gdfs = []
        failed = 0
        for ti, tbbox in enumerate(tile_bboxes):
            if ti > 0:
                time.sleep(tile_delay)
            gdf = _overpass_roads_gdf(tbbox, timeout=timeout)
            if gdf is not None and not gdf.empty:
                all_gdfs.append(gdf)
            else:
                failed += 1

        if failed == 0:
            combined = pd.concat(all_gdfs, ignore_index=True)
            combined = combined.drop_duplicates(subset="geometry")
            print(f"{len(combined)} roads")
            return combined

        print(f"{failed}/{len(tile_bboxes)} tiles failed")
        if splits >= max_splits:
            return None

    return None


def _verify_geojson(filepath: str) -> bool:
    """Verify a GeoJSON file is valid and contains road data."""
    try:
        gdf = gpd.read_file(filepath)
        return len(gdf) > 0
    except Exception:
        return False


def download_osm_cache(
    input_dir: str,
    cache_dir: str,
    osm_dir: str | None = None,
    city_limit: int | None = None,
    city_names: list[str] | None = None,
) -> tuple[int, int, int]:
    """Download OSM road networks and cache as GeoJSON.

    If *city_names* is provided (e.g. from parquet), it is used directly
    instead of reading from *input_dir*.

    Returns:
        tuple[int, int, int]: (downloaded, skipped, failed) city counts.
    """
    os.makedirs(cache_dir, exist_ok=True)
    if osm_dir:
        os.makedirs(osm_dir, exist_ok=True)

    downloaded = skipped = failed = 0

    if city_names is not None:
        city_list = city_names
    else:
        city_list = sorted(
            [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
        )

    if osm_dir:
        present = {
            c
            for c in city_list
            if os.path.exists(os.path.join(osm_dir, f"{sanitize_city_name(c)}.geojson"))
        }
        skipped += len(present)
        city_list = [c for c in city_list if c not in present]
    if city_limit:
        city_list = city_list[:city_limit]

    for city in city_list:
        safe = sanitize_city_name(city)
        osm_path = os.path.join(osm_dir or "", f"{safe}.geojson")
        cache_path = os.path.join(cache_dir, f"{safe}.geojson")

        if osm_dir and os.path.exists(osm_path):
            skipped += 1
            continue

        if os.path.exists(cache_path):
            print(f"[{city}] cached, verifying ...", end=" ", flush=True)
            if _verify_geojson(cache_path):
                if osm_dir:
                    shutil.move(cache_path, osm_path)
                    print("moved to osm/")
                else:
                    print("ok")
                skipped += 1
            else:
                print("invalid, re-downloading")
                os.remove(cache_path)
                failed += 1
            continue

        if city_names is not None:
            print(f"[{city}] geocoding ...", end=" ", flush=True)
            try:
                g = ox.geocode(city)
                lat, lon = g[0], g[1] if len(g) == 2 else (g[1], g[0])
                bounds = (lon - 0.15, lat - 0.15, lon + 0.15, lat + 0.15)
            except Exception as e:
                print(f"geocode FAILED: {e}")
                failed += 1
                continue
            print(f"bounds={[round(b, 3) for b in bounds]}", end=" ", flush=True)
        else:
            shp_files = [f for f in os.listdir(os.path.join(input_dir, city)) if f.endswith(".shp")]
            if not shp_files:
                failed += 1
                continue
            gdf_grid = gpd.read_file(os.path.join(input_dir, city, shp_files[0]))
            bounds = gdf_grid.total_bounds
            print(
                f"[{city}] {len(gdf_grid)} cells, bounds={bounds.round(3).tolist()}",
                end=" ",
                flush=True,
            )

        gdf_roads = _download_city_roads_adaptive(bounds)
        if gdf_roads is None:
            print(" -> SKIP (all tiles failed)")
            failed += 1
            continue

        gdf_roads.to_file(cache_path, driver="GeoJSON")
        print(f" -> cached ({len(gdf_roads)} roads)", end="")

        if osm_dir:
            if _verify_geojson(cache_path):
                shutil.move(cache_path, osm_path)
                print(", verified, moved to osm/")
                downloaded += 1
            else:
                print(", but verification FAILED")
                os.remove(cache_path)
                failed += 1
        else:
            print()
            downloaded += 1

    return downloaded, skipped, failed

"""CRHD image rendering from OSM networks."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import osmnx as ox
from shapely.geometry import box
from tqdm import tqdm

from data_generator.osm_graph import sanitize_city_name

logger = logging.getLogger(__name__)

STREET_TYPES = [
    "service",
    "residential",
    "tertiary_link",
    "tertiary",
    "secondary_link",
    "primary_link",
    "motorway_link",
    "secondary",
    "trunk",
    "primary",
    "motorway",
]

STREET_WIDTHS = {
    "service": 1,
    "residential": 1,
    "tertiary_link": 1,
    "tertiary": 2,
    "secondary_link": 2,
    "primary_link": 2,
    "motorway_link": 2,
    "secondary": 3,
    "trunk": 4,
    "primary": 4,
    "motorway": 2.5,
}

STREET_COLORS = {
    "service": "skyblue",
    "residential": "skyblue",
    "tertiary_link": "skyblue",
    "tertiary": "cornflowerblue",
    "secondary_link": "cornflowerblue",
    "primary_link": "cornflowerblue",
    "trunk_link": "cornflowerblue",
    "motorway_link": "darkred",
    "secondary": "darkblue",
    "trunk": "black",
    "primary": "black",
    "motorway": "darkred",
}


# ── Core CRHD generation ──────────────────────────────────────────────


def _make_bbox_polygon(lon: float, lat: float, dist_m: float) -> box:
    """WGS84 bounding box polygon around a point (avoids osmnx projection)."""
    deg_per_m_lat = 1.0 / 111320.0
    deg_per_m_lon = 1.0 / (111320.0 * abs(math.cos(math.radians(lat))) + 1e-12)
    half_lon = dist_m * deg_per_m_lon
    half_lat = dist_m * deg_per_m_lat
    return box(lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat)


def graph_from_point(lon: float, lat: float, dist: int = 1000, network_type: str = "all"):
    """Download OSMnx graph around a point, with shapely 2.x compat."""
    poly = _make_bbox_polygon(lon, lat, dist)
    return ox.graph_from_polygon(poly, network_type=network_type)


def graph_to_crhd(
    graph, output_path: str, figsize: tuple[int, int] = (6, 6), dpi: int = 300, fmt: str = "png"
):
    """Render an OSMnx road graph as a CRHD image file."""
    gdf = ox.graph_to_gdfs(graph, nodes=False, edges=True)
    gdf.highway = gdf.highway.map(lambda x: x[0] if isinstance(x, list) else x)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    gdf.plot(ax=ax, linewidth=0.5, edgecolor="lightgreen")
    for stype in STREET_TYPES:
        if (gdf.highway == stype).any():
            gdf[gdf.highway == stype].plot(
                ax=ax, linewidth=STREET_WIDTHS[stype], edgecolor=STREET_COLORS[stype]
            )
    plt.axis("off")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0, format=fmt)
    plt.close()


def generate_centroid_crhd(
    lon: float,
    lat: float,
    output_path: str,
    dist: int = 1000,
    figsize: tuple[int, int] = (6, 6),
    dpi: int = 300,
    network_type: str = "all",
) -> bool:
    """Download OSM data around a point and save a CRHD image."""
    try:
        graph = graph_from_point(lon, lat, dist, network_type)
        graph_to_crhd(graph, output_path, figsize, dpi)
        return True
    except Exception as e:
        logger.warning("Failed for (%.4f, %.4f): %s", lat, lon, e)
        return False


# ── Rendering helpers ─────────────────────────────────────────────────


def _clip_gdf_to_bbox(gdf_edges, bbox_polygon):
    """Spatially clip an edge GeoDataFrame to a bounding box polygon."""
    sindex = gdf_edges.sindex
    possible = list(sindex.intersection(bbox_polygon.bounds))
    if not possible:
        return gdf_edges.iloc[:0]
    candidates = gdf_edges.iloc[possible]
    clipped = candidates[candidates.intersects(bbox_polygon)].copy()
    clipped.loc[:, "geometry"] = clipped.geometry.intersection(bbox_polygon)
    return clipped[~clipped.geometry.is_empty & ~clipped.geometry.isna()]


def _render_gdf(
    gdf,
    output_path: str,
    figsize: tuple[float, float] = (5.12, 5.12),
    dpi: int = 300,
    fmt: str = "png",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
):
    """Render a road GeoDataFrame as a CRHD image."""
    gdf.highway = gdf.highway.map(lambda x: x[0] if isinstance(x, list) else x)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if not gdf.empty:
        gdf.plot(ax=ax, linewidth=0.5, edgecolor="lightgreen")
        for stype in STREET_TYPES:
            if (gdf.highway == stype).any():
                gdf[gdf.highway == stype].plot(
                    ax=ax, linewidth=STREET_WIDTHS[stype], edgecolor=STREET_COLORS[stype]
                )
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    plt.axis("off")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0, format=fmt)
    plt.close()


# ── Batch CRHD rendering from cached OSM ──────────────────────────────


def render_crhd_from_cache(
    input_dir: str,
    cache_dir: str,
    output_dir: str,
    manifest_path: str | None = None,
    image_size: int = 512,
    dpi: int = 300,
    limit: int | None = None,
    city_limit: int | None = None,
    skip_existing: bool = False,
    context_size: int = 0,
):
    """Render CRHD images from cached OSM road networks.

    Two modes:
    - Cell-clip mode (context_size=0): clip roads to grid cell bounding box.
    - Context mode (context_size>0): centroid-centred 2*context_size area.
    """
    os.makedirs(output_dir, exist_ok=True)

    if context_size > 0:
        render_figsize = (6, 6)
        render_dpi = 300
    else:
        render_figsize = (image_size / dpi, image_size / dpi)
        render_dpi = dpi

    city_list = sorted(
        [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    )

    manifest = []
    total_ok = total_fail = total_skipped = 0

    for city in tqdm(city_list, desc="Cities", unit="city"):
        shp_files = [f for f in os.listdir(os.path.join(input_dir, city)) if f.endswith(".shp")]
        if not shp_files:
            continue

        safe = sanitize_city_name(city)
        cache_path = os.path.join(cache_dir, f"{safe}.geojson")
        if not os.path.exists(cache_path):
            print(f"[{city}] no cache, skip")
            continue

        gdf_grid = gpd.read_file(os.path.join(input_dir, city, shp_files[0]))
        gdf_roads = gpd.read_file(cache_path)

        cell_ok = cell_fail = cell_skipped = 0
        for _, row in tqdm(
            gdf_grid.iterrows(), desc=f"  {city}", total=len(gdf_grid), leave=False, unit="cell"
        ):
            if limit and total_ok + total_fail >= limit:
                break

            cell_id = row.get("id", f"{city}_{total_ok}")
            img_name = f"{cell_id}.png"
            img_path = os.path.join(output_dir, img_name)
            label = row.get("mid_cls", "Unknown")
            centroid = row.geometry.centroid
            clat, clon = centroid.y, centroid.x

            if skip_existing and os.path.exists(img_path):
                manifest.append(
                    {
                        "source_path": str(cell_id),
                        "crhd_path": os.path.relpath(img_path),
                        "city": city,
                        "lat": clat,
                        "lon": clon,
                        "label": label,
                        "status": "success",
                    }
                )
                total_skipped += 1
                cell_skipped += 1
                continue

            try:
                if context_size > 0:
                    deg_per_m_lat = 1.0 / 111320.0
                    deg_per_m_lon = 1.0 / (111320.0 * abs(math.cos(math.radians(clat))) + 1e-12)
                    half_lon = context_size * deg_per_m_lon
                    half_lat = context_size * deg_per_m_lat
                    ctx_bbox = box(
                        clon - half_lon, clat - half_lat, clon + half_lon, clat + half_lat
                    )
                    clipped = _clip_gdf_to_bbox(gdf_roads, ctx_bbox)
                    _render_gdf(
                        clipped,
                        img_path,
                        render_figsize,
                        render_dpi,
                        xlim=(clon - half_lon, clon + half_lon),
                        ylim=(clat - half_lat, clat + half_lat),
                    )
                else:
                    cell_bounds = row.geometry.bounds
                    clipped = _clip_gdf_to_bbox(gdf_roads, box(*cell_bounds))
                    _render_gdf(
                        clipped,
                        img_path,
                        render_figsize,
                        render_dpi,
                        xlim=(cell_bounds[0], cell_bounds[2]),
                        ylim=(cell_bounds[1], cell_bounds[3]),
                    )
                manifest.append(
                    {
                        "source_path": str(cell_id),
                        "crhd_path": os.path.relpath(img_path),
                        "city": city,
                        "lat": clat,
                        "lon": clon,
                        "label": label,
                        "status": "success",
                    }
                )
                total_ok += 1
                cell_ok += 1
            except Exception as e:
                print(f"  [warn] Cell {cell_id} failed: {e}")
                manifest.append(
                    {
                        "source_path": str(cell_id),
                        "crhd_path": None,
                        "city": city,
                        "lat": clat,
                        "lon": clon,
                        "label": label,
                        "status": "failed",
                    }
                )
                total_fail += 1
                cell_fail += 1

        print(
            f"-> {cell_ok} OK, {cell_fail} failed"
            + (f", {cell_skipped} skipped" if cell_skipped else "")
        )
        if limit and total_ok + total_fail >= limit:
            break

    print(
        f"\nTotal rendered: {total_ok}, failed: {total_fail}"
        + (f", skipped: {total_skipped}" if total_skipped else "")
    )

    if manifest_path:
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Manifest saved to: {manifest_path}")

    return manifest, total_ok, total_fail


def render_city_wide_crhd(
    osm_dir: str,
    output_dir: str,
    manifest_path: str | None = None,
    context_size: int = 7500,
    image_size: int = 1200,
    dpi: int = 300,
    skip_existing: bool = False,
    city_limit: int | None = None,
):
    """Render one CRHD per city, centred on the city's GeoJSON bounding box centre.

    Creates a 2*context_size × 2*context_size metre image for each city.
    """
    os.makedirs(output_dir, exist_ok=True)
    figsize = (image_size / dpi, image_size / dpi)

    city_names = sorted(
        [f.replace(".geojson", "") for f in os.listdir(osm_dir) if f.endswith(".geojson")]
    )
    if city_limit:
        city_names = city_names[:city_limit]

    manifest = []
    total_ok = total_fail = 0

    for city in tqdm(city_names, desc="Cities", unit="city"):
        safe = sanitize_city_name(city)
        img_path = os.path.join(output_dir, f"{safe}.png")

        if skip_existing and os.path.exists(img_path):
            manifest.append(
                {
                    "source_path": safe,
                    "city": city,
                    "crhd_path": os.path.relpath(img_path),
                    "status": "success",
                }
            )
            total_ok += 1
            continue

        geojson_path = os.path.join(osm_dir, f"{safe}.geojson")
        if not os.path.exists(geojson_path):
            continue

        gdf = gpd.read_file(geojson_path)
        if gdf.empty:
            continue

        bounds = gdf.total_bounds
        clon = (bounds[0] + bounds[2]) / 2.0
        clat = (bounds[1] + bounds[3]) / 2.0

        deg_per_m_lat = 1.0 / 111320.0
        deg_per_m_lon = 1.0 / (111320.0 * abs(math.cos(math.radians(clat))) + 1e-12)
        half_lon = context_size * deg_per_m_lon
        half_lat = context_size * deg_per_m_lat
        ctx_bbox = box(clon - half_lon, clat - half_lat, clon + half_lon, clat + half_lat)

        try:
            clipped = _clip_gdf_to_bbox(gdf, ctx_bbox)
            _render_gdf(
                clipped,
                img_path,
                figsize,
                dpi,
                xlim=(clon - half_lon, clon + half_lon),
                ylim=(clat - half_lat, clat + half_lat),
            )
            manifest.append(
                {
                    "source_path": safe,
                    "city": city,
                    "lat": clat,
                    "lon": clon,
                    "crhd_path": os.path.relpath(img_path),
                    "status": "success",
                }
            )
            total_ok += 1
        except Exception as e:
            print(f"  [warn] {city} failed: {e}")
            manifest.append(
                {
                    "source_path": safe,
                    "city": city,
                    "lat": clat,
                    "lon": clon,
                    "crhd_path": None,
                    "status": "failed",
                }
            )
            total_fail += 1

    print(f"\nTotal rendered: {total_ok}, failed: {total_fail}")

    if manifest_path:
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Manifest saved to: {manifest_path}")

    return manifest, total_ok, total_fail


def main():
    """CLI entry point for CRHD rendering."""
    parser = argparse.ArgumentParser(description="Render CRHD images from cached OSM GeoJSON")
    parser.add_argument(
        "--input", type=str, default=None, help="Grid shapefiles directory (for cell-based mode)"
    )
    parser.add_argument("--osm-dir", type=str, required=True, help="OSM GeoJSON directory")
    parser.add_argument("--output", type=str, required=True, help="CRHD output directory")
    parser.add_argument(
        "--manifest", type=str, default=None, help="Manifest path (default: <output>/manifest.json)"
    )
    parser.add_argument("--image-size", type=int, default=600)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--city-limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--context-size", type=int, default=1000, help="Context radius in metres")
    parser.add_argument(
        "--city-wide",
        action="store_true",
        help="Render one CRHD per city (centred on GeoJSON centroid) instead of grid cells",
    )
    args = parser.parse_args()

    manifest_path = args.manifest or os.path.join(args.output, "manifest.json")

    if args.city_wide:
        _, ok, failed = render_city_wide_crhd(
            osm_dir=args.osm_dir,
            output_dir=args.output,
            manifest_path=manifest_path,
            context_size=args.context_size,
            image_size=args.image_size,
            dpi=args.dpi,
            skip_existing=args.skip_existing,
            city_limit=args.city_limit,
        )
    else:
        if not args.input or not os.path.isdir(args.input):
            print("[error] Input directory required for cell-based mode", file=sys.stderr)
            sys.exit(1)
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

    print(f"\nDone. Rendered: {ok}, Failed: {failed}")


if __name__ == "__main__":
    main()

# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""OSM road network downloader CLI."""

from __future__ import annotations

import argparse

from data_generator.osm_downloader import download_osm_cache


def main(argv: list[str] | None = None) -> None:
    """Download OSM road networks via Overpass API and cache as GeoJSON."""
    parser = argparse.ArgumentParser(
        description="Download OSM road networks (Overpass API) and cache as GeoJSON"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input",
        type=str,
        help="Grid shapefiles directory (per-city .shp subdirs); city list and bbox from grid bounds",
    )
    source.add_argument(
        "--cities",
        type=str,
        help="Comma-separated city names (geocoded via osmnx, +/-0.15 deg bbox)",
    )
    parser.add_argument(
        "--cache", type=str, default="data/osm_cache", help="GeoJSON cache directory"
    )
    parser.add_argument(
        "--osm-dir", type=str, default="data/osm", help="Verified GeoJSON output directory"
    )
    parser.add_argument(
        "--city-limit", type=int, default=None, help="Only download the first N cities"
    )
    args = parser.parse_args(argv)

    city_names = [c.strip() for c in args.cities.split(",")] if args.cities else None
    downloaded, skipped, failed = download_osm_cache(
        input_dir=args.input,
        cache_dir=args.cache,
        osm_dir=args.osm_dir,
        city_limit=args.city_limit,
        city_names=city_names,
    )
    print(f"\nDone. Downloaded: {downloaded}, skipped: {skipped}, failed: {failed}.")


if __name__ == "__main__":
    main()

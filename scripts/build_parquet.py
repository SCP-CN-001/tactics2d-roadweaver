#!/usr/bin/env python3
"""Parquet dataset builder for urban priors."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from multiprocessing import Pool
from typing import Any

import pandas as pd

from data_generator.osm_graph import (
    build_city_name_map,
    clip_gdf_to_bbox,
    compute_bbox_latlon,
    load_city_geojson,
    patch_id_to_city,
)
from data_generator.prior_extractor import extract_all_priors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("data_generator")


def parse_args():
    """Parse command-line arguments for the builder."""
    parser = argparse.ArgumentParser(
        description="Build Urban Structural Prior Dataset (Parquet output)"
    )
    parser.add_argument("--crhd-root", default="data/crhd_2km")
    parser.add_argument("--graph-root", default="data/osm")
    parser.add_argument("--output", default="data/urban_prior/urban_prior.parquet")
    parser.add_argument("--context-size-m", type=float, default=2000.0)
    parser.add_argument("--image-size", type=int, default=600)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Worker processes (default: min(cpu_count, 16))",
    )
    parser.add_argument(
        "--include-tertiary",
        action="store_true",
        help="Include tertiary roads in skeleton graph (default: fallback only)",
    )
    parser.add_argument(
        "--road-types",
        type=str,
        default=None,
        help="Comma-separated road types for skeleton graph "
        '(e.g. "motorway,trunk,primary,secondary,tertiary"). '
        "Overrides --include-tertiary.",
    )
    return parser.parse_args()


def _flatten_record(rec: dict) -> dict[str, Any]:
    """Convert a nested sample dict into a flat row for Parquet."""
    gp = rec.get("global_prior", {}) or {}
    bp = rec.get("block_prior", {}) or {}
    sg = rec.get("urban_skeleton_graph", {}) or {}
    quality = rec.get("quality", {}) or {}
    source = rec.get("source", {}) or {}

    out = {"patch_id": rec.get("patch_id", "unknown")}

    for k in [
        "road_density_km_per_km2",
        "major_road_density_km_per_km2",
        "minor_road_density_km_per_km2",
        "node_count",
        "edge_count",
        "intersection_count",
        "major_node_count",
        "boundary_entry_count",
        "avg_degree",
        "dead_end_ratio",
        "three_way_ratio",
        "four_way_ratio",
        "road_length_mean_m",
        "road_length_std_m",
        "road_length_median_m",
        "orientation_entropy",
        "bearing_entropy",
        "gridness_score",
        "radialness_score",
        "organic_score",
        "estimated_lane_mean",
        "estimated_road_width_mean_m",
    ]:
        out[k] = gp.get(k) if gp else None

    for k in [
        "block_count",
        "block_area_mean_m2",
        "block_area_std_m2",
        "block_area_median_m2",
        "block_aspect_ratio_mean",
        "block_compactness_mean",
        "block_scale_m",
    ]:
        out[k] = bp.get(k) if bp else None
    out["block_prior_available"] = bool(bp.get("block_prior_available", False)) if bp else False

    skeleton_nodes = sg.get("nodes", []) if sg else []
    skeleton_edges = sg.get("edges", []) if sg else []
    out["skeleton_node_count"] = len(skeleton_nodes)
    out["skeleton_edge_count"] = len(skeleton_edges)
    out["skeleton_graph_json"] = json.dumps({"nodes": skeleton_nodes, "edges": skeleton_edges})

    out["quality_valid_graph"] = bool(quality.get("valid_graph", False)) if quality else False
    out["quality_error"] = quality.get("error") or ""

    out["crhd_image_path"] = source.get("crhd_image_path", "")
    out["graph_path"] = source.get("graph_path", "")
    return out


def _process_one_patch(
    patch_id,
    city_gdf,
    center_lat,
    center_lon,
    context_size_m,
    image_size,
    crhd_root,
    include_tertiary=False,
    road_types=None,
) -> dict:
    bbox = compute_bbox_latlon(center_lat, center_lon, context_size_m)
    try:
        clipped_gdf = clip_gdf_to_bbox(city_gdf, bbox)
    except Exception as e:
        return {
            "patch_id": patch_id,
            "quality": {"valid_graph": False, "error": f"Clip failed: {e}"},
        }

    if clipped_gdf.empty:
        return {
            "patch_id": patch_id,
            "quality": {"valid_graph": False, "error": "No roads in patch bounding box"},
        }

    priors = extract_all_priors(
        clipped_gdf,
        center_lat,
        center_lon,
        context_size_m,
        include_tertiary=include_tertiary,
        road_types=road_types,
    )

    graph_city_name = patch_id_to_city(patch_id)
    graph_rel = os.path.join("data", "osm", f"{graph_city_name.replace(' ', '_')}.geojson")
    crhd_rel = os.path.join(crhd_root, f"{patch_id}.png")

    return {
        "patch_id": patch_id,
        "source": {"crhd_image_path": crhd_rel, "graph_path": graph_rel},
        "global_prior": priors.get("global_prior", {}),
        "urban_skeleton_graph": priors.get("urban_skeleton_graph", {"nodes": [], "edges": []}),
        "block_prior": priors.get("block_prior", {}),
        "quality": {
            "matched": True,
            "valid_graph": priors["quality"]["valid_graph"],
            "error": priors["quality"]["error"],
        },
    }


def _worker_city(city_args: tuple) -> list:
    """Process all patches belonging to one city."""
    city_name, geojson_path, tasks, ctx_m, img_sz, crhd_root, incl_tert, road_types = city_args
    gdf = load_city_geojson(geojson_path)
    if gdf is None or gdf.empty:
        return []

    results = []
    for pid, lat, lon in tasks:
        sample = _process_one_patch(
            pid,
            gdf,
            lat,
            lon,
            ctx_m,
            img_sz,
            crhd_root,
            include_tertiary=incl_tert,
            road_types=road_types,
        )
        results.append(_flatten_record(sample))
    return results


def main():
    """Build the urban prior Parquet dataset."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Urban Structural Prior Dataset Builder")
    logger.info("=" * 60)

    project_root = os.getcwd()
    logger.info("Working directory: %s", project_root)

    # 1. Load manifest for patch list
    manifest_path = os.path.join(args.crhd_root, "manifest.json")
    if not os.path.isfile(manifest_path):
        logger.error("Manifest not found: %s", manifest_path)
        return

    with open(manifest_path) as f:
        manifest_list = json.load(f)
    logger.info("Loaded manifest with %d patches", len(manifest_list))

    if args.max_samples > 0:
        manifest_list = manifest_list[: args.max_samples]
        logger.info("Limited to %d samples", len(manifest_list))

    # 2. Build city name map for OSM GeoJSON files
    display_to_file, _ = build_city_name_map(args.graph_root)
    logger.info("Found %d OSM GeoJSON files", len(display_to_file))

    # 3. Group patches by city
    city_tasks = defaultdict(list)
    failed_count = 0
    for entry in manifest_list:
        patch_id = entry.get("source_path", "")
        city_name = entry.get("city")
        lat, lon = entry.get("lat"), entry.get("lon")
        if not patch_id or not city_name or lat is None or lon is None:
            failed_count += 1
            continue
        if city_name not in display_to_file:
            failed_count += 1
            continue

        city_tasks[city_name].append((patch_id, lat, lon))

    logger.info(
        "Assembled tasks for %d cities (%d patches, %d failed)",
        len(city_tasks),
        sum(len(v) for v in city_tasks.values()),
        failed_count,
    )

    crhd_root = args.crhd_root
    if not os.path.isabs(crhd_root):
        crhd_root = os.path.join(project_root, crhd_root)

    # Parse road-types argument
    road_types = None
    if args.road_types:
        road_types = {h.strip() for h in args.road_types.split(",")}

    # 4. Dispatch per-city work to pool
    pool_args = []
    for city_name, tasks in city_tasks.items():
        geojson_path = display_to_file[city_name]
        pool_args.append(
            (
                city_name,
                geojson_path,
                tasks,
                args.context_size_m,
                args.image_size,
                crhd_root,
                args.include_tertiary,
                road_types,
            )
        )

    num_workers = args.num_workers or min(os.cpu_count() or 4, 16)
    logger.info("Processing %d cities with %d workers ...", len(pool_args), num_workers)

    t_start = time.time()
    all_records = []

    with Pool(num_workers) as pool:
        for i, chunk in enumerate(pool.imap_unordered(_worker_city, pool_args)):
            all_records.extend(chunk)
            if (i + 1) % 10 == 0 or i == len(pool_args) - 1:
                elapsed = time.time() - t_start
                logger.info(
                    "Progress: %d/%d cities (%.1f%%) | %.1f cities/min | %d records",
                    i + 1,
                    len(pool_args),
                    100.0 * (i + 1) / len(pool_args),
                    (i + 1) / elapsed * 60,
                    len(all_records),
                )

    t_elapsed = time.time() - t_start
    logger.info(
        "Processing complete: %d records in %.1f seconds (%.1f samples/sec)",
        len(all_records),
        t_elapsed,
        len(all_records) / t_elapsed,
    )

    # 5. Write Parquet
    output_path = (
        args.output if os.path.isabs(args.output) else os.path.join(project_root, args.output)
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.DataFrame(all_records)
    df.to_parquet(output_path, index=False)
    logger.info(
        "Dataset saved: %s (%d rows, %d cols, %.1f MB)",
        output_path,
        len(df),
        len(df.columns),
        os.path.getsize(output_path) / (1024 * 1024),
    )

    # 6. Summary
    valid_count = df["quality_valid_graph"].sum()
    logger.info("=" * 60)
    logger.info("BUILD COMPLETE")
    logger.info("Total:     %d", len(manifest_list))
    logger.info("In Parquet: %d", len(all_records))
    logger.info("Valid:     %d", valid_count)
    logger.info("Failed:    %d", failed_count)


if __name__ == "__main__":
    main()

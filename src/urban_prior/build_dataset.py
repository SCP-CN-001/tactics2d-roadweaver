#!/usr/bin/env python3
"""Build the Urban Structural Prior Dataset as a flat Parquet table.

Pipeline:
1. Load style vectors from JSON.
2. Match each patch to its OSM road network via manifest.json centroids.
3. Clip roads to 2km bounding box.
4. Extract global priors, skeleton graph, and block priors.
5. Flatten into a single Parquet file with numeric columns + JSON string columns
   for complex data (skeleton graph).

Usage:
    python -m src.urban_prior.build_dataset \
        --style-json outputs/style_vectors.json \
        --crhd-root data/crhd_2km_context_600x600 \
        --output data/urban_prior/urban_prior.parquet \
        --context-size-m 2000 --image-size 600
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_urban_prior")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build Urban Structural Prior Dataset (Parquet output)"
    )
    parser.add_argument(
        "--style-json",
        default="outputs/style_vectors.json",
        help="Path to style_vectors.json",
    )
    parser.add_argument(
        "--crhd-root",
        default="data/crhd_2km_context_600x600",
        help="CRHD image directory",
    )
    parser.add_argument(
        "--graph-root",
        default="data/osm",
        help="Directory containing per-city GeoJSON road files",
    )
    parser.add_argument(
        "--output",
        default="data/urban_prior/urban_prior.parquet",
        help="Output Parquet path",
    )
    parser.add_argument("--context-size-m", type=float, default=2000.0)
    parser.add_argument("--image-size", type=int, default=600)
    parser.add_argument(
        "--max-samples", type=int, default=-1,
        help="Max samples to process (-1 = all)",
    )
    return parser.parse_args()


def load_style_vectors(path: str) -> List[Dict]:
    if not os.path.isfile(path):
        logger.error("Style vectors file not found: %s", path)
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)
    logger.info("Loaded %d style vector records", len(data))
    return data


def load_manifest(manifest_path: str) -> Dict[str, Dict]:
    if not os.path.isfile(manifest_path):
        logger.warning("Manifest not found: %s", manifest_path)
        return {}
    with open(manifest_path) as f:
        manifest_list = json.load(f)
    return {entry.get("source_path", ""): entry for entry in manifest_list}


def extract_patch_id(image_path: str) -> Optional[str]:
    basename = os.path.basename(image_path)
    root, _ = os.path.splitext(basename)
    return root


def build_city_graph_cache(
    graph_root: str,
    style_records: List[Dict],
    manifest_index: Dict[str, Dict],
) -> tuple:
    from src.urban_prior.graph_utils import load_city_geojson, build_city_name_map, patch_id_to_city

    display_to_file, _ = build_city_name_map(graph_root)
    logger.info("Found %d OSM GeoJSON files", len(display_to_file))

    needed_cities = set()
    patch_to_city = {}

    for rec in style_records:
        patch_id = extract_patch_id(rec.get("image_path", ""))
        if not patch_id:
            continue
        city_from_manifest = manifest_index.get(patch_id, {}).get("city")
        if city_from_manifest:
            city_name = city_from_manifest
        else:
            city_name = patch_id_to_city(patch_id)
        needed_cities.add(city_name)
        patch_to_city[patch_id] = city_name

    logger.info("Need data for %d unique cities", len(needed_cities))

    cache = {}
    for city_name in needed_cities:
        if city_name in display_to_file:
            gdf = load_city_geojson(display_to_file[city_name])
            if gdf is not None and not gdf.empty:
                cache[city_name] = gdf

    logger.info("City graph cache: %d loaded, %d missing",
                len(cache), len(needed_cities) - len(cache))
    return cache, patch_to_city


def _flatten_record(rec: Dict) -> Dict[str, Any]:
    """Convert a nested sample dict into a flat row for Parquet."""
    patch_id = rec.get("patch_id", "unknown")
    condition = rec.get("condition", {}) or {}
    gp = rec.get("global_prior", {}) or {}
    bp = rec.get("block_prior", {}) or {}
    sg = rec.get("urban_skeleton_graph", {}) or {}
    quality = rec.get("quality", {}) or {}
    source = rec.get("source", {}) or {}

    sv = condition.get("style_vector", [None] * 6) or [None] * 6
    while len(sv) < 6:
        sv.append(None)

    out = {"patch_id": patch_id}
    for i in range(6):
        out[f"style_vector_{i}"] = sv[i]
    out["top_pattern"] = condition.get("top_pattern")
    out["top_pattern_name"] = condition.get("top_pattern_name", "Unknown")
    out["confidence"] = condition.get("confidence")
    out["context_size_m"] = condition.get("context_size_m")
    out["meters_per_pixel"] = condition.get("meters_per_pixel")

    # Global prior fields
    for k in [
        "road_density_km_per_km2", "major_road_density_km_per_km2",
        "minor_road_density_km_per_km2", "node_count", "edge_count",
        "intersection_count", "major_node_count", "boundary_entry_count",
        "avg_degree", "dead_end_ratio", "three_way_ratio", "four_way_ratio",
        "road_length_mean_m", "road_length_std_m", "road_length_median_m",
        "orientation_entropy", "bearing_entropy", "gridness_score",
        "radialness_score", "organic_score", "estimated_lane_mean",
        "estimated_road_width_mean_m",
    ]:
        out[k] = gp.get(k) if gp else None

    # Block prior fields
    for k in [
        "block_count", "block_area_mean_m2", "block_area_std_m2",
        "block_area_median_m2", "block_aspect_ratio_mean",
        "block_compactness_mean", "block_scale_m",
    ]:
        out[k] = bp.get(k) if bp else None
    out["block_prior_available"] = bool(bp.get("block_prior_available", False)) if bp else False

    # Skeleton graph
    skeleton_nodes = sg.get("nodes", []) if sg else []
    skeleton_edges = sg.get("edges", []) if sg else []
    out["skeleton_node_count"] = len(skeleton_nodes)
    out["skeleton_edge_count"] = len(skeleton_edges)
    out["skeleton_graph_json"] = json.dumps({"nodes": skeleton_nodes, "edges": skeleton_edges})

    # Quality
    out["quality_matched"] = bool(quality.get("matched", False)) if quality else False
    out["quality_valid_graph"] = bool(quality.get("valid_graph", False)) if quality else False
    out["quality_error"] = quality.get("error") or ""

    # Source metadata
    out["crhd_image_path"] = source.get("crhd_image_path", "")
    out["graph_path"] = source.get("graph_path", "")

    return out


def process_one_patch(rec, city_gdf, center_lat, center_lon,
                      context_size_m, image_size, patch_id, crhd_root) -> Dict:
    from src.urban_prior.graph_utils import clip_gdf_to_bbox, compute_bbox_latlon, patch_id_to_city
    from src.urban_prior.extractor import extract_all_priors

    bbox = compute_bbox_latlon(center_lat, center_lon, context_size_m)
    try:
        clipped_gdf = clip_gdf_to_bbox(city_gdf, bbox)
    except Exception as e:
        return {"patch_id": patch_id,
                "quality": {"matched": True, "valid_graph": False, "error": f"Clip failed: {e}"}}

    if clipped_gdf.empty:
        return {"patch_id": patch_id,
                "quality": {"matched": True, "valid_graph": False,
                            "error": "No roads in patch bounding box"}}

    priors = extract_all_priors(clipped_gdf, center_lat, center_lon, context_size_m)

    style_vector = rec.get("style_vector", [])
    meters_per_pixel = context_size_m / image_size

    condition = {
        "style_vector": style_vector,
        "style_dim": rec.get("style_dim", len(style_vector)),
        "top_pattern": rec.get("top_pattern"),
        "top_pattern_name": rec.get("top_pattern_name"),
        "confidence": rec.get("confidence"),
        "context_size_m": context_size_m,
        "meters_per_pixel": meters_per_pixel,
    }

    graph_city_name = patch_id_to_city(patch_id)
    graph_relative_path = os.path.join("data", "osm", f"{graph_city_name.replace(' ', '_')}.geojson")
    crhd_relative = rec.get("image_path", os.path.join(crhd_root, f"{patch_id}.png"))

    return {
        "patch_id": patch_id,
        "source": {"crhd_image_path": crhd_relative, "graph_path": graph_relative_path,
                    "style_source": os.path.basename(rec.get("_style_source", ""))},
        "condition": condition,
        "global_prior": priors.get("global_prior", {}),
        "urban_skeleton_graph": priors.get("urban_skeleton_graph", {"nodes": [], "edges": []}),
        "block_prior": priors.get("block_prior", {}),
        "quality": {
            "matched": True,
            "valid_graph": priors["quality"]["valid_graph"],
            "error": priors["quality"]["error"],
        },
    }


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Urban Structural Prior Dataset Builder")
    logger.info("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    os.chdir(project_root)
    logger.info("Working directory: %s", project_root)

    # 1. Load style vectors
    style_path = args.style_json
    if not os.path.isabs(style_path):
        style_path = os.path.join(project_root, style_path)
    style_records = load_style_vectors(style_path)
    for rec in style_records:
        rec["_style_source"] = style_path

    if args.max_samples > 0:
        style_records = style_records[:args.max_samples]
        logger.info("Limited to %d samples", len(style_records))

    # 2. Load manifest
    manifest_path = os.path.join(args.crhd_root, "manifest.json")
    manifest_index = load_manifest(manifest_path)

    # 3. Build city graph cache
    graph_root = os.path.join(project_root, args.graph_root) if not os.path.isabs(args.graph_root) else args.graph_root
    city_cache, patch_to_city = build_city_graph_cache(graph_root, style_records, manifest_index)

    # 4. Process each patch
    output_path = args.output if os.path.isabs(args.output) else os.path.join(project_root, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    crhd_root = os.path.join(project_root, args.crhd_root) if not os.path.isabs(args.crhd_root) else args.crhd_root

    records = []
    failed_count = 0
    pattern_counter = Counter()
    total = len(style_records)
    t_start = time.time()

    for i, rec in enumerate(style_records):
        patch_id = extract_patch_id(rec.get("image_path", ""))
        if not patch_id:
            failed_count += 1
            continue

        city_name = manifest_index.get(patch_id, {}).get("city") or patch_to_city.get(patch_id)
        if not city_name:
            failed_count += 1
            continue

        cent = manifest_index.get(patch_id, {})
        center_lat, center_lon = cent.get("lat"), cent.get("lon")
        if center_lat is None or center_lon is None:
            failed_count += 1
            continue

        city_gdf = city_cache.get(city_name)
        if city_gdf is None:
            failed_count += 1
            continue

        sample = process_one_patch(rec, city_gdf, center_lat, center_lon,
                                    args.context_size_m, args.image_size, patch_id, crhd_root)
        records.append(_flatten_record(sample))

        top_name = rec.get("top_pattern_name", "Unknown")
        pattern_counter[top_name] += 1

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t_start
            logger.info("Progress: %d/%d (%.1f%%) | %.1f samples/sec",
                        i + 1, total, 100.0 * (i + 1) / total, (i + 1) / elapsed)

    t_elapsed = time.time() - t_start
    logger.info("Processing complete: %d records in %.1f seconds", len(records), t_elapsed)

    # 5. Write Parquet
    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)
    logger.info("Dataset saved to: %s (%d rows, %d columns)",
                output_path, len(df), len(df.columns))

    # 6. Summary log
    valid_count = df["quality_valid_graph"].sum()
    logger.info("=" * 60)
    logger.info("BUILD COMPLETE")
    logger.info("=" * 60)
    logger.info("Total style records: %d", total)
    logger.info("Records in Parquet:  %d", len(records))
    logger.info("Valid priors:        %d", valid_count)
    logger.info("Failed records:      %d", failed_count)
    logger.info("Pattern distribution:")
    for pattern, count in pattern_counter.most_common():
        logger.info("  %s: %d", pattern, count)
    logger.info("Parquet file: %s (%.1f MB)",
                output_path, os.path.getsize(output_path) / (1024 * 1024))


if __name__ == "__main__":
    main()

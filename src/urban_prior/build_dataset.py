#!/usr/bin/env python3
"""Build the Urban Structural Prior Dataset as a flat Parquet table.

Pipeline:
1. Load style vectors from JSON.
2. Match each patch to its OSM road network via manifest.json centroids.
3. Group patches by city, then process each city in parallel:
   clip roads → extract priors → flatten.
4. Write a single Parquet file with numeric columns + skeleton graph as JSON string.

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
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from typing import Any, Dict, List, Optional

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
    parser.add_argument("--style-json", default="outputs/style_vectors.json")
    parser.add_argument("--crhd-root", default="data/crhd_2km_context_600x600")
    parser.add_argument("--graph-root", default="data/osm")
    parser.add_argument("--output", default="data/urban_prior/urban_prior.parquet")
    parser.add_argument("--context-size-m", type=float, default=2000.0)
    parser.add_argument("--image-size", type=int, default=600)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Worker processes (default: min(cpu_count, 16))")
    parser.add_argument("--include-tertiary", action="store_true",
                        help="Include tertiary roads in skeleton graph (default: fallback only)")
    return parser.parse_args()


# ─── Module-level helpers (must be importable by worker processes) ────


def extract_patch_id(image_path: str) -> Optional[str]:
    root, _ = os.path.splitext(os.path.basename(image_path))
    return root


def _flatten_record(rec: Dict) -> Dict[str, Any]:
    """Convert a nested sample dict into a flat row for Parquet."""
    patch_id = rec.get("patch_id", "unknown")
    condition = rec.get("condition", {}) or {}
    gp = rec.get("global_prior", {}) or {}
    bp = rec.get("block_prior", {}) or {}
    sg = rec.get("urban_skeleton_graph", {}) or {}
    quality = rec.get("quality", {}) or {}
    source = rec.get("source", {}) or {}

    sv = (condition.get("style_vector") or [None] * 6)[:6]
    while len(sv) < 6:
        sv.append(None)

    out = {"patch_id": patch_id}
    for i in range(6):
        out[f"style_vector_{i}"] = sv[i]
    out["top_pattern"] = condition.get("top_pattern")
    out["top_pattern_name"] = condition.get("top_pattern_name", "Unknown")
    out["confidence"] = condition.get("confidence")

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

    for k in [
        "block_count", "block_area_mean_m2", "block_area_std_m2",
        "block_area_median_m2", "block_aspect_ratio_mean",
        "block_compactness_mean", "block_scale_m",
    ]:
        out[k] = bp.get(k) if bp else None
    out["block_prior_available"] = bool(bp.get("block_prior_available", False)) if bp else False

    skeleton_nodes = sg.get("nodes", []) if sg else []
    skeleton_edges = sg.get("edges", []) if sg else []
    out["skeleton_node_count"] = len(skeleton_nodes)
    out["skeleton_edge_count"] = len(skeleton_edges)
    out["skeleton_graph_json"] = json.dumps({"nodes": skeleton_nodes, "edges": skeleton_edges})

    out["quality_matched"] = bool(quality.get("matched", False)) if quality else False
    out["quality_valid_graph"] = bool(quality.get("valid_graph", False)) if quality else False
    out["quality_error"] = quality.get("error") or ""

    out["crhd_image_path"] = source.get("crhd_image_path", "")
    out["graph_path"] = source.get("graph_path", "")
    return out


def _process_one_patch(rec, city_gdf, center_lat, center_lon,
                       context_size_m, image_size, patch_id, crhd_root,
                       include_tertiary=False) -> Dict:
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

    priors = extract_all_priors(clipped_gdf, center_lat, center_lon, context_size_m,
                                include_tertiary=include_tertiary)

    condition = {
        "style_vector": rec.get("style_vector", []),
        "style_dim": rec.get("style_dim", 0),
        "top_pattern": rec.get("top_pattern"),
        "top_pattern_name": rec.get("top_pattern_name"),
        "confidence": rec.get("confidence"),
        "context_size_m": context_size_m,
        "meters_per_pixel": context_size_m / image_size,
    }

    graph_city_name = patch_id_to_city(patch_id)
    graph_rel = os.path.join("data", "osm", f"{graph_city_name.replace(' ', '_')}.geojson")
    crhd_rel = rec.get("image_path", os.path.join(crhd_root, f"{patch_id}.png"))

    return {
        "patch_id": patch_id,
        "source": {"crhd_image_path": crhd_rel, "graph_path": graph_rel},
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


# ─── Worker function: process all patches for one city ────────────────


def _worker_city(city_args: tuple) -> list:
    """Process all patches belonging to one city.

    city_args = (city_name, geojson_path, task_list, context_size_m, image_size, crhd_root, include_tertiary)
    task_list = [(rec, lat, lon, patch_id), ...]
    """
    city_name, geojson_path, tasks, ctx_m, img_sz, crhd_root, incl_tert = city_args

    # Each worker loads the GeoJSON once for its assigned city
    from src.urban_prior.graph_utils import load_city_geojson
    gdf = load_city_geojson(geojson_path)
    if gdf is None or gdf.empty:
        return []

    results = []
    for rec, lat, lon, pid in tasks:
        sample = _process_one_patch(rec, gdf, lat, lon, ctx_m, img_sz, pid, crhd_root,
                                     include_tertiary=incl_tert)
        results.append(_flatten_record(sample))
    return results


# ─── Main ─────────────────────────────────────────────────────────────


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
    with open(style_path) as f:
        style_records = json.load(f)
    logger.info("Loaded %d style vector records", len(style_records))

    if args.max_samples > 0:
        style_records = style_records[:args.max_samples]
        logger.info("Limited to %d samples", len(style_records))

    # 2. Load manifest
    manifest_path = os.path.join(args.crhd_root, "manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            manifest_list = json.load(f)
        manifest_index = {e.get("source_path", ""): e for e in manifest_list}
    else:
        manifest_index = {}

    # 3. Build per-city task lists
    from src.urban_prior.graph_utils import build_city_name_map, patch_id_to_city
    display_to_file, _ = build_city_name_map(args.graph_root)
    logger.info("Found %d OSM GeoJSON files", len(display_to_file))

    city_tasks = defaultdict(list)  # city_name → [(rec, lat, lon, patch_id)]
    failed_count = 0
    for rec in style_records:
        patch_id = extract_patch_id(rec.get("image_path", ""))
        if not patch_id:
            failed_count += 1
            continue

        city_name = manifest_index.get(patch_id, {}).get("city") or patch_id_to_city(patch_id)
        cent = manifest_index.get(patch_id, {})
        lat, lon = cent.get("lat"), cent.get("lon")
        if not city_name or lat is None or lon is None:
            failed_count += 1
            continue
        if city_name not in display_to_file:
            failed_count += 1
            continue

        city_tasks[city_name].append((rec, lat, lon, patch_id))

    logger.info("Assembled tasks for %d cities (%d patches, %d failed)",
                len(city_tasks), sum(len(v) for v in city_tasks.values()), failed_count)

    crhd_root = args.crhd_root
    if not os.path.isabs(crhd_root):
        crhd_root = os.path.join(project_root, crhd_root)

    # 4. Dispatch per-city work to pool
    pool_args = []
    for city_name, tasks in city_tasks.items():
        geojson_path = display_to_file[city_name]
        pool_args.append((city_name, geojson_path, tasks,
                          args.context_size_m, args.image_size, crhd_root,
                          args.include_tertiary))

    num_workers = args.num_workers or min(os.cpu_count() or 4, 16)
    logger.info("Processing %d cities with %d workers ...", len(pool_args), num_workers)

    t_start = time.time()
    all_records = []

    with Pool(num_workers) as pool:
        for i, chunk in enumerate(pool.imap_unordered(_worker_city, pool_args)):
            all_records.extend(chunk)
            if (i + 1) % 10 == 0 or i == len(pool_args) - 1:
                elapsed = time.time() - t_start
                logger.info("Progress: %d/%d cities (%.1f%%) | %.1f cities/min | %d records so far",
                            i + 1, len(pool_args), 100.0 * (i + 1) / len(pool_args),
                            (i + 1) / elapsed * 60, len(all_records))

    t_elapsed = time.time() - t_start
    logger.info("Processing complete: %d records in %.1f seconds (%.1f samples/sec)",
                len(all_records), t_elapsed, len(all_records) / t_elapsed)

    # 5. Write Parquet
    output_path = args.output if os.path.isabs(args.output) else os.path.join(project_root, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.DataFrame(all_records)
    df.to_parquet(output_path, index=False)
    logger.info("Dataset saved: %s (%d rows, %d cols, %.1f MB)",
                output_path, len(df), len(df.columns),
                os.path.getsize(output_path) / (1024 * 1024))

    # 6. Summary
    valid_count = df["quality_valid_graph"].sum()
    pattern_dist = df["top_pattern_name"].value_counts()
    logger.info("=" * 60)
    logger.info("BUILD COMPLETE")
    logger.info("=" * 60)
    logger.info("Total:     %d", len(style_records))
    logger.info("In Parquet: %d", len(all_records))
    logger.info("Valid:     %d", valid_count)
    logger.info("Failed:    %d", failed_count)
    logger.info("Patterns:")
    for p, c in pattern_dist.items():
        logger.info("  %s: %d", p, c)


if __name__ == "__main__":
    main()

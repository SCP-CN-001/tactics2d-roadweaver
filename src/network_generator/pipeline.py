"""Road graph generation pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
import yaml

from hdmap_generator import assign_lanes, graph_to_map
from network_generator.growth.config import GrowthConfig
from network_generator.growth.growth import grow
from network_generator.topology.graph_cleanup import (
    clean_parallel_roads,
    clean_sharp_angles,
    fix_edge_crossings,
    keep_lcc,
    prune_dead_ends,
    snap_endpoints,
)
from network_generator.topology.graph_connector import EndpointConnector
from network_generator.topology.graph_intersection import (
    classify_nodes,
    compress_to_intersection_graph,
    propagate_road_class,
)
from network_generator.topology.graph_merge import merge_close_nodes
from network_generator.topology.graph_refine import (
    align_geometries_to_nodes,
    clean_growth_parallels,
    fix_abnormal_edges,
    fix_growth_crossings,
    merge_compressed_graph,
    merge_near_miss_edges,
    merge_nearby_junctions,
    merge_parallel_edges,
    remove_redundant_chords,
    snap_edges_to_nodes,
)
from network_generator.topology.graph_simplify import simplify_chains
from utils.geometry import chaikin
from utils.geometry import segment_intersection as _segment_intersection


def generate_skeleton(
    gen: Any,
    condition: torch.Tensor,
    structural_priors: torch.Tensor,
    *,
    map_w: float = 2000.0,
    map_h: float = 2000.0,
    vq_map_size_m: float = 2000.0,
    min_spacing_m: float = 80.0,
    anchor_ratio: float = 0.08,
    temperature: float = 0.75,
    top_p: float = 0.65,
    seed: int | None = None,
    return_intermediates: bool = False,
) -> dict:
    """Generate and clean a skeleton graph from VQ + Transformer.

    Returns
    -------
    dict with keys ``coords``, ``edge_index``, ``road_field``, ``map_size_m``,
    ``gridness``, ``organic``, ``density``, ``condition``.
    """
    density = float(structural_priors[0]) if structural_priors.dim() > 0 else 20.0
    gridness = float(structural_priors[1]) if structural_priors.numel() > 1 else 0.5
    organic = float(structural_priors[3]) if structural_priors.numel() > 3 else 0.5
    map_max = max(map_w, map_h)

    with torch.no_grad():
        raw = gen.generate(
            condition.unsqueeze(0) if condition.dim() == 1 else condition,
            anchor_ratio=anchor_ratio,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
    field = raw["road_field"]
    coords_n = raw["coords"].copy()  # normalized [0, 1] in VQ space
    ei = raw["edge_index"].copy()
    intermediates = {"raw": (coords_n.copy(), ei.copy())}  # c1: post-VQ skeleton

    if len(coords_n) < 5:
        raise ValueError(f"Skeleton too small ({len(coords_n)} nodes)")

    # Endpoint connector (VQ native size)
    try:
        conn = EndpointConnector(map_size_m=vq_map_size_m).run(
            raw,
            field,
            max_connections=30,
            connect_remaining=True,
            max_remaining_m=600,
            simplify=False,
        )
        coords_n = conn["coords"]
        ei = conn["edge_index"]
    except Exception:
        pass

    # Simplify chains
    try:
        simp = simplify_chains(coords_n, ei, angle_threshold_deg=15, dp_epsilon_norm=0.002)
        coords_n, ei = simp[0], simp[1]
    except Exception:
        pass
    intermediates["simplified"] = (coords_n.copy(), ei.copy())  # c2: after chain simplify

    # Scale to target map size
    sx, sy = map_w / vq_map_size_m, map_h / vq_map_size_m
    coords_n = coords_n * np.array([sx, sy])

    # Spacing cleanup
    merge_dist = max(0.005, min_spacing_m / map_max * 0.25)
    merged = merge_close_nodes(
        coords_n,
        ei,
        np.zeros(len(coords_n), dtype=np.int64),
        merge_dist=merge_dist,
        map_size_m=map_max,
    )
    coords_n, ei = merged[0], merged[1]

    return {
        "coords": coords_n,
        "edge_index": ei,
        "road_field": field,
        "map_size_m": map_max,
        "density": density,
        "gridness": gridness,
        "organic": organic,
        "condition": condition,
        **({"intermediates": intermediates} if return_intermediates else {}),
    }


# ── Phase 2: Branch ────────────────────────────────────────────────


def generate_branch(
    coords: np.ndarray,
    edge_index: np.ndarray,
    road_field: np.ndarray,
    condition: torch.Tensor,
    *,
    map_w: float = 2000.0,
    map_h: float = 2000.0,
    density: float = 20.0,
    gridness: float = 0.5,
    organic: float = 0.5,
    local_spacing_m: float = 80.0,
    g1_branch_p: float = 0.04,
    grid_cuts: int = 15,
    organic_cuts: int = 20,
    prune_chain_m: float = 120.0,
    snap_dist_m: float = 50.0,
    angle_clean_deg: float = 15.0,
    angle_clean_compressed_deg: float = 20.0,
    parallel_angle_deg: float = 30.0,
    parallel_dist_m: float = 40.0,
    merge_junction_dist_m: float = 50.0,
    chord_dup_dist_m: float = 40.0,
    chord_detour_ratio: float = 0.10,
    chord_angle_deg: float = 20.0,
    return_intermediates: bool = False,
) -> dict:
    """Grow collector roads (G1) and local roads (G2), then clean up.

    Parameters
    ----------
    coords, edge_index :
        Skeleton graph from ``generate_skeleton`` (normalised [0, 1] in map space).
    road_field :
        VQ road field (for A* endpoint closure).
    condition :
        11-dim style + structural condition tensor.
    density, gridness, organic :
        Structural priors (read from ``structural_priors`` if available).

    Returns
    -------
    dict with keys ``coords_int``, ``edge_index_int``, ``node_types``,
    ``lanes_per_dir``, ``geometries``, ``road_class``.
    """
    map_max = max(map_w, map_h)

    # ── Growth ─────────────────────────────────────────────────────
    gc = GrowthConfig.from_condition(
        condition.cpu().numpy() if condition.dim() > 0 else np.zeros(11),
        local_spacing_m=local_spacing_m,
        map_size_m=map_max,
    )
    gc.map_width_m = map_w
    gc.map_height_m = map_h

    # Style-aware G2 overrides
    gc.apply_style_overrides(
        gridness=gridness,
        organic=organic,
        grid_cuts=grid_cuts,
        organic_cuts=organic_cuts,
        g1_branch_p=g1_branch_p,
    )

    grown = grow(
        coords * np.array([map_w, map_h]),
        edge_index,
        np.zeros(len(coords), dtype=np.int64),
        road_field,
        gc,
    )
    coords_norm = grown["coords"] / np.array([map_w, map_h])
    ei = grown["edge_index"].copy()
    road_class = grown.get("road_class", np.ones(len(ei), dtype=np.int64))
    intermediates = {"growth": (coords_norm.copy(), ei.copy())}  # c5: growth output

    # ── Merge close nodes in growth graph ─────────────────────────
    merge_dist_norm = max(0.003, 30.0 / map_max)
    merged_result = merge_close_nodes(
        coords_norm,
        ei,
        np.zeros(len(coords_norm), dtype=np.int64),
        merge_dist=merge_dist_norm,
        map_size_m=map_max,
    )
    coords_norm, ei = merged_result[0], merged_result[1]
    # Fix edge crossings in growth graph
    coords_norm, ei = fix_growth_crossings(coords_norm, ei, map_max)

    # ── Cleanup raw graph ──────────────────────────────────────────
    coords_norm, ei = prune_dead_ends(coords_norm, ei, prune_chain_m, map_max)
    coords_norm, ei = keep_lcc(coords_norm, ei)
    coords_norm, ei = clean_sharp_angles(coords_norm, ei, min_deg=angle_clean_deg)
    coords_norm, ei = snap_endpoints(coords_norm, ei, map_max, snap_dist_m=snap_dist_m)
    coords_norm, ei = keep_lcc(coords_norm, ei)
    # Clean parallel roads in growth graph before compression
    coords_norm, ei = clean_growth_parallels(
        coords_norm, ei, map_max, angle_deg=20.0, max_dist_m=25.0
    )
    coords_norm, ei = keep_lcc(coords_norm, ei)
    intermediates["cleanup"] = (coords_norm.copy(), ei.copy())  # c6: raw cleaned

    # ── Compress to intersection graph ─────────────────────────────
    # 1. Compress + initial merge
    c_int, ei_int, geoms = compress_to_intersection_graph(coords_norm, ei)
    c_int, ei_int, geoms = merge_compressed_graph(c_int, ei_int, geoms, map_max, merge_dist_m=60.0)
    fix_abnormal_edges(c_int, ei_int, geoms, map_max)
    c_int, ei_int, geoms = snap_edges_to_nodes(c_int, ei_int, geoms, map_max, snap_dist_m=15.0)
    nt = classify_nodes(c_int, ei_int, map_max, merge_dist_m=merge_junction_dist_m, compressed=True)

    # 2. Structural changes
    c_int, ei_int, nt, geoms = merge_nearby_junctions(
        c_int, ei_int, nt, geoms, map_max, merge_dist_m=merge_junction_dist_m
    )
    nt = classify_nodes(c_int, ei_int, map_max, merge_dist_m=merge_junction_dist_m, compressed=True)
    # NOTE: the tactics2d Intersection builder supports 3-6 arms directly, so
    # high-degree junctions (>=5) are no longer split into sub-junctions.
    c_int, ei_int, nt, geoms = merge_nearby_junctions(
        c_int, ei_int, nt, geoms, map_max, merge_dist_m=merge_junction_dist_m
    )
    nt = classify_nodes(c_int, ei_int, map_max, merge_dist_m=merge_junction_dist_m, compressed=True)
    c_int, ei_int, geoms = merge_parallel_edges(
        c_int, ei_int, geoms, map_max, angle_deg=parallel_angle_deg, dist_m=parallel_dist_m
    )

    # 3. Geometry alignment + smoothing
    geoms = align_geometries_to_nodes(c_int, ei_int, geoms)
    geoms = [chaikin(g, iterations=2) if len(g) >= 3 else g for g in geoms]
    geoms = align_geometries_to_nodes(c_int, ei_int, geoms)

    # 4. One-shot final cleanup (crossings -> merge -> parallels)
    c_int, ei_int, geoms = snap_edges_to_nodes(c_int, ei_int, geoms, map_max, snap_dist_m=8.0)
    # Crossing fix (shared implementation — one-pass + spatial index)
    c_int, ei_int, geoms = fix_edge_crossings(c_int, ei_int, geoms, map_max)
    # Near-miss edges (close but not crossing) get a shared intersection node.
    c_int, ei_int, geoms = merge_near_miss_edges(c_int, ei_int, geoms, map_max)
    c_int, ei_int, geoms = merge_parallel_edges(
        c_int, ei_int, geoms, map_max, angle_deg=parallel_angle_deg, dist_m=parallel_dist_m
    )
    # Chord (u,v) vs two-edge path u-w-v: shares endpoints with both path
    # edges, so it escapes all parallel cleanup — remove it explicitly.
    c_int, ei_int, geoms = remove_redundant_chords(
        c_int,
        ei_int,
        geoms,
        map_max,
        max_dist_m=chord_dup_dist_m,
        detour_ratio=chord_detour_ratio,
        angle_deg=chord_angle_deg,
    )
    c_int, ei_int, geoms = clean_parallel_roads(
        c_int, ei_int, geoms, map_max, angle_deg=parallel_angle_deg, max_dist_m=parallel_dist_m
    )

    # 5. Final classify
    nt = classify_nodes(c_int, ei_int, map_max, merge_dist_m=merge_junction_dist_m, compressed=True)

    # ---- Propagate road_class through compression -------------------------
    road_class = propagate_road_class(coords_norm, ei, road_class, ei_int)

    # ---- Lane assignment (via shared function) -----------------------
    lanes = assign_lanes(c_int, ei_int, geoms, road_class, density=density)

    return {
        "coords_int": c_int,
        "edge_index_int": ei_int,
        "node_types": nt,
        "lanes_per_dir": lanes,
        "geometries": geoms,
        "road_class": road_class,
        **({"intermediates": intermediates} if return_intermediates else {}),
    }


def load_pipeline_config(path: str | None = None) -> dict:
    """Load ``config/pipeline_config.yaml`` into a flat dict.

    Returns an empty dict if the file is not found.
    """
    if path is None:
        _here = os.path.dirname(__file__)
        path = os.path.join(_here, "..", "..", "config", "pipeline_config.yaml")
    try:
        with open(path) as _f:
            return yaml.safe_load(_f) or {}
    except FileNotFoundError:
        print(f"[Config] {path} not found — using code defaults")
        return {}


def generate_full(
    gen,
    condition: torch.Tensor,
    structural_priors: torch.Tensor,
    *,
    map_w: float = 2000.0,
    map_h: float = 2000.0,
    config: dict | None = None,
    **kwargs,
) -> dict:
    """Run both phases: skeleton → branch, return merged result.

    Parameters
    ----------
    config :
        Optional dict loaded via :func:`load_pipeline_config`.  Keys become
        keyword arguments to the inner phase functions.  Explicit ``**kwargs``
        take priority over ``config`` values.
    """
    if config is None:
        config = {}
    merged = {**config, **kwargs}

    skel = generate_skeleton(gen, condition, structural_priors, map_w=map_w, map_h=map_h, **merged)
    branch = generate_branch(
        skel["coords"],
        skel["edge_index"],
        skel["road_field"],
        skel["condition"],
        map_w=map_w,
        map_h=map_h,
        density=skel["density"],
        gridness=skel["gridness"],
        organic=skel["organic"],
        **{
            k: v
            for k, v in merged.items()
            if k not in ("anchor_ratio", "temperature", "top_p", "vq_map_size_m", "min_spacing_m")
        },
    )
    return {**skel, **branch}


def run_pipeline(
    gen,
    condition: torch.Tensor,
    structural_priors: torch.Tensor,
    *,
    map_w: float = 2000.0,
    map_h: float = 2000.0,
    name: str = "roadweaver",
    scenario_type: str = "urban",
    assemble_hdmap: bool = True,
    config: dict | None = None,
    **kwargs,
) -> dict:
    """Run the full RoadWeaver pipeline: skeleton → branch → HD map.

    Returns a dict with ``skeleton``, ``branch`` and (when ``assemble_hdmap``)
    ``hdmap`` keys, so scripts only keep their CLI/visualization code.  The
    flat ``{**skeleton, **branch}`` keys are also available via ``generate_full``.
    """
    merged = {**(config or {}), **kwargs}
    # ``map_w/map_h/name/scenario_type`` are consumed by ``run_pipeline`` itself.
    skel_kw = {
        k: v
        for k, v in merged.items()
        if k not in ("map_w", "map_h", "name", "scenario_type", "assemble_hdmap")
    }
    skel = generate_skeleton(gen, condition, structural_priors, map_w=map_w, map_h=map_h, **skel_kw)
    branch = generate_branch(
        skel["coords"],
        skel["edge_index"],
        skel["road_field"],
        skel["condition"],
        map_w=map_w,
        map_h=map_h,
        density=skel["density"],
        gridness=skel["gridness"],
        organic=skel["organic"],
        **{
            k: v
            for k, v in skel_kw.items()
            if k
            not in (
                "anchor_ratio",
                "temperature",
                "top_p",
                "vq_map_size_m",
                "min_spacing_m",
                "seed",
            )
        },
    )
    result = {"skeleton": skel, "branch": branch}
    if assemble_hdmap:
        result["hdmap"] = graph_to_map(
            branch["coords_int"],
            branch["edge_index_int"],
            branch["geometries"],
            node_types=branch["node_types"],
            lanes_per_dir=branch["lanes_per_dir"],
            road_class=branch["road_class"],
            map_w=map_w,
            map_h=map_h,
            name=name,
            scenario_type=scenario_type,
        )
    return result


def build_map(
    seed: int = 0,
    *,
    map_size_m: float = 2000.0,
    map_w: float | None = None,
    map_h: float | None = None,
    device: str | None = None,
    vq_ckpt: str | None = None,
    tfm_ckpt: str | None = None,
    cache_path: str | None = None,
):
    """Generate one RoadWeaver HD map from *seed* (thin wrapper over run_pipeline).

    Builds an 11-dim condition (style = one-hot over ``seed % 6`` patterns,
    structural priors at the 2km defaults), runs the full pipeline and returns
    the assembled tactics2d Map.  ``map_w``/``map_h`` override ``map_size_m``
    per-axis so non-square maps (width != height) are supported.  Mirrors the
    former ``render_tactics2d.build_map`` (now removed) so demos can swap it in
    directly.
    """
    from network_generator.backbone.generator import make_generator

    repo = Path(__file__).resolve().parent.parent.parent
    vq_ckpt = vq_ckpt or str(repo / "runtimes/vq_vae_2km/best.pth")
    tfm_ckpt = tfm_ckpt or str(repo / "runtimes/transformer_2km/best.pth")
    cache_path = cache_path or str(repo / "cache/masked_code_maps_2km/train.npz")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    gen = make_generator(vq_ckpt, tfm_ckpt, cache_path=cache_path, device=device)
    sv = torch.zeros(1, 6, device=device)
    sv[0, seed % 6] = 1.0
    sp = torch.tensor([[15.0, 0.5, 0.5, 0.5, 0.5]], dtype=torch.float, device=device)
    cond = torch.cat([sv, sp], dim=1)
    result = run_pipeline(
        gen,
        cond,
        cond[0, 6:],
        map_w=map_size_m if map_w is None else map_w,
        map_h=map_size_m if map_h is None else map_h,
        name=f"rw_{seed}",
        scenario_type="urban",
        seed=seed,
    )
    return result["hdmap"]

"""
GrowthConfig — scale-aware configuration for paper-style road growth.

All distance parameters are in **metres**.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class GrowthConfig:
    """Configuration for paper-style tensor-guided road growth."""

    # ── Map ───────────────────────────────────────────────────────────
    map_width_m: float = 2000.0
    map_height_m: float = 2000.0

    # ── Tensor field ──────────────────────────────────────────────────
    tensor_sample_step_m: float = 20.0
    tensor_radius_m: float = 220.0
    tensor_sigma_m: float = 90.0

    # ── G1: collector roads ───────────────────────────────────────────
    g1_seed_spacing_m: float = 180.0
    g1_step_m: float = 35.0
    g1_max_length_m: float = 450.0
    g1_max_steps: int = 18
    g1_branch_p: float = 0.03
    g1_branch_angle_deg: float = 45.0

    # ── Spatial constraints ───────────────────────────────────────────
    exclusion_radius_m: float = 22.0
    snap_radius_m: float = 70.0
    merge_radius_m: float = 45.0
    max_turn_deg: float = 38.0

    # ── Local cycle validation ────────────────────────────────────────
    min_cycle_area_m2: float = 2500.0
    max_cycle_area_m2: float = 140000.0
    min_cycle_compactness: float = 0.035
    max_new_edge_ratio: float = 0.35

    # ── Direction weights ─────────────────────────────────────────────
    w_tensor: float = 0.65
    w_inertia: float = 0.25
    w_demand: float = 0.10

    # ── G2: face infill ───────────────────────────────────────────────
    g2_min_face_area_m2: float = 12000.0
    g2_target_face_area_m2: float = 8000.0
    g2_max_cuts_per_pass: int = 4
    g2_min_cut_length_m: float = 50.0

    # ── L-Level defaults ──────────────────────────────────────────────
    arterial_lanes_per_dir: int = 4
    collector_lanes_per_dir: int = 2
    local_lanes_per_dir: int = 1

    # ── Routing ───────────────────────────────────────────────────────
    random_seed: int = 7

    @classmethod
    def from_condition(
        cls,
        condition: np.ndarray,
        local_spacing_m: float = 200.0,
        map_size_m: float = 2000.0,
        **overrides,
    ) -> "GrowthConfig":
        """Build from 11-dim condition + H-Graph local spacing."""
        cond = np.asarray(condition).ravel()
        density = float(cond[6]) if len(cond) > 6 else 20.0
        gridness = float(cond[7]) if len(cond) > 7 else 0.5
        radialness = float(cond[8]) if len(cond) > 8 else 0.0

        s = max(local_spacing_m, 30.0)
        block = max(s / 4.0, 30.0)
        density_factor = max(density / 20.0, 0.5)

        # Absolute metre values (not block-relative)
        # These worked well in the individual-front version
        seed_sp = map_size_m * 0.09       # ~180m between seeds
        step = map_size_m * 0.02          # ~40m per step
        max_len = map_size_m * 0.40       # ~800m max length
        max_steps = 18
        snap_r = map_size_m * 0.04        # ~80m snap radius
        turn_deg = 30.0
        branch_p = 0.0                    # no branching for now

        if gridness > 0.7:
            w_t, w_i, w_d = 0.75, 0.25, 0.0
            turn_deg = 25.0
        elif radialness > 0.6:
            w_t, w_i, w_d = 0.65, 0.35, 0.0
            turn_deg = 30.0
        else:
            w_t, w_i, w_d = 0.60, 0.40, 0.0
            turn_deg = 35.0

        snap_r = max(snap_r, 50.0)  # min 50m snap

        cfg = cls(
            map_width_m=map_size_m,
            map_height_m=map_size_m,
            g1_seed_spacing_m=float(seed_sp),
            g1_step_m=float(step),
            g1_max_length_m=float(max_len),
            g1_max_steps=int(max_steps),
            g1_branch_p=float(branch_p),
            snap_radius_m=float(snap_r),
            max_turn_deg=float(turn_deg),
            w_tensor=float(w_t),
            w_inertia=float(w_i),
            w_demand=float(w_d),
        )
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

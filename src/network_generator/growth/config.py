"""Scale-aware growth configuration parameters."""

from __future__ import annotations

from dataclasses import dataclass

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
    tensor_smooth_radius_m: float = 0.0  # tangent smoothing (grid→large, organic→0)

    # ── G1: collector roads ───────────────────────────────────────────
    g1_seed_spacing_m: float = 180.0
    g1_step_m: float = 35.0
    g1_max_length_m: float = 450.0
    g1_max_steps: int = 18
    g1_branch_p: float = 0.03
    g1_branch_angle_deg: float = 45.0
    g1_seed_jitter: float = 0.15  # direction randomness per seed

    # ── Spatial constraints ───────────────────────────────────────────
    exclusion_radius_m: float = 22.0
    snap_radius_m: float = 70.0
    snap_radius_scale: float = 1.0  # style multiplier: <1 = early snap, >1 = late snap
    merge_radius_m: float = 45.0
    max_turn_deg: float = 38.0

    # ── Style parameters ──────────────────────────────────────────────
    per_step_jitter_deg: float = 2.0  # direction noise per growth step
    g2_jitter_deg: float = 0.0  # face cut direction jitter

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
    ) -> GrowthConfig:
        """Build from 11-dim condition: density, gridness, organic, bearing_entropy."""
        cond = np.asarray(condition).ravel()
        density = float(cond[6]) if len(cond) > 6 else 20.0
        gridness = float(cond[7]) if len(cond) > 7 else 0.5
        radialness = float(cond[8]) if len(cond) > 8 else 0.0
        organic = float(cond[9]) if len(cond) > 9 else 0.5
        bearing_entropy = float(cond[10]) if len(cond) > 10 else 0.8

        # ── Density-driven scaling ──────────────────────────────────────
        # Dense cities (→ 50 km/km²): more seeds, shorter steps, longer reach
        # Sparse cities (→ 5 km/km²): fewer seeds, longer steps, shorter reach
        df = density / 20.0  # 1.0 at 20 km/km²
        df = max(0.3, min(df, 4.0))  # clamp (3.0→4.0: density scales to ~80)

        seed_sp = map_size_m * 0.09 / df  # no 0.04 floor: seeds keep scaling with density
        step = map_size_m * max(0.008, 0.02 / (df**0.5))
        max_len = map_size_m * min(0.6, 0.4 * (df**0.5))
        max_steps = max(8, min(30, int(18 * df**0.3)))
        # Budget guard: without it, step shrinks faster than max_steps grows at
        # high density, so per-front reachable length falls with density instead
        # of rising.  Let fronts actually reach max_len.
        max_steps = max(max_steps, int(np.ceil(max_len / step)))
        # G2: dense → more cuts
        g2_cuts = max(1, int(4 * df))

        # ── Gridness / Organic → direction weights, turning, jitter, snap ──
        _STYLE_PRESETS = [
            (
                gridness > 0.7,
                {
                    "w_t": 0.85,
                    "w_i": 0.15,
                    "turn_deg": 18.0,
                    "jitter": max(0.02, 0.08 * (1.0 - gridness)),
                    "step_jitter": 1.0,
                    "snap_scale": 0.65,
                    "g2_jitter": 0.0,
                },
            ),
            (
                organic > 0.7,
                {
                    "w_t": 0.35,
                    "w_i": 0.65,
                    "turn_deg": 50.0,
                    "jitter": 0.25,
                    "step_jitter": 8.0,
                    "snap_scale": 1.5,
                    "g2_jitter": 30.0,
                },
            ),
            (
                radialness > 0.6,
                {
                    "w_t": 0.65,
                    "w_i": 0.35,
                    "turn_deg": 30.0,
                    "jitter": 0.12,
                    "step_jitter": 3.0,
                    "snap_scale": 0.9,
                    "g2_jitter": 5.0,
                },
            ),
        ]
        p = {
            "w_t": 0.55,
            "w_i": 0.45,
            "turn_deg": 35.0,
            "jitter": 0.15,
            "step_jitter": 4.0,
            "snap_scale": 1.0,
            "g2_jitter": 10.0,
        }
        for cond, vals in _STYLE_PRESETS:
            if cond:
                p = vals
                break
        w_t, w_i = p["w_t"], p["w_i"]
        turn_deg = p["turn_deg"]
        jitter = p["jitter"]
        step_jitter = p["step_jitter"]
        snap_scale = p["snap_scale"]
        g2_jitter = p["g2_jitter"]

        # ── Bearing entropy → seed jitter ──────────────────────────────
        be_factor = 1.0 - bearing_entropy * 0.5
        jitter *= be_factor
        step_jitter *= 0.5 + bearing_entropy * 0.5

        # Snap radius decoupled from density: a linear-in-df snap_r grows
        # past the (shrinking) seed spacing, so dense cities' fronts snap to a
        # neighbour after ~1 step and G1 growth collapses.  Tight baseline
        # (0.02 * map_size) lets fronts weave past each other before snapping,
        # lifting end-to-end road density; style scale still differentiates
        # grid/organic/radial.
        base_snap = map_size_m * 0.02
        snap_r = max(map_size_m * 0.012, base_snap * snap_scale)

        # ── Tensor field params: style-aware ───────────────────────────
        # Grid → large radius + tangent smoothing = uniform field
        # Organic → small radius + no smoothing = local variation
        tf_radius = max(120.0, map_size_m * 0.044 * (1.5 - 0.5 * gridness))
        tf_sigma = tf_radius * 0.4
        tf_smooth = 0.0
        if gridness > 0.6:
            tf_smooth = tf_radius * 0.6  # strong smoothing → uniform directions
        elif gridness > 0.4:
            tf_smooth = tf_radius * 0.3
        # Organic: no smoothing, keep local variation

        cfg = cls(
            map_width_m=map_size_m,
            map_height_m=map_size_m,
            tensor_radius_m=float(tf_radius),
            tensor_sigma_m=float(tf_sigma),
            tensor_smooth_radius_m=float(tf_smooth),
            g1_seed_spacing_m=float(seed_sp),
            g1_step_m=float(step),
            g1_max_length_m=float(max_len),
            g1_max_steps=int(max_steps),
            g1_branch_p=0.0,
            g1_seed_jitter=float(jitter),
            snap_radius_m=float(snap_r),
            snap_radius_scale=float(snap_scale),
            max_turn_deg=float(turn_deg),
            per_step_jitter_deg=float(step_jitter),
            g2_jitter_deg=float(g2_jitter),
            w_tensor=float(w_t),
            w_inertia=float(w_i),
            w_demand=0.0,
            g2_max_cuts_per_pass=int(g2_cuts),
        )
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    def apply_style_overrides(
        self,
        *,
        gridness: float = 0.5,
        organic: float = 0.5,
        grid_cuts: int = 15,
        organic_cuts: int = 20,
        g1_branch_p: float = 0.04,
    ) -> None:
        """Apply gridness/organic-aware G2 and branch overrides in place."""
        if gridness > 0.6:
            self.g2_max_cuts_per_pass = max(self.g2_max_cuts_per_pass, grid_cuts)
            self.g2_jitter_deg = 5.0
            self.g1_seed_jitter = 0.15
            self.per_step_jitter_deg = 2.0
        elif organic > 0.6:
            self.g2_max_cuts_per_pass = max(self.g2_max_cuts_per_pass, organic_cuts)
            self.g2_jitter_deg = 35.0
            self.g1_seed_jitter = 0.35
            self.per_step_jitter_deg = 8.0
        self.g1_branch_p = g1_branch_p

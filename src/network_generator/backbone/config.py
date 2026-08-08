"""Backbone configuration hyper-parameters."""

from __future__ import annotations

from dataclasses import dataclass

from utils.patterns import PATTERN_NAMES  # noqa: F401  (re-exported for callers)


@dataclass
class BackboneConfig:
    # ── Raster field ────────────────────────────────────────────────
    # Raster field resolution in pixels.
    # 128 for 2 km maps, 256 for 5 km maps.
    resolution: int = 256

    # VQ code map spatial size.
    # 32 for 4x downsampling (256 → 32), 64 for 2x downsampling.
    code_map_size: int = 32

    # ── Condition ───────────────────────────────────────────────────
    # Dimension of the style vector (output of the style encoder).
    style_dim: int = 6

    # Dimension of structural priors:
    # road_density, gridness, radialness, organic, bearing_entropy.
    extra_condition_dim: int = 5

    # Map size used to normalise coordinates before feeding into the
    # condition encoder.  Value in metres.
    map_size_scale: float = 5000.0

    # ── Data paths ─────────────────────────────────────────────────
    # Path to the training / validation split Parquet files.
    train_split_path: str = "data/urban_prior/2km/splits/train.parquet"
    val_split_path: str = "data/urban_prior/2km/splits/val.parquet"

    # Maximum number of skeleton nodes per sample during training.
    # Samples exceeding this limit are filtered out.
    max_nodes_in_training: int = 1000

    # BFS neighbour ordering strategy for skeleton graph traversal.
    # Options: 'bearing' | 'distance' | 'index'.
    bfs_neighbor_sort_by: str = "bearing"


# Singleton.
CONFIG = BackboneConfig()

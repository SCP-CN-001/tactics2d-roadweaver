"""
Road growth — pair-based block growth + G2 raster infill.

Components:
  config:        GrowthConfig — pattern- & scale-aware parameters (in metres).
  tensor_field:  GraphTensorField — local direction field from H-Graph tangents.
  growth:        G1 pair-based cooperative growth + G2 face infill.
  attribute:     Per-edge attribute assignment (lanes, speed, direction).
"""
from .config import GrowthConfig
from .tensor_field import GraphTensorField
from .growth import grow
from .growth import assign_attributes

__all__ = [
    "GrowthConfig",
    "GraphTensorField",
    "grow",
    "assign_attributes",
]

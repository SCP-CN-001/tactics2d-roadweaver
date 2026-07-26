"""
Road growth — G1 collector roads + A* closure + G2 face infill.

Components:
  config:        GrowthConfig — pattern- & scale-aware parameters (in metres).
  tensor_field:  GraphTensorField — local direction field from road tangents.
  growth:        G1 pair-based cooperative growth + G2 face infill.
"""

from .config import GrowthConfig
from .growth import assign_attributes, grow
from .tensor_field import GraphTensorField

__all__ = ["GrowthConfig", "GraphTensorField", "grow", "assign_attributes"]

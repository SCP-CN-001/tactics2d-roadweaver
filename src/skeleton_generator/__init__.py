"""
Road Skeleton Graph Generator.

Mainline:
  condition + AnchorSampler
    → MaskedCodeModel (Transformer)
    → VQVAE decoder
    → field_to_graph → skeleton graph

Components:
    - VQVAE:                      road field ↔ discrete code map (512-1024 codes)
    - MaskedCodeModel:            conditional masked code completion
    - AnchorSampler:              anchor token retrieval from similar conditions
    - AnchorGenerator:            full anchor-based conditional generation pipeline
    - skeleton_dataset:           GT skeleton graphs → raster fields (6ch)
    - graph_to_field:             skeleton graph → raster field
    - field_to_graph:             binary road field → skeleton graph
    - field_to_graph:             binary road field → skeleton graph (incl. cleanup)
    - field_refiner:              ResUNet to repair VQ-decoder artifacts
    - losses:                     FieldLoss (BCE + Dice + Focal)
"""

from .config import CONFIG, PATTERN_NAMES
from .bfs_ordering import BFSOrdering
from .vq_vae import VQVAE, VectorQuantizer
from .masked_transformer import MaskedCodeModel
from .anchor_sampler import AnchorGenerator, AnchorSampler

__all__ = [
    "CONFIG", "PATTERN_NAMES",
    "BFSOrdering",
    "VQVAE", "VectorQuantizer",
    "MaskedCodeModel", "AnchorGenerator", "AnchorSampler",
]

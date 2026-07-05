"""
Road Skeleton Graph Generator.

Mainline:
  condition + AnchorSampler
    → MaskedCodeModel (Transformer)
    → VQVAE decoder
    → field_to_graph → skeleton graph

Components:
    - ConditionEncoder:           style + priors → conditioning embedding
    - VQVAE:                      road field ↔ discrete 32×32 code map
    - MaskedCodeModel:            conditional masked code completion
    - AnchorGenerator:            anchor-based conditional generation pipeline
    - skeleton_dataset:           GT skeleton graphs → raster fields
    - graph_to_field:             skeleton graph → raster field
    - field_to_graph:             binary road field → skeleton graph
    - graph_cleanup:              morphological cleanup + graph pruning
    - graph_metrics:              roundtrip evaluation
    - losses:                     FieldLoss (BCE + Dice + Focal)
"""

from .config import CONFIG, PATTERN_NAMES
from .condition_encoder import ConditionEncoder
from .bfs_ordering import BFSOrdering
from .vq_vae import VQVAE, VectorQuantizer
from .masked_transformer import MaskedCodeModel
from .anchor_sampler import AnchorGenerator, AnchorSampler

__all__ = [
    "CONFIG", "PATTERN_NAMES",
    "ConditionEncoder", "BFSOrdering",
    "VQVAE", "VectorQuantizer",
    "MaskedCodeModel", "AnchorGenerator", "AnchorSampler",
]

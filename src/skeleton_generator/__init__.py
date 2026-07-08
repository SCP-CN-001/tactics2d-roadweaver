"""
Road Skeleton Graph Generator.

Mainline:
  condition + AnchorSampler
    → MaskedCodeModel (Transformer)
    → VQVAE decoder
    → field_to_graph → skeleton graph

Components:
    - VQVAE:                      road field ↔ discrete 32×32 code map (512 codes)
    - MaskedCodeModel:            conditional masked code completion (vocab=513)
    - AnchorSampler:              anchor token retrieval from similar conditions
    - AnchorGenerator:            full anchor-based conditional generation pipeline
    - skeleton_dataset:           GT skeleton graphs → raster fields (6ch, 128×128)
    - graph_to_field:             skeleton graph → raster field
    - field_to_graph:             binary road field → skeleton graph
    - graph_cleanup:              morphological cleanup + graph pruning
    - graph_metrics:              roundtrip evaluation
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

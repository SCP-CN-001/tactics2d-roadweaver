"""
Spatial Conditional Skeleton Generator — Configuration.

Centralised hyper-parameters.  A single CONFIG singleton mirrors the style
of the sibling skeleton_generator module but is completely independent.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class GeneratorConfig:
    # ------ Architecture ------
    condition_dim: int = 256
    """Output dim of the condition encoder MLP."""

    hidden_dim: int = 256
    """GRU hidden state dimension."""

    node_type_emb_dim: int = 32
    """Embedding dim for node type input."""

    coord_emb_dim: int = 64
    """Embedding dim for coordinate input."""

    edge_summary_dim: int = 64
    """Hidden dim for edge summary MLP that pools previous connections."""

    step_emb_dim: int = 32
    """Embedding dim for step position encoding."""

    max_prev_window: int = 32
    """Number of candidate previous nodes to consider for edge prediction."""

    num_node_types: int = 5
    """0=Waypoint(deg2), 1=Junction(deg3), 2=Roundabout, 3=Cross(deg≥4), 4=Terminal(deg1)."""

    coord_mode: str = "absolute"
    """'absolute' or 'delta'.  Absolute predicts coords in [0,1]; delta predicts offsets."""

    max_nodes: int = 256
    """Hard upper bound on skeleton node count."""

    max_edges: int = 512
    """Soft upper bound on skeleton edge count."""

    map_size_scale: float = 2000.0
    """Used to normalise map_size before feeding into condition encoder."""

    style_dim: int = 6
    """Dimension of the style vector."""

    extra_condition_dim: int = 5
    """Structural prior metrics: road_density, gridness, radialness, organic, bearing_entropy."""

    # ------ Edge predictor (v3: Pointer) ------
    edge_hidden_dim: int = 256
    """Hidden dim for the per-pair edge MLP."""


    # ------ BFS ordering ------
    bfs_neighbor_sort_by: str = "bearing"
    """How to sort BFS neighbours: 'bearing' | 'distance' | 'index'."""

    # ------ Loss weights (v3) ------
    loss_weight_coord: float = 10.0
    loss_weight_node_type: float = 1.0
    loss_weight_stop: float = 1.0
    loss_weight_parent: float = 3.0
    loss_weight_extra_edge: float = 1.0
    loss_weight_budget: float = 0.1

    # ------ Training ------
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 100
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    num_workers: int = 4
    save_every_n_epochs: int = 5

    # ------ Data ------
    train_split_path: str = "data/urban_prior/splits/train.parquet"
    val_split_path: str = "data/urban_prior/splits/val.parquet"
    use_existing_graph_as_skeleton: bool = True
    """If True, treat the skeleton_graph_json field as the ground truth skeleton."""
    max_nodes_in_training: int = 150

    # ------ Checkpoint ------
    checkpoint_dir: str = "checkpoints/spatial_conditional_skeleton_generator"

    # ------ Inference ------
    default_node_budget: int = 50
    default_edge_budget: int = 100
    decoding: str = "greedy"
    edge_threshold: float = 0.5
    temperature: float = 1.0
    force_connect_isolated: bool = True

    # ------ Seed ------
    seed: int = 42

    # ------ Style distribution cache ------
    style_dist_path: str = "data/urban_prior/urban_prior.parquet"


# Pattern names (shared across the project)
PATTERN_NAMES = ["Gridiron", "Linear", "No pattern", "Organic", "Radial", "Tributary"]

# Singleton
CONFIG = GeneratorConfig()

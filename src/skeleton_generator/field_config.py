"""Shared field generation hyper-parameters."""
from dataclasses import dataclass


@dataclass
class FieldConfig:
    # Refinement
    num_refine_steps: int = 20
    shared_refiner: bool = True
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    dim_feedforward: int = 1024

    # DETR-style slot selection
    default_num_slots: int = 150  # N_max — fixed capacity, not GT count
    max_prev_window: int = 32

    # Loss weights — denoising curriculum (v5.2): coord dominates, spread/repulsion minimal
    loss_weight_coord: float = 50.0
    loss_weight_node_type: float = 1.0
    loss_weight_existence: float = 1.0
    loss_weight_skeleton: float = 5.0
    loss_weight_degree: float = 0.3
    loss_weight_backbone: float = 0.3
    loss_weight_centrality: float = 0.1
    loss_weight_repulsion: float = 0.01
    loss_weight_spread: float = 0.01

    # Denoising curriculum
    denoise_noise_std_start: float = 0.02
    denoise_noise_std_end: float = 0.30
    denoise_uniform_ratio_start: float = 0.0
    denoise_uniform_ratio_end: float = 0.50

    # Multi-step coord weights (sum = 1)
    step_coord_weights: tuple = (
        0.005, 0.010, 0.014, 0.019, 0.024,
        0.029, 0.033, 0.038, 0.043, 0.048,
        0.052, 0.057, 0.062, 0.067, 0.071,
        0.076, 0.081, 0.086, 0.090, 0.095,
    )

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 50
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    num_workers: int = 8

    # Data
    max_nodes_in_training: int = 150
    max_nodes: int = 256
    num_node_types: int = 5
    condition_dim: int = 256
    style_dim: int = 6
    extra_condition_dim: int = 5

    # Inference
    default_node_budget: int = 50
    default_edge_budget: int = 100

    # Seed
    seed: int = 42


FIELD_CONFIG = FieldConfig()

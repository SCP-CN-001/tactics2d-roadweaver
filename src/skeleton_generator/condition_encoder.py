"""
Condition Encoder — style + structural priors → conditioning embedding.

Used by both the conditional generator and the VAE.
Checkpoint compatible: state-dict key prefix is ``condition_encoder.*``.
"""

import torch
from torch import nn

from .config import CONFIG as BASE_CFG
from .field_config import FIELD_CONFIG


class ConditionEncoder(nn.Module):
    """
    Encodes (style, structural_priors, map_size, budgets, complexity) into
    a single condition embedding.

    Input dim: 6 + 5 + 2 + 1 + 1 + 1 = 16
    Output dim: FIELD_CONFIG.condition_dim (256)
    """

    def __init__(self):
        super().__init__()
        input_dim = (
            FIELD_CONFIG.style_dim
            + FIELD_CONFIG.extra_condition_dim
            + 2  # map_size
            + 1  # node_budget
            + 1  # edge_budget
            + 1  # complexity
        )
        self.net = nn.Sequential(
            nn.Linear(input_dim, FIELD_CONFIG.condition_dim),
            nn.LayerNorm(FIELD_CONFIG.condition_dim),
            nn.ReLU(inplace=True),
            nn.Linear(FIELD_CONFIG.condition_dim, FIELD_CONFIG.condition_dim),
            nn.LayerNorm(FIELD_CONFIG.condition_dim),
            nn.ReLU(inplace=True),
            nn.Linear(FIELD_CONFIG.condition_dim, FIELD_CONFIG.condition_dim),
            nn.LayerNorm(FIELD_CONFIG.condition_dim),
        )

    def forward(
        self,
        style_vector: torch.Tensor,          # (B, 6)
        structural_priors: torch.Tensor,     # (B, 5)
        map_size: torch.Tensor,              # (B, 2)  metres
        node_budget: torch.Tensor,           # (B, 1)
        edge_budget: torch.Tensor,           # (B, 1)
        complexity: torch.Tensor,            # (B, 1)
    ) -> torch.Tensor:                       # (B, condition_dim)
        map_size_norm = map_size / BASE_CFG.map_size_scale
        nb_norm = node_budget / FIELD_CONFIG.max_nodes
        eb_norm = edge_budget / (FIELD_CONFIG.max_nodes * 2)

        sp = structural_priors.clone()
        if sp.shape[-1] >= 1:
            sp[..., 0] = sp[..., 0] / 60.0  # road_density normalise

        x = torch.cat(
            [style_vector, sp, map_size_norm, nb_norm, eb_norm, complexity],
            dim=-1,
        )
        return self.net(x)

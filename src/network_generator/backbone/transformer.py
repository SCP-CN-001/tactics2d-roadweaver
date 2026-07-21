"""
Conditional masked Transformer for code map generation.

Predicts masked code tokens given visible tokens and an 11-dim condition.
Architecture: token embed + pos embed + condition embed → Transformer → head.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class MaskedCodeModel(nn.Module):
    """
    Conditional masked code model.

    Args:
        vocab_size: number of codes + 1 MASK token.
        d_model: transformer hidden dim.
        nhead: number of attention heads.
        num_layers: number of transformer blocks.
        cond_dim: raw condition dimension (11).
        max_seq_len: code map flattened length (1024 for 32×32).
    """

    def __init__(
        self,
        vocab_size: int = 513,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        cond_dim: int = 11,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.mask_token_id = vocab_size - 1

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        # Condition projection: 11-dim → d_model
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, d_model), nn.ReLU(inplace=True), nn.Linear(d_model, d_model)
        )

        # Standard PyTorch pre-norm Transformer encoder
        layer = TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = TransformerEncoder(layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.02)

    def forward(self, code_tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            code_tokens: (B, S) token IDs.
            condition: (B, cond_dim) raw condition vector.
        Returns:
            logits: (B, S, vocab_size).
        """
        x = self.token_embed(code_tokens)
        x = x + self.pos_embed[:, : x.shape[1]]
        x = x + self.cond_embed(condition).unsqueeze(1)
        x = self.transformer(x)
        x = self.norm(x)
        return self.head(x)

    @torch.no_grad()
    def sample(
        self, condition: torch.Tensor, num_steps: int = 8, temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Iterative decoding from fully masked code map.

        Args:
            condition: (B, cond_dim).
            num_steps: number of decoding steps.
            temperature: sampling temperature.
        Returns:
            code_tokens: (B, H, W) code IDs.
        """
        B = condition.shape[0]
        device = condition.device
        S = self.max_seq_len
        side = int(S**0.5)

        tokens = torch.full((B, S), self.mask_token_id, device=device)

        for step in range(num_steps):
            curr_mask = tokens == self.mask_token_id
            if not curr_mask.any():
                break

            logits = self.forward(tokens, condition)
            probs = F.softmax(logits / temperature, dim=-1)
            confidence, predictions = probs.max(dim=-1)

            # Cosine schedule: how many tokens stay masked
            ratio = math.cos(math.pi / 2 * (step + 1) / num_steps)
            n_keep = max(1, int(S * ratio))

            flat_conf = confidence.clone()
            flat_conf[~curr_mask] = float("inf")
            threshold = flat_conf.flatten().kthvalue(n_keep + 1).values
            unmask = (confidence > threshold) & curr_mask
            tokens[unmask] = predictions[unmask]

        # Fallback for any remaining masked positions
        remaining = tokens == self.mask_token_id
        if remaining.any():
            logits = self.forward(tokens, condition)
            tokens[remaining] = logits[remaining].argmax(dim=-1)

        return tokens.reshape(B, side, side)

"""Conditional masked Transformer for code maps."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class AdaLNTransformerBlock(nn.Module):
    """
    Pre-norm Transformer block with adaptive layer norm (AdaLN) modulation.

    Condition is projected to 6 × ``d_model`` at each block:
      (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp).

    Zero-initialised so every block starts as identity.
    """

    def __init__(self, d_model: int, nhead: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        # AdaLN modulation: per-block 6 × d_model
        self.adaln = nn.Linear(cond_dim, 6 * d_model)
        nn.init.zeros_(self.adaln.weight)
        nn.init.zeros_(self.adaln.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Apply one AdaLN-transformer block.

        Args:
            x: (B, S, D) sequence.
            c: (B, D_cond) condition vector.
        Returns:
            x: (B, S, D) transformed sequence.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln(c).chunk(
            6, dim=-1
        )  # each (B, D)

        # Attention sub-layer
        x_norm = F.layer_norm(x, x.shape[-1:])
        modulated = x_norm * (1.0 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out = self.attn(modulated, modulated, modulated, need_weights=False)[0]
        x = x + gate_msa.unsqueeze(1) * attn_out

        # MLP sub-layer
        x_norm = F.layer_norm(x, x.shape[-1:])
        modulated = x_norm * (1.0 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(modulated)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


class MaskedCodeModel(nn.Module):
    """
    Conditional masked code model.

    Args:
        vocab_size: number of codes + 1 MASK token.
        d_model: transformer hidden dim.
        nhead: number of attention heads.
        num_layers: number of transformer blocks.
        cond_dim: raw condition dimension (default 11).
        max_seq_len: code map flattened length (1024 for 32×32).
        use_adaln: if True, replace additive condition with per-block AdaLN.
    """

    def __init__(
        self,
        vocab_size: int = 513,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        cond_dim: int = 11,
        max_seq_len: int = 1024,
        use_adaln: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.mask_token_id = vocab_size - 1
        self.use_adaln = use_adaln

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        if use_adaln:
            # ── AdaLN path ────────────────────────────────────────────
            # Condition is passed directly to per-block modulation —
            # no additive embedding needed at the input.
            self.cond_proj = nn.Linear(cond_dim, d_model)  # optional: project once
            self.blocks = nn.ModuleList(
                [AdaLNTransformerBlock(d_model, nhead, cond_dim) for _ in range(num_layers)]
            )
        else:
            # ── Additive path ─────────────────────────────────────────
            self.cond_embed = nn.Sequential(
                nn.Linear(cond_dim, d_model), nn.ReLU(inplace=True), nn.Linear(d_model, d_model)
            )
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

        if self.use_adaln:
            for block in self.blocks:
                x = block(x, condition)
        else:
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

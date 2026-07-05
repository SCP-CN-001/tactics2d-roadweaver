"""
Masked Code Transformer — conditional masked code completion.


Predicts masked code tokens in a 32×32 code map given:
  - visible (unmasked) code tokens
  - 11-dim condition (style + structural priors)

Architecture:
  token embed (256 + 1 MASK) → learned pos embed (1024)
  + condition embed (via MLP, broadcast)
  → Transformer encoder (6 layers, 8 heads, 512 dim)
  → linear head → logits (257)

Training: masked language modeling (random mask, predict masked only).
Inference: start from all-MASK, iteratively unmask high-confidence tokens.
"""

import torch
from torch import nn
import torch.nn.functional as F
import math


class ConditionEmbed(nn.Module):
    """Project 11-dim condition to Transformer-compatible embedding."""

    def __init__(self, cond_dim: int = 11, d_model: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, d_model)


class PositionalEncoding(nn.Module):
    """Learned 1D positional embedding for 1024 tokens."""

    def __init__(self, seq_len: int = 1024, d_model: int = 512):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embed[:, :x.shape[1]]


class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block."""

    def __init__(self, d_model: int = 512, nhead: int = 8, dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x


class MaskedCodeModel(nn.Module):
    """
    Conditional masked code model for 32×32 → 1024 token sequences.

    Args:
        vocab_size: number of codes (256) + 1 MASK token = 257
        d_model: transformer hidden dim
        nhead: number of attention heads
        num_layers: number of transformer blocks
        cond_dim: raw condition dimension (11)
    """

    MASK_TOKEN_ID = 256  # reserved MASK token ID

    def __init__(self, vocab_size: int = 257, d_model: int = 512,
                 nhead: int = 8, num_layers: int = 6, cond_dim: int = 11,
                 max_seq_len: int = 1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, d_model)

        # Condition embedding
        self.cond_embed = ConditionEmbed(cond_dim=cond_dim, d_model=d_model)

        # Positional encoding
        self.pos_embed = PositionalEncoding(seq_len=max_seq_len, d_model=d_model)

        # Transformer encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, d_model * 4)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Output head
        self.head = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.02)

    def forward(self, code_tokens: torch.Tensor,
                mask: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        """
        code_tokens: (B, 1024) — token IDs (0-255 for codes, 256 for MASK)
        mask:        (B, 1024) — 1 = masked (predict), 0 = visible
        condition:   (B, 11)   — raw style + structural priors
        Returns:     (B, 1024, vocab_size) — logits for all positions
        """
        B, S = code_tokens.shape

        # Token embeddings
        x = self.token_embed(code_tokens)  # (B, S, d_model)

        # Add positional encoding
        x = self.pos_embed(x)

        # Add condition embedding (broadcast to all tokens)
        cond = self.cond_embed(condition).unsqueeze(1)  # (B, 1, d_model)
        x = x + cond

        # Transformer
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # Output logits
        logits = self.head(x)  # (B, S, vocab_size)
        return logits

    @torch.no_grad()
    def sample(self, condition: torch.Tensor, num_steps: int = 8,
               temperature: float = 1.0, mask_schedule: str = "cosine") -> torch.Tensor:
        """
        Iterative decoding from fully masked code map.

        Args:
            condition: (B, 11)
            num_steps: number of iterative decoding steps
            temperature: sampling temperature
            mask_schedule: "cosine" or "linear"
        Returns:
            code_tokens: (B, 1024) — final predicted code IDs (0-255)
        """
        B = condition.shape[0]
        device = condition.device
        S = self.max_seq_len

        # Start from fully masked
        tokens = torch.full((B, S), self.MASK_TOKEN_ID, device=device)

        for step in range(num_steps):
            # Current mask: which tokens are still MASK
            curr_mask = (tokens == self.MASK_TOKEN_ID)

            if not curr_mask.any():
                break

            # Predict
            logits = self.forward(tokens, curr_mask, condition)  # (B, S, V)

            # Compute confidence for each position
            probs = F.softmax(logits / temperature, dim=-1)
            # Confidence = max probability (only for masked positions)
            confidence, predictions = probs.max(dim=-1)  # (B, S)

            # How many tokens to unmask in this step
            if mask_schedule == "cosine":
                ratio = math.cos(math.pi / 2 * (step + 1) / num_steps)
            else:  # linear
                ratio = 1.0 - (step + 1) / num_steps
            n_keep_masked = max(1, int(S * ratio))

            # Sort masked positions by confidence (lowest confidence = unmask last)
            flat_conf = confidence.clone()
            # Set visible positions to inf (don't unmask what's already unmasked)
            flat_conf[~curr_mask] = float('inf')

            # Get threshold: at step 0, keep 1/8 of tokens masked
            # Higher confidence = unmask first
            threshold = flat_conf.flatten().kthvalue(n_keep_masked + 1).values

            # Unmask positions with confidence > threshold
            unmask = (confidence > threshold) & curr_mask

            tokens[unmask] = predictions[unmask]
            # Remaining masked tokens stay as MASK_TOKEN_ID

        # Fallback: unmask any remaining masked positions with argmax
        remaining_mask = (tokens == self.MASK_TOKEN_ID)
        if remaining_mask.any():
            logits = self.forward(tokens, remaining_mask, condition)
            tokens[remaining_mask] = logits[remaining_mask].argmax(dim=-1)

        return tokens  # (B, 1024)

    @torch.no_grad()
    def sample_code_map(self, condition: torch.Tensor, num_steps: int = 8,
                         temperature: float = 1.0) -> torch.Tensor:
        """Sample and reshape to (B, 32, 32) code map."""
        tokens = self.sample(condition, num_steps=num_steps, temperature=temperature)
        return tokens.reshape(-1, 32, 32)  # (B, 32, 32)
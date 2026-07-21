"""
VQ-VAE: road field ↔ discrete code map.

Encoder/decoder architecture is computed programmatically from
resolution and code_map_size, eliminating redundant hardcoded branches.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import CONFIG


class ResBlock(nn.Module):
    """Pre-activation residual block with batch normalisation."""

    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu_(x + self.net(x))


class VectorQuantizer(nn.Module):
    """
    VQ-VAE quantizer with EMA codebook update.

    Uses Euclidean distance for code lookup and EMA for codebook learning,
    which is more stable than gradient-based optimisation and prevents
    codebook collapse (dead codes are periodically reset).
    """

    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", self.embedding.weight.data.clone())

    def forward(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        B, D, H, W = z_e.shape
        z_e_flat = z_e.permute(0, 2, 3, 1).reshape(-1, D)
        z_e_flat = z_e_flat.to(self.embedding.weight.dtype)

        # Code lookup by Euclidean distance
        dist = torch.cdist(z_e_flat, self.embedding.weight)
        indices_flat = dist.argmin(dim=1)

        # Quantize
        z_q_flat = self.embedding(indices_flat)
        z_q = z_q_flat.reshape(B, H, W, D).permute(0, 3, 1, 2)

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        # EMA codebook update (training only)
        if self.training:
            with torch.no_grad():
                encodings = torch.zeros(
                    indices_flat.shape[0], self.num_embeddings, device=z_e.device
                )
                encodings.scatter_(1, indices_flat.unsqueeze(1), 1)

                self.ema_cluster_size = self.decay * self.ema_cluster_size + (
                    1 - self.decay
                ) * encodings.sum(dim=0)

                n = self.ema_cluster_size.sum()
                smoothed_size = (
                    (self.ema_cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
                )

                dw = encodings.T @ z_e_flat
                self.ema_w = self.decay * self.ema_w + (1 - self.decay) * dw

                self.embedding.weight.data = self.ema_w / smoothed_size.unsqueeze(1)

                # Reset dead codes by sampling random encoder outputs
                threshold = 1.0 / self.num_embeddings * 0.1
                dead_mask = self.ema_cluster_size < threshold * n
                dead_codes = dead_mask.nonzero(as_tuple=True)[0]

                if len(dead_codes) > 0 and z_e_flat.shape[0] > len(dead_codes):
                    rand_idx = torch.randint(
                        0, z_e_flat.shape[0], (len(dead_codes),), device=z_e.device
                    )
                    self.embedding.weight.data[dead_codes] = z_e_flat[rand_idx]
                    self.ema_w[dead_codes] = z_e_flat[rand_idx].detach()
                    self.ema_cluster_size[dead_codes] = 1.0

        # Commitment loss
        commitment_loss = F.mse_loss(z_e_flat, z_q_flat.detach())

        # Metrics
        with torch.no_grad():
            counts = torch.bincount(indices_flat, minlength=self.num_embeddings).float()
            usage = (counts > 0).float().mean().item()
            prob = counts / counts.sum()
            perplexity = torch.exp(-(prob * torch.log(prob + 1e-10)).sum()).item()

        return (
            z_q_st,
            indices_flat.reshape(B, H, W),
            {
                "vq_loss": self.commitment_cost * commitment_loss,
                "codebook_loss": 0.0,
                "commitment_loss": commitment_loss.item(),
                "perplexity": perplexity,
                "usage": usage,
                "dead_codes": len(dead_codes) if self.training else 0,
            },
        )


class VQVAE(nn.Module):
    """
    VQ-VAE with programmatically computed encoder/decoder depth.

    The number of downsampling stages is determined by
    ``log2(resolution / code_map_size)``, so any power-of-two code map size
    is supported without hardcoded branches.
    """

    # Channel plan keyed by number of downsampling stages.
    # These values are tuned to match the original hardcoded architectures
    # while keeping the total parameter count reasonable across depths.
    _ENCODER_CHANNELS: dict[int, list[int]] = {
        1: [24, 48],
        2: [32, 64],
        3: [24, 48, 48],
        4: [16, 32, 32, 32],
        5: [12, 24, 24, 16, 16],
    }

    _DECODER_CHANNELS: dict[int, list[int]] = {
        1: [48, 16],
        2: [64, 16],
        3: [48, 24, 16],
        4: [32, 32, 16, 16],
        5: [24, 16, 16, 16, 16],
    }

    def __init__(
        self,
        in_ch: int = 6,
        embed_dim: int = 64,
        num_codes: int = 512,
        resolution: int | None = None,
        code_map_size: int | None = None,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
    ):
        """
        Args:
            in_ch: number of input field channels (6).
            embed_dim: code embedding dimension.
            num_codes: number of discrete codes in the codebook.
            resolution: raster field resolution (default: CONFIG.resolution).
            code_map_size: code map spatial size (default: CONFIG.code_map_size).
                Must be a power-of-two divisor of ``resolution``.
            commitment_cost: VQ commitment loss weight.
            decay: EMA decay for codebook update.
        """
        super().__init__()

        if resolution is None:
            resolution = CONFIG.resolution
        if code_map_size is None:
            code_map_size = CONFIG.code_map_size

        self.embed_dim = embed_dim
        self.num_codes = num_codes
        self.code_map_size = code_map_size
        self.code_map_hw = code_map_size
        self.resolution = resolution

        # Number of downsampling / upsampling stages
        ratio = resolution // code_map_size
        n_stages = int(math.log2(ratio))
        if 2**n_stages != ratio:
            raise ValueError(
                f"resolution / code_map_size must be a power of 2, "
                f"got {resolution} / {code_map_size} = {ratio}"
            )

        # ── Encoder ──
        enc_ch = self._ENCODER_CHANNELS[n_stages]
        self.encoder = nn.Sequential(
            *self._downsample_layers(in_ch, enc_ch),
            nn.Conv2d(enc_ch[-1], embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            ResBlock(embed_dim),
        )

        # ── Decoder ──
        dec_ch = self._DECODER_CHANNELS[n_stages]
        self.decoder = nn.Sequential(*self._upsample_layers(embed_dim, dec_ch))

        # ── Output head ──
        self.coord_conv = nn.Conv2d(2, 16, 3, padding=1)
        self.out_conv = nn.Sequential(
            nn.Conv2d(dec_ch[-1] + 16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_ch, 3, padding=1),
        )

        # ── Quantizer ──
        self.quantizer = VectorQuantizer(
            num_embeddings=num_codes,
            embedding_dim=embed_dim,
            commitment_cost=commitment_cost,
            decay=decay,
        )

    @staticmethod
    def _downsample_layers(in_ch: int, channels: list[int]) -> list[nn.Module]:
        """Build a list of stride-2 downsampling blocks."""
        layers = []
        prev = in_ch
        for ch in channels:
            layers.extend(
                [
                    nn.Conv2d(prev, ch, 4, stride=2, padding=1),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True),
                    ResBlock(ch),
                ]
            )
            prev = ch
        return layers

    @staticmethod
    def _upsample_layers(embed_dim: int, channels: list[int]) -> list[nn.Module]:
        """Build a list of stride-2 transposed-conv upsampling blocks."""
        layers = []
        prev = embed_dim
        for ch in channels:
            layers.extend(
                [
                    nn.ConvTranspose2d(prev, ch, 4, stride=2, padding=1),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True),
                    ResBlock(ch),
                ]
            )
            prev = ch
        return layers

    # ── Public interface ──────────────────────────────────────────────────

    def encode_to_code(self, field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a road field into its quantized code map."""
        z_e = self.encoder(field)
        z_q, indices, _ = self.quantizer(z_e.float())
        return z_q.to(z_e.dtype), indices

    def decode_from_code(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode a discrete code map back into a road field."""
        z_q = self.quantizer.embedding(indices)
        z_q = z_q.permute(0, 3, 1, 2)
        return self.decode(z_q)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode a quantized latent into a road field."""
        B = z_q.shape[0]
        y = torch.linspace(-1, 1, self.resolution, device=z_q.device)
        x = torch.linspace(-1, 1, self.resolution, device=z_q.device)
        gy, gx = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack([gx, gy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        grid_feat = self.coord_conv(grid)
        feat = self.decoder(z_q)
        return self.out_conv(torch.cat([feat, grid_feat], dim=1))

    def forward(self, field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        z_e = self.encoder(field)
        with torch.amp.autocast("cuda", enabled=False):
            z_q, indices, info = self.quantizer(z_e.float())
        recon = self.decode(z_q.to(z_e.dtype))
        return recon, indices, info

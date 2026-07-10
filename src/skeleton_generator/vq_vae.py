"""
VQ-VAE: road field ↔ discrete code map.

Encoder: 128→64→32, embed_dim=64
Quantizer: 512 codes × 64 dim (Euclidean distance, EMA codebook update)
Decoder: 32→64→128

Uses Exponential Moving Average (EMA) for codebook learning instead of
gradient-based optimization, which prevents codebook collapse and ensures
high perplexity throughout training.
"""

from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu_(x + self.net(x))


class VectorQuantizer(nn.Module):
    """
    VQ-VAE quantizer with EMA codebook update.

    Uses Euclidean distance for code lookup and EMA for codebook learning,
    which is more stable than gradient-based optimization and prevents
    codebook collapse (dead codes are periodically reset).
    """

    def __init__(self, num_embeddings: int = 512, embedding_dim: int = 64,
                 commitment_cost: float = 0.25, decay: float = 0.99):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay

        # Codebook embeddings (learned via EMA, not gradients)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

        # EMA accumulators
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embedding.weight.data.clone())

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            z_e: (B, D, H, W) encoder output
        Returns:
            z_q: quantized output (same shape, straight-through gradient)
            indices: (B, H, W) code indices
            info: dict with loss terms and metrics
        """
        B, D, H, W = z_e.shape
        z_e_flat = z_e.permute(0, 2, 3, 1).reshape(-1, D)
        # Match embedding dtype (AMP produces float16 encoder output)
        z_e_flat = z_e_flat.to(self.embedding.weight.dtype)

        # --- Code lookup by Euclidean distance ---
        dist = torch.cdist(z_e_flat, self.embedding.weight)  # (N, num_embeddings)
        indices_flat = dist.argmin(dim=1)  # (N,)

        # --- Quantize ---
        z_q_flat = self.embedding(indices_flat)
        z_q = z_q_flat.reshape(B, H, W, D).permute(0, 3, 1, 2)

        # --- Straight-through estimator ---
        z_q_st = z_e + (z_q - z_e).detach()

        # --- EMA codebook update (training only) ---
        if self.training:
            with torch.no_grad():
                # One-hot encoding of assignments
                encodings = torch.zeros(
                    indices_flat.shape[0], self.num_embeddings,
                    device=z_e.device
                )
                encodings.scatter_(1, indices_flat.unsqueeze(1), 1)

                # Update EMA cluster size
                self.ema_cluster_size = self.decay * self.ema_cluster_size + \
                    (1 - self.decay) * encodings.sum(dim=0)

                # Laplace smoothing to avoid zero cluster sizes
                n = self.ema_cluster_size.sum()
                smoothed_size = (self.ema_cluster_size + 1e-5) / \
                    (n + self.num_embeddings * 1e-5) * n

                # Update EMA codebook vectors
                dw = encodings.T @ z_e_flat  # (num_embeddings, D)
                self.ema_w = self.decay * self.ema_w + \
                    (1 - self.decay) * dw

                # Update codebook
                self.embedding.weight.data = \
                    self.ema_w / smoothed_size.unsqueeze(1)

                # --- Codebook reset: replace dead codes ---
                # A code is dead if its EMA cluster size is near zero
                threshold = 1.0 / self.num_embeddings * 0.1
                dead_mask = self.ema_cluster_size < threshold * n
                dead_codes = dead_mask.nonzero(as_tuple=True)[0]

                if len(dead_codes) > 0 and z_e_flat.shape[0] > len(dead_codes):
                    # Sample random encoder outputs to replace dead codes
                    rand_idx = torch.randint(
                        0, z_e_flat.shape[0], (len(dead_codes),),
                        device=z_e.device
                    )
                    self.embedding.weight.data[dead_codes] = z_e_flat[rand_idx]
                    self.ema_w[dead_codes] = z_e_flat[rand_idx].detach()
                    # Reset EMA cluster size for revived codes
                    self.ema_cluster_size[dead_codes] = 1.0

        # --- Commitment loss ---
        commitment_loss = F.mse_loss(z_e_flat, z_q_flat.detach())

        # --- Metrics ---
        with torch.no_grad():
            counts = torch.bincount(
                indices_flat, minlength=self.num_embeddings).float()
            usage = (counts > 0).float().mean().item()
            prob = counts / counts.sum()
            perplexity = torch.exp(
                -(prob * torch.log(prob + 1e-10)).sum()).item()

        return z_q_st, indices_flat.reshape(B, H, W), {
            "vq_loss": self.commitment_cost * commitment_loss,  # only commitment loss with EMA
            "codebook_loss": 0.0,
            "commitment_loss": commitment_loss.item(),
            "perplexity": perplexity,
            "usage": usage,
            "dead_codes": len(dead_codes) if self.training else 0,
        }


class VQVAE(nn.Module):
    def __init__(self, in_ch: int = 6, embed_dim: int = 64, num_codes: int = 512,
                 resolution: int = 128, commitment_cost: float = 0.25,
                 decay: float = 0.99, code_map_size: int = 32):
        """
        Args:
            in_ch: input channels (6).
            embed_dim: code embedding dimension.
            num_codes: number of discrete codes.
            resolution: field resolution (128).
            commitment_cost: VQ commitment loss weight.
            decay: EMA decay for codebook update.
            code_map_size: 32 (current, 4× down) or 64 (ablation, 2× down).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_codes = num_codes
        self.code_map_size = code_map_size
        self.code_map_hw = resolution // (4 if code_map_size == 32 else 2)
        self.resolution = resolution

        # ── Encoder ──
        if code_map_size == 32:
            # Current: 128→64→32 (2× stride-2)
            self.encoder = nn.Sequential(
                nn.Conv2d(in_ch, 32, 3, stride=2, padding=1),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True), ResBlock(32),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True), ResBlock(64),
                nn.Conv2d(64, embed_dim, 3, padding=1),
                nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True), ResBlock(embed_dim),
            )
            # Decoder: 32→64→128 (2× stride-2 tconv)
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, 64, 4, stride=2, padding=1),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True), ResBlock(64),
                nn.ConvTranspose2d(64, 16, 4, stride=2, padding=1),
                nn.BatchNorm2d(16), nn.ReLU(inplace=True), ResBlock(16),
            )
        else:
            # 64×64 code map: 128→64 (1× stride-2), more depth at 64×64
            self.encoder = nn.Sequential(
                nn.Conv2d(in_ch, 24, 3, stride=2, padding=1),
                nn.BatchNorm2d(24), nn.ReLU(inplace=True), ResBlock(24),
                nn.Conv2d(24, 48, 3, stride=1, padding=1),
                nn.BatchNorm2d(48), nn.ReLU(inplace=True), ResBlock(48), ResBlock(48),
                nn.Conv2d(48, embed_dim, 3, padding=1),
                nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True), ResBlock(embed_dim),
            )
            # Decoder: 64→128 (1× stride-2 tconv)
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, 48, 4, stride=2, padding=1),
                nn.BatchNorm2d(48), nn.ReLU(inplace=True), ResBlock(48), ResBlock(48),
                nn.ConvTranspose2d(48, 16, 3, stride=1, padding=1),
                nn.BatchNorm2d(16), nn.ReLU(inplace=True), ResBlock(16),
            )

        self.quantizer = VectorQuantizer(
            num_embeddings=num_codes, embedding_dim=embed_dim,
            commitment_cost=commitment_cost, decay=decay)

        self.coord_conv = nn.Conv2d(2, 16, 3, padding=1)
        self.out_conv = nn.Sequential(
            nn.Conv2d(16 + 16, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 6, 3, padding=1),
        )

    def encode_to_code(self, field: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_e = self.encoder(field)
        z_q, indices, _ = self.quantizer(z_e.float())
        return z_q.to(z_e.dtype), indices

    def decode_from_code(self, indices: torch.Tensor) -> torch.Tensor:
        z_q = self.quantizer.embedding(indices)
        z_q = z_q.permute(0, 3, 1, 2)
        return self.decode(z_q)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        B = z_q.shape[0]
        y = torch.linspace(-1, 1, self.resolution, device=z_q.device)
        x = torch.linspace(-1, 1, self.resolution, device=z_q.device)
        gy, gx = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack([gx, gy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        grid_feat = self.coord_conv(grid)
        feat = self.decoder(z_q)
        return self.out_conv(torch.cat([feat, grid_feat], dim=1))

    def forward(self, field: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        z_e = self.encoder(field)
        with torch.cuda.amp.autocast(enabled=False):
            z_q, indices, info = self.quantizer(z_e.float())
        return self.decode(z_q.to(z_e.dtype)), indices, info

"""
VQ-VAE: road field ↔ discrete code map.

Encoder: 128→64→32, embed_dim=64
Quantizer: 256 codes × 64 dim (cosine similarity)
Decoder: 32→64→128
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
    def __init__(self, num_embeddings: int = 256, embedding_dim: int = 64,
                 commitment_cost: float = 0.1):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.normal_()
        self.embedding.weight.data = F.normalize(self.embedding.weight.data, dim=1)

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        B, D, H, W = z_e.shape
        z_e_flat = z_e.permute(0, 2, 3, 1).reshape(-1, D)
        z_e_norm = F.normalize(z_e_flat, dim=1)
        cb_norm = F.normalize(self.embedding.weight, dim=1)
        sim = z_e_norm @ cb_norm.T
        indices_flat = sim.argmax(dim=1)
        z_q_flat = self.embedding(indices_flat)
        z_q = z_q_flat.reshape(B, H, W, D).permute(0, 3, 1, 2)
        z_q_st = z_e + (z_q - z_e).detach()
        codebook_loss = F.mse_loss(z_q_flat, z_e_flat.detach())
        commitment_loss = F.mse_loss(z_e_flat, z_q_flat.detach())
        with torch.no_grad():
            counts = torch.bincount(indices_flat, minlength=self.num_embeddings).float()
            usage = (counts > 0).float().mean().item()
            prob = counts / counts.sum()
            perplexity = torch.exp(-(prob * torch.log(prob + 1e-10)).sum()).item()
        return z_q_st, indices_flat.reshape(B, H, W), {
            "vq_loss": codebook_loss + self.commitment_cost * commitment_loss,
            "codebook_loss": codebook_loss.item(),
            "commitment_loss": commitment_loss.item(),
            "perplexity": perplexity, "usage": usage,
        }


class VQVAE(nn.Module):
    def __init__(self, in_ch: int = 6, embed_dim: int = 64, num_codes: int = 256,
                 resolution: int = 128, commitment_cost: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_codes = num_codes
        self.code_map_hw = resolution // 4
        self.resolution = resolution
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), ResBlock(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), ResBlock(64),
            nn.Conv2d(64, embed_dim, 3, padding=1), nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True), ResBlock(embed_dim),
        )
        self.quantizer = VectorQuantizer(num_embeddings=num_codes, embedding_dim=embed_dim, commitment_cost=commitment_cost)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), ResBlock(64),
            nn.ConvTranspose2d(64, 16, 4, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), ResBlock(16),
        )
        self.coord_conv = nn.Conv2d(2, 16, 3, padding=1)
        self.out_conv = nn.Sequential(
            nn.Conv2d(16 + 16, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 6, 3, padding=1),
        )

    def encode_to_code(self, field: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_e = self.encoder(field)
        z_q, indices, _ = self.quantizer(z_e)
        return z_q, indices

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
        z_q, indices, info = self.quantizer(z_e)
        return self.decode(z_q), indices, info

"""
ResUNet Field Refiner — repairs fragmented VQ-decoder fields.

Takes the 6-channel VQ decoder output (128×128, often fragmented)
and produces a cleaner, more continuous road field.

Architecture: 4-level ResUNet with skip connections
  Encoder: 128 → 64 → 32 → 16 → 8 (bottleneck)
  Decoder:   8 → 16 → 32 → 64 → 128
  All internals use pre-activation ResBlocks.

Usage:
    refiner = ResUNet()
    refined_field = refiner(vq_decoder_output)  # (B, 6, 128, 128)
"""

import torch
from torch import nn


class ResBlock(nn.Module):
    """Pre-activation ResBlock."""

    def __init__(self, ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
        ]
        if dropout > 0:
            layers.insert(4, nn.Dropout2d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ResUNet(nn.Module):
    """
    4-level ResUNet for 128×128 field refinement.

    Levels (H×W → ch):
      in_proj:      128 → 32
      enc1/down1:    64 → 64
      enc2/down2:    32 → 128
      enc3/down3:    16 → 256
      bottle:         8 → 512
      dec3/up3:      16 → 256   (skip: enc3)
      dec2/up2:      32 → 128   (skip: enc2)
      dec1/up1:      64 → 64    (skip: enc1)
      out:          128 → out_ch (skip: in_proj)

    Args:
        in_ch: input channels (6).
        out_ch: output channels (6).
        base_ch: channel count at first level (default 32).
        dropout: ResBlock dropout rate.
    """

    def __init__(self, in_ch: int = 6, out_ch: int = 6,
                 base_ch: int = 32, dropout: float = 0.0):
        super().__init__()
        C = base_ch

        # ── Input projection ──
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_ch, C, 3, padding=1),
            nn.BatchNorm2d(C), nn.ReLU(inplace=True),
            ResBlock(C, dropout),
        )

        # ── Encoder down blocks (stride-2) ──
        self.down1 = nn.Sequential(  # 128→64
            nn.Conv2d(C, C * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(C * 2), nn.ReLU(inplace=True),
            ResBlock(C * 2, dropout), ResBlock(C * 2, dropout),
        )
        self.down2 = nn.Sequential(  # 64→32
            nn.Conv2d(C * 2, C * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(C * 4), nn.ReLU(inplace=True),
            ResBlock(C * 4, dropout), ResBlock(C * 4, dropout),
        )
        self.down3 = nn.Sequential(  # 32→16
            nn.Conv2d(C * 4, C * 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(C * 8), nn.ReLU(inplace=True),
            ResBlock(C * 8, dropout), ResBlock(C * 8, dropout),
        )

        # ── Bottleneck (8×8) ──
        self.bottle = nn.Sequential(
            nn.Conv2d(C * 8, C * 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(C * 16), nn.ReLU(inplace=True),
            ResBlock(C * 16, dropout), ResBlock(C * 16, dropout),
            nn.ConvTranspose2d(C * 16, C * 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(C * 8), nn.ReLU(inplace=True),
        )

        # ── Decoder up blocks (stride-2 transpose, then conv after concat) ──
        # Each up block: upconv → concat skip → ResBlocks
        self.up3 = nn.Sequential(  # 16→32
            nn.ConvTranspose2d(C * 8, C * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(C * 4), nn.ReLU(inplace=True),
        )
        self.up3_conv = nn.Sequential(
            ResBlock(C * 4 + C * 4, dropout),  # +skip from enc2/down2
            ResBlock(C * 4 + C * 4, dropout),
        )

        self.up2 = nn.Sequential(  # 32→64
            nn.ConvTranspose2d(C * 4 + C * 4, C * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(C * 2), nn.ReLU(inplace=True),
        )
        self.up2_conv = nn.Sequential(
            ResBlock(C * 2 + C * 2, dropout),  # +skip from enc1/down1
            ResBlock(C * 2 + C * 2, dropout),
        )

        self.up1 = nn.Sequential(  # 64→128
            nn.ConvTranspose2d(C * 2 + C * 2, C, 4, stride=2, padding=1),
            nn.BatchNorm2d(C), nn.ReLU(inplace=True),
        )
        self.up1_conv = nn.Sequential(
            ResBlock(C + C, dropout),  # +skip from in_proj
            ResBlock(C + C, dropout),
        )

        # ── Output projection (d1=64ch + s0=32ch = 96ch) ──
        self.out_conv = nn.Sequential(
            nn.Conv2d(C + C + C, C, 3, padding=1),  # 96 → 32
            nn.BatchNorm2d(C), nn.ReLU(inplace=True),
            nn.Conv2d(C, out_ch, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        s0 = self.in_proj(x)   # 128
        s1 = self.down1(s0)    # 64
        s2 = self.down2(s1)    # 32
        s3 = self.down3(s2)    # 16
        b = self.bottle(s3)    # 16

        # Decode with skip connections
        # up3: b (16×256) → up → 32×128, concat s2 (32×128) → process
        d3 = self.up3(b)
        d3 = self.up3_conv(torch.cat([d3, s2], dim=1))  # (B, 256, 32, 32)

        # up2: d3 (32×256) → up → 64×64, concat s1 (64×64) → process
        d2 = self.up2(d3)
        d2 = self.up2_conv(torch.cat([d2, s1], dim=1))  # (B, 128, 64, 64)

        # up1: d2 (64×128) → up → 128×32, concat s0 (128×32) → process
        d1 = self.up1(d2)
        d1 = self.up1_conv(torch.cat([d1, s0], dim=1))  # (B, 64, 128, 128)

        # Output
        return self.out_conv(torch.cat([d1, s0], dim=1))  # (B, 6, 128, 128)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

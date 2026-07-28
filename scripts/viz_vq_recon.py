#!/usr/bin/env python3
"""
Visualise VQ-VAE reconstruction: original vs decoded road fields.

Usage:
    python scripts/viz_vq_recon.py
"""

from __future__ import annotations

import os

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from network_generator.backbone.config import CONFIG
from network_generator.backbone.dataset import make_field_dataloader
from network_generator.backbone.vq_vae import VQVAE

CKPT = "runtimes/vq_vae_5km_32/checkpoints/best.pth"
OUTPUT = "analysis/vq_recon"
N_SAMPLES = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    os.makedirs(OUTPUT, exist_ok=True)

    CONFIG.train_split_path = "data/urban_prior/5km/splits/train.parquet"
    CONFIG.val_split_path = "data/urban_prior/5km/splits/val.parquet"

    loader = make_field_dataloader(
        "val", batch_size=N_SAMPLES, shuffle=True, num_workers=2, limit_samples=200, resolution=128
    )

    # Load model
    model = VQVAE(resolution=128, num_codes=1024, code_map_size=32).to(DEVICE)
    state = torch.load(CKPT, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {CKPT}")

    batch = next(iter(loader))
    field = batch["field"].to(DEVICE)

    with torch.no_grad():
        recon, indices, info = model(field)

    # Compute IoU
    orig_bin = (field[:, 0] > 0.5).float()
    rec_bin = (torch.sigmoid(recon[:, 0]) > 0.5).float()
    inter = (orig_bin * rec_bin).sum(dim=(1, 2))
    union = orig_bin.sum(dim=(1, 2)) + rec_bin.sum(dim=(1, 2)) - inter
    ious = inter / (union + 1e-8)

    # Plot original vs reconstructed
    for i in range(min(N_SAMPLES, len(field))):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        orig = field[i, 0].cpu().numpy()
        rec = torch.sigmoid(recon[i, 0]).cpu().numpy()
        diff = np.abs(orig - rec)

        axes[0].imshow(orig, cmap="gray_r", vmin=0, vmax=1)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(rec, cmap="gray_r", vmin=0, vmax=1)
        axes[1].set_title("Reconstructed")
        axes[1].axis("off")

        axes[2].imshow(diff, cmap="hot", vmin=0, vmax=0.5)
        axes[2].set_title(f"Error (IoU={ious[i]:.3f})")
        axes[2].axis("off")

        plt.tight_layout()
        path = os.path.join(OUTPUT, f"recon_{i}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")

    # Summary grid
    fig, axes = plt.subplots(N_SAMPLES, 2, figsize=(8, 4 * N_SAMPLES))
    for i in range(N_SAMPLES):
        orig = field[i, 0].cpu().numpy()
        rec = torch.sigmoid(recon[i, 0]).cpu().numpy()
        axes[i, 0].imshow(orig, cmap="gray_r", vmin=0, vmax=1)
        axes[i, 0].set_title(f"Original {i}")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(rec, cmap="gray_r", vmin=0, vmax=1)
        axes[i, 1].set_title(f"Recon {i}")
        axes[i, 1].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, "recon_grid.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT}/recon_grid.png")
    print(f"  Avg IoU: {ious.mean():.4f}")


if __name__ == "__main__":
    main()

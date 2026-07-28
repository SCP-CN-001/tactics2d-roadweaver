#!/usr/bin/env python3
"""
VQ-VAE training: road field → discrete code map → field reconstruction.

Usage:
    conda activate road-weaver
    python train_vq_vae.py                                # uses config_vq_vae.yaml
    python train_vq_vae.py --config my_config.yaml         # custom config
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict

import torch
import yaml
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import wandb
from loss import FieldLoss
from network_generator.backbone.config import CONFIG
from network_generator.backbone.dataset import make_field_dataloader
from network_generator.backbone.vq_vae import VQVAE

DEFAULT_CONFIG = "config_vq_vae.yaml"


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():
    import argparse

    parser = argparse.ArgumentParser(description="VQ-VAE training")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Apply config overrides
    CONFIG.resolution = cfg.get("resolution", CONFIG.resolution)
    CONFIG.code_map_size = cfg.get("code_map_size", CONFIG.code_map_size)
    if data_dir := cfg.get("data_dir"):
        CONFIG.train_split_path = os.path.join(data_dir, "train.parquet")
        CONFIG.val_split_path = os.path.join(data_dir, "val.parquet")

    resolution = CONFIG.resolution
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = cfg.get("output_dir", "runtimes/vq_vae")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    # Data
    train_loader = make_field_dataloader(
        "train",
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 8),
        limit_samples=cfg.get("limit_train"),
        resolution=resolution,
    )
    val_loader = make_field_dataloader(
        "val",
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 8),
        limit_samples=cfg.get("limit_val", 200),
        resolution=resolution,
    )

    # Model
    model = VQVAE(
        resolution=resolution,
        num_codes=cfg.get("num_codes", 512),
        code_map_size=CONFIG.code_map_size,
        embed_dim=cfg.get("embed_dim", 64),
        commitment_cost=cfg.get("commitment_cost", 0.25),
        decay=cfg.get("decay", 0.99),
        spread_margin=cfg.get("spread_margin", 0.0),
        spread_weight=cfg.get("spread_weight", 0.0),
    ).to(device)
    print(
        f"[VQ-VAE] Code map: {model.code_map_hw}x{model.code_map_hw}, "
        f"codebook: {model.num_codes}x{model.embed_dim}"
    )
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Wandb ──
    wandb_project = cfg.get("wandb_project", "roadweaver-vq").strip()
    if wandb_project:
        wandb.init(project=wandb_project, config=cfg, settings=wandb.Settings(console="off"))
        wandb.watch(model, log="gradients", log_freq=50)
        print(f"  [wandb] Logging to {wandb_project}")

    # Loss, optimiser, scheduler
    criterion = FieldLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"] - 5, eta_min=cfg["lr"] * 0.01)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    train_log, best_loss = [], float("inf")
    for epoch in range(cfg["epochs"]):
        t0 = time.time()
        model.train()
        losses, n_batches = defaultdict(float), 0

        for batch in tqdm(train_loader, desc=f"E{epoch}", leave=False):
            target = batch["field"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                recon, _, vq_info = model(target)
                ls = criterion(recon, target)
            total = ls["total"] + vq_info["vq_loss"] + vq_info["code_spread_weighted"]
            if not (torch.isnan(total) or torch.isinf(total)):
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                for k, v in ls.items():
                    losses[k] += v.item() if isinstance(v, torch.Tensor) else v
                losses["vq_loss"] += vq_info["vq_loss"].item()
                losses["code_spread_loss"] += vq_info["code_spread_loss"]
                losses["perplexity"] += vq_info["perplexity"]
                losses["code_usage"] += vq_info["usage"]
                n_batches += 1

        if epoch >= 5:
            scheduler.step()

        model.eval()
        val_ls, vn = defaultdict(float), 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False):
                target = batch["field"].to(device)
                with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                    recon, _, vq_info = model(target)
                    ls = criterion(recon, target)
                for k, v in ls.items():
                    val_ls[k] += v.item() if isinstance(v, torch.Tensor) else v
                val_ls["vq_loss"] += vq_info["vq_loss"].item()
                val_ls["code_spread_loss"] += vq_info["code_spread_loss"]
                val_ls["perplexity"] += vq_info["perplexity"]
                val_ls["code_usage"] += vq_info["usage"]
                vn += 1

        avg = {f"train_{k}": v / n_batches for k, v in losses.items()}
        avg.update({f"val_{k}": v / vn for k, v in val_ls.items()})
        avg["epoch"] = epoch
        avg["lr"] = optimizer.param_groups[0]["lr"]
        avg["time_s"] = round(time.time() - t0, 1)
        if wandb_project:
            wandb.log(avg)
        train_log.append(avg)

        print(
            f"E{epoch:2d} | val_iou={avg.get('val_road_iou', 0):.4f} "
            f"vq={avg.get('val_vq_loss', 0):.4f} "
            f"ppl={avg.get('val_perplexity', 0):.1f} "
            f"usage={avg.get('val_code_usage', 0):.2%}"
        )

        if avg.get("val_total", 1) < best_loss:
            best_loss = avg["val_total"]
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict()},
                os.path.join(output_dir, "checkpoints", "best.pth"),
            )
        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict()},
            os.path.join(output_dir, "checkpoints", "latest.pth"),
        )
        with open(os.path.join(output_dir, "train_log.json"), "w") as f:
            json.dump(train_log, f, indent=2)

    if wandb_project:
        wandb.finish()
    print(f"Done. Best val_total={best_loss:.4f}")


if __name__ == "__main__":
    main()

"""
VQ-VAE Training: road field → discrete code map → field reconstruction.

Usage:
    cd /home/rowena/Documents/tactics2d-roadweaver
    PYTHONPATH=src:$PYTHONPATH conda run -n road-weaver \
        python scripts/train/train_vqae.py
"""

import argparse
import json
import os
import time
from collections import defaultdict

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import wandb
from src.skeleton_generator.config import CONFIG
from src.skeleton_generator.losses import FieldLoss
from src.skeleton_generator.vq_vae import VQVAE


def parse_args():
    p = argparse.ArgumentParser(description="VQ-VAE Training")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-val", type=int, default=200)
    p.add_argument("--output-dir", type=str, default="runtimes/vq_vae")
    p.add_argument(
        "--wandb-project",
        type=str,
        default="roadweaver-ae",
        help="Wandb project name. Empty string = no logging.",
    )
    p.add_argument("--num-codes", type=int, default=512, help="Number of VQ codebook entries")
    p.add_argument(
        "--code-map-size",
        type=int,
        default=32,
        choices=[32, 64],
        help="Code map resolution (32 = current, 64 = ablation)",
    )
    p.add_argument(
        "--resolution",
        type=int,
        default=128,
        help="Raster field resolution (128 for 2km, 256 for 5km)",
    )
    p.add_argument(
        "--map-size",
        type=float,
        default=2000.0,
        help="Map size in meters (2000 for 2km, 5000 for 5km)",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data split directory (e.g. data/urban_prior/splits_5km)",
    )
    p.add_argument(
        "--style-loss-weight",
        type=float,
        default=0.0,
        help="Weight for style prediction auxiliary loss (0 = disable). "
        "If >0, trains VQ with style-aware supervision on z_q.",
    )
    p.add_argument(
        "--style-target-cols",
        type=int,
        nargs="+",
        default=[0, 1, 3, 4],
        help="Indices of structural_priors columns to regress. "
        "Default [0,1,3,4] = density, gridness, organic, bearing_entropy "
        "(excludes radialness at index 2).",
    )
    p.add_argument(
        "--style-weights",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5, 1.5],
        help="Per-attribute loss weights (must match len(style-target-cols)).",
    )
    return p.parse_args()


def main():
    args = parse_args()


class VQLoss(nn.Module):
    """VQ-VAE loss combining field reconstruction + VQ commitment."""

    def __init__(self):
        super().__init__()
        self.field_loss = FieldLoss()

    def forward(self, recon_logits, target, vq_info):
        ls = self.field_loss(recon_logits, target)
        ls["vq_loss"] = vq_info["vq_loss"]
        ls["codebook_loss"] = vq_info["codebook_loss"]
        ls["commitment_loss"] = vq_info["commitment_loss"]
        ls["perplexity"] = vq_info["perplexity"]
        ls["code_usage"] = vq_info["usage"]
        ls["total"] = ls["total"] + vq_info["vq_loss"]
        return ls


def main():
    args = parse_args()

    # Apply config overrides from CLI
    CONFIG.resolution = args.resolution
    CONFIG.code_map_size = args.code_map_size
    CONFIG.map_size_scale = args.map_size
    if args.data_dir:
        CONFIG.train_split_path = os.path.join(args.data_dir, "train.parquet")
        CONFIG.val_split_path = os.path.join(args.data_dir, "val.parquet")

    from src.skeleton_generator.skeleton_dataset import (
        make_field_dataloader,  # delayed import uses updated CONFIG
    )

    resolution = CONFIG.resolution

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    train_loader = make_field_dataloader(
        "train",
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        limit_samples=args.limit_train,
        resolution=resolution,
    )
    val_loader = make_field_dataloader(
        "val",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        limit_samples=args.limit_val,
        resolution=resolution,
    )

    # Style-aware VQ configuration
    use_style = args.style_loss_weight > 0
    n_style = len(args.style_target_cols)
    style_weights = torch.tensor(args.style_weights[:n_style], device=device)
    # Normalise style weights to sum to n_style
    style_weights = style_weights / style_weights.mean()

    model = VQVAE(
        resolution=resolution,
        num_codes=args.num_codes,
        code_map_size=args.code_map_size,
        use_style_head=use_style,
        n_style_attrs=n_style,
    ).to(device)
    print(
        f"[VQ-VAE] Code map: {model.code_map_hw}x{model.code_map_hw}, codebook: {model.num_codes}x{model.embed_dim}"
    )
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    if use_style:
        # Precompute normalisation stats for style targets (train set)
        style_all = []
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Computing style stats", leave=False):
                style_all.append(batch["structural_priors"][:, args.style_target_cols])
        style_all = torch.cat(style_all, dim=0)
        style_mean = style_all.mean(dim=0).to(device)
        style_std = style_all.std(dim=0).clamp(min=1e-6).to(device)
        print(
            f"  Style normalisation: mean={style_mean.cpu().tolist()}, std={style_std.cpu().tolist()}"
        )
        print(
            f"  Style head: {n_style} attrs, weight={args.style_loss_weight:.3f}, "
            f"per-attr weights: {style_weights.tolist()}"
        )

    if args.wandb_project:
        wandb.init(project=args.wandb_project, config=vars(args))
        wandb.watch(model, log="gradients", log_freq=100)

    criterion = VQLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - 5, eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    train_log, best_loss = [], float("inf")
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        losses, n_batches = defaultdict(float), 0

        for batch in tqdm(train_loader, desc=f"E{epoch}", leave=False):
            target = batch["field"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                if use_style:
                    recon, indices, vq_info, style_pred = model(target)
                    # Normalised style loss on selected structural priors
                    raw_target = batch["structural_priors"][:, args.style_target_cols].to(device)
                    style_target = (raw_target - style_mean) / style_std
                    style_loss = (style_weights * (style_pred - style_target) ** 2).mean()
                else:
                    recon, indices, vq_info = model(target)
                ls = criterion(recon, target, vq_info)
            total = ls["total"]
            if use_style:
                total = total + args.style_loss_weight * style_loss
                ls["style_loss"] = style_loss.detach()
            if not (torch.isnan(total) or torch.isinf(total)):
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                for k, v in ls.items():
                    losses[k] += v.item() if isinstance(v, torch.Tensor) else v
                n_batches += 1

        if epoch >= 5:
            scheduler.step()
        model.eval()
        val_ls, vn = defaultdict(float), 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False):
                target = batch["field"].to(device)
                with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                    if use_style:
                        recon, indices, vq_info, style_pred = model(target)
                        raw_target = batch["structural_priors"][:, args.style_target_cols].to(
                            device
                        )
                        style_target = (raw_target - style_mean) / style_std
                        style_loss = (style_weights * (style_pred - style_target) ** 2).mean()
                    else:
                        recon, indices, vq_info = model(target)
                    ls = criterion(recon, target, vq_info)
                if use_style:
                    ls["style_loss"] = style_loss
                for k, v in ls.items():
                    val_ls[k] += v.item() if isinstance(v, torch.Tensor) else v
                vn += 1

        avg = {f"train_{k}": v / n_batches for k, v in losses.items()}
        avg.update({f"val_{k}": v / vn for k, v in val_ls.items()})
        avg["epoch"] = epoch
        avg["lr"] = optimizer.param_groups[0]["lr"]
        avg["time_s"] = round(time.time() - t0, 1)
        train_log.append(avg)

        line = (
            f"E{epoch:2d} | val_iou={avg.get('val_road_iou',0):.4f} "
            f"vq={avg.get('val_vq_loss',0):.4f}"
        )
        if use_style:
            line += f" sty={avg.get('val_style_loss',0):.5f}"
        line += f" ppl={avg.get('val_perplexity',0):.1f} usage={avg.get('val_code_usage',0):.2%}"
        print(line)

        if args.wandb_project:
            wandb.log(avg)
        if avg.get("val_total", 1) < best_loss:
            best_loss = avg["val_total"]
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict()},
                os.path.join(args.output_dir, "checkpoints", "best.pth"),
            )
        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict()},
            os.path.join(args.output_dir, "checkpoints", "latest.pth"),
        )
        with open(os.path.join(args.output_dir, "train_log.json"), "w") as f:
            json.dump(train_log, f, indent=2)

    if args.wandb_project:
        wandb.finish()
    print(f"Done. Best val_total={best_loss:.4f}")


if __name__ == "__main__":
    main()

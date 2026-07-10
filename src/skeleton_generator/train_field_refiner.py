"""
ResUNet Field Refiner Training.

Trains a lightweight ResUNet to repair fragmented VQ-decoder outputs.

Pipeline:
  1. Freeze VQ-VAE.
  2. For each batch: GT field → VQ encode → VQ decode (fragmented)
     → ResUNet → refined field → compare with GT field.
  3. Loss: FieldLoss (BCE + Dice + orientation + junction/endpoint).

Usage:
    PYTHONPATH=src:$PYTHONPATH conda run -n road-weaver \
        python -m src.skeleton_generator.train_field_refiner
"""

import argparse
import json
import os
import time
from collections import defaultdict

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import CONFIG
from .field_refiner import ResUNet
from .vq_vae import VQVAE
from .losses import FieldLoss


def parse_args():
    p = argparse.ArgumentParser(description="ResUNet Field Refiner Training")
    p.add_argument("--vq-checkpoint", default="checkpoints/skeleton_generator/vq_vae.pth")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--output-dir", default="runtimes/field_refiner")
    p.add_argument("--limit-train", type=int, default=0,
                   help="0 = all (53430)")
    p.add_argument("--limit-val", type=int, default=200)
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--wandb-project", type=str, default="roadweaver-ae",
                   help="Wandb project. Empty = no logging.")
    p.add_argument("--resolution", type=int, default=128,
                   help="Raster field resolution (128 for 2km, 256 for 5km)")
    p.add_argument("--code-map-size", type=int, default=32, choices=[32, 64],
                   help="VQ code map size (must match VQ-VAE checkpoint)")
    p.add_argument("--map-size", type=float, default=2000.0,
                   help="Map size in meters")
    p.add_argument("--data-dir", type=str, default=None,
                   help="Override data split directory")
    p.add_argument("--num-codes", type=int, default=512,
                   help="Number of VQ codebook entries (must match VQ-VAE checkpoint)")
    return p.parse_args()


def main():
    args = parse_args()

    # Apply config overrides from CLI
    CONFIG.resolution = args.resolution
    CONFIG.code_map_size = args.code_map_size
    CONFIG.map_size_scale = args.map_size
    if args.data_dir:
        CONFIG.train_split_path = os.path.join(args.data_dir, "train.parquet")
        CONFIG.val_split_path = os.path.join(args.data_dir, "val.parquet")

    from .skeleton_dataset import make_field_dataloader  # delayed import
    resolution = CONFIG.resolution

    args.limit_train = args.limit_train if args.limit_train > 0 else None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    # ── Frozen VQ-VAE ──
    print("[Train] Loading frozen VQ-VAE...")
    vq = VQVAE(resolution=resolution, num_codes=args.num_codes,
               code_map_size=args.code_map_size).to(device)
    vq.eval()
    for p in vq.parameters():
        p.requires_grad = False
    state = torch.load(args.vq_checkpoint, map_location=device, weights_only=True)
    vs = vq.state_dict()
    for k, v in state["model_state_dict"].items():
        if k in vs and v.shape == vs[k].shape:
            vs[k] = v
    vq.load_state_dict(vs, strict=False)
    print(f"  VQ-VAE frozen ({sum(p.numel() for p in vq.parameters()):,} params)")

    # ── Data loaders ──
    print("[Train] Loading data...")
    train_loader = make_field_dataloader(
        "train", batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, limit_samples=args.limit_train,
        resolution=resolution,
    )
    val_loader = make_field_dataloader(
        "val", batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, limit_samples=args.limit_val,
        resolution=resolution,
    )

    # ── Model ──
    refiner = ResUNet(in_ch=6, out_ch=6, base_ch=args.base_ch,
                      dropout=args.dropout).to(device)
    print(f"[Train] ResUNet: {refiner.param_count:,} params "
          f"({refiner.param_count/1e6:.2f}M)")

    # ── Loss ──
    field_loss = FieldLoss().to(device)

    # ── Optimiser ──
    opt = optim.AdamW(refiner.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs - 3, eta_min=args.lr * 0.01)

    # ── Wandb ──
    use_wandb = bool(args.wandb_project)
    if use_wandb:
        import wandb
        run = wandb.init(project=args.wandb_project,
                         name=f"field_refiner_{args.base_ch}ch",
                         config=vars(args),
                         settings=wandb.Settings(init_timeout=120))
        # Remove images from config to avoid serialization error
        run.config.update({"vq_params": sum(p.numel() for p in vq.parameters())})

    # ── Training loop ──
    field_loss_fn = FieldLoss()
    train_log = []
    best_loss = float("inf")
    total_start = time.time()

    for epoch in range(args.epochs):
        t0 = time.time()

        # ── Train ──
        refiner.train()
        losses = defaultdict(float)
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"E{epoch}", leave=False)
        for batch in pbar:
            field_gt = batch["field"].to(device, non_blocking=True)  # (B, 6, 128, 128)

            # VQ encode → decode to get fragmented field
            with torch.no_grad():
                _, indices = vq.encode_to_code(field_gt)  # (B, 32, 32)
                field_vq = vq.decode_from_code(indices)   # (B, 6, 128, 128)

            # Refiner forward
            field_refined = refiner(field_vq)

            # Loss vs GT field
            ls = field_loss_fn(field_refined, field_gt)

            opt.zero_grad(set_to_none=True)
            ls["total"].backward()
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
            opt.step()

            for k, v in ls.items():
                losses[k] += v.item() if isinstance(v, torch.Tensor) else v
            n_batches += 1

        avg = {f"train_{k}": v / n_batches for k, v in losses.items()}
        if epoch >= 3:
            scheduler.step()

        # ── Validation ──
        refiner.eval()
        val_losses = defaultdict(float)
        val_n = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False):
                field_gt = batch["field"].to(device, non_blocking=True)
                _, indices = vq.encode_to_code(field_gt)
                field_vq = vq.decode_from_code(indices)
                field_refined = refiner(field_vq)
                ls = field_loss_fn(field_refined, field_gt)

                for k, v in ls.items():
                    val_losses[k] += v.item() if isinstance(v, torch.Tensor) else v
                val_n += 1

        avg_val = {f"val_{k}": v / max(val_n, 1) for k, v in val_losses.items()}

        # ── Log ──
        line = (f"E{epoch:2d} | "
                f"train_road_iou={avg.get('train_road_iou', 0):.4f} "
                f"train_total={avg.get('train_total', 0):.4f} | "
                f"val_road_iou={avg_val.get('val_road_iou', 0):.4f} "
                f"val_total={avg_val.get('val_total', 0):.4f} | "
                f"{time.time()-t0:.0f}s")
        print(line)

        log_entry = {"epoch": epoch, "time": time.time() - t0}
        log_entry.update(avg)
        log_entry.update(avg_val)
        train_log.append(log_entry)

        if use_wandb:
            wandb.log(log_entry)

        # ── Checkpoint ──
        val_iou = avg_val.get("val_road_iou", 0)
        if val_iou > best_loss:
            best_loss = val_iou
            torch.save({
                "epoch": epoch,
                "model_state_dict": refiner.state_dict(),
                "best_val_iou": best_loss,
                "opt_state_dict": opt.state_dict(),
            }, os.path.join(args.output_dir, "checkpoints", "best.pth"))

        torch.save({
            "epoch": epoch,
            "model_state_dict": refiner.state_dict(),
        }, os.path.join(args.output_dir, "checkpoints", "latest.pth"))

        with open(os.path.join(args.output_dir, "train_log.json"), "w") as f:
            json.dump(train_log, f, indent=2)

    # ── Summary ──
    wall = time.time() - total_start
    print(f"\nDone. {wall/60:.1f} min | Best val_road_iou={best_loss:.4f}")
    print(f"Checkpoint: {os.path.join(args.output_dir, 'checkpoints', 'best.pth')}")

    if use_wandb:
        run.finish()


if __name__ == "__main__":
    main()

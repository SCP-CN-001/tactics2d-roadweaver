"""
VQ-VAE Training: road field → discrete code map → field reconstruction.

Usage:
    cd /home/rowena/Documents/tactics2d-roadweaver
    PYTHONPATH=src:$PYTHONPATH conda run -n road-weaver \
        python -m src.spatial_conditional_skeleton_generator.train_vector_quantized_autoencoder
"""

import argparse, json, os, time
from collections import defaultdict
import wandb

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .skeleton_dataset import make_field_dataloader, RESOLUTION
from .vq_vae import VQVAE
from .losses import FieldLoss


class VQLoss(nn.Module):
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


def parse_args():
    p = argparse.ArgumentParser(description="VQ-VAE Training")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-val", type=int, default=200)
    p.add_argument("--output-dir", type=str, default="runtimes/vq_vae")
    p.add_argument("--wandb-project", type=str, default="roadweaver-ae",
                   help="Wandb project name. Empty string = no logging.")
    p.add_argument("--num-codes", type=int, default=512,
                   help="Number of VQ codebook entries")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    train_loader = make_field_dataloader("train", batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, limit_samples=args.limit_train, resolution=RESOLUTION)
    val_loader = make_field_dataloader("val", batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, limit_samples=args.limit_val, resolution=RESOLUTION)

    model = VQVAE(resolution=RESOLUTION, num_codes=args.num_codes).to(device)
    print(f"[VQ-VAE] Code map: {model.code_map_hw}×{model.code_map_hw}, codebook: {model.num_codes}×{model.embed_dim}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

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
                recon, indices, vq_info = model(target)
                ls = criterion(recon, target, vq_info)
            total = ls["total"]
            if not (torch.isnan(total) or torch.isinf(total)):
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
                for k, v in ls.items(): losses[k] += v.item() if isinstance(v, torch.Tensor) else v
                n_batches += 1
                ls.get("perplexity", 0)

        if epoch >= 5: scheduler.step()
        model.eval()
        val_ls, vn = defaultdict(float), 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False):
                target = batch["field"].to(device)
                with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                    recon, indices, vq_info = model(target)
                    ls = criterion(recon, target, vq_info)
                for k, v in ls.items(): val_ls[k] += v.item() if isinstance(v, torch.Tensor) else v
                vn += 1

        avg = {f"train_{k}": v/n_batches for k, v in losses.items()}
        avg.update({f"val_{k}": v/vn for k, v in val_ls.items()})
        avg["epoch"] = epoch
        avg["lr"] = optimizer.param_groups[0]["lr"]
        avg["time_s"] = round(time.time() - t0, 1)
        train_log.append(avg)

        print(f"E{epoch:2d} | val_iou={avg.get('val_road_iou',0):.4f} vq={avg.get('val_vq_loss',0):.4f} ppl={avg.get('val_perplexity',0):.1f} usage={avg.get('val_code_usage',0):.2%}")

        if args.wandb_project: wandb.log(avg)
        if avg.get("val_total", 1) < best_loss:
            best_loss = avg["val_total"]
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, os.path.join(args.output_dir, "checkpoints", "best.pth"))
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, os.path.join(args.output_dir, "checkpoints", "latest.pth"))
        with open(os.path.join(args.output_dir, "train_log.json"), "w") as f: json.dump(train_log, f, indent=2)

    if args.wandb_project: wandb.finish()
    print(f"Done. Best val_total={best_loss:.4f}")

if __name__ == "__main__":
    main()

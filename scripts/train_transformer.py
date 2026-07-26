#!/usr/bin/env python3
"""
Masked Code Transformer training.

Frozen VQ-VAE encodes road fields into discrete code maps,
a masked Transformer learns to predict masked tokens given condition.

Usage:
    python scripts/train_transformer.py                             # uses config_transformer.yaml
    python scripts/train_transformer.py --config my_config.yaml      # custom config
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

import wandb
from network_generator.backbone.config import CONFIG
from network_generator.backbone.dataset import make_field_dataloader
from network_generator.backbone.transformer import MaskedCodeModel
from network_generator.backbone.vq_vae import VQVAE

DEFAULT_CONFIG = "scripts/config_transformer.yaml"


class CachedCodeMapDataset(Dataset):
    """Load pre-computed code maps + conditions from a cached .npz file."""

    def __init__(self, cache_path: str):
        data = np.load(cache_path)
        self.code_maps = torch.from_numpy(data["code_maps"])
        self.conditions = torch.from_numpy(data["conditions"])
        print(f"  Loaded {len(self.code_maps)} samples from {cache_path}")

    def __len__(self):
        return len(self.code_maps)

    def __getitem__(self, idx):
        return {
            "code_tokens": self.code_maps[idx].long(),
            "condition": self.conditions[idx].float(),
        }


@torch.no_grad()
def extract_code_maps(vq, loader, device, code_map_hw=64):
    """Encode road fields into code maps + conditions."""
    all_codes, all_conds = [], []
    for batch in tqdm(loader, desc="Extracting"):
        field = batch["field"].to(device)
        style = batch["style_vector"]
        struct = batch["structural_priors"]
        cond = torch.cat([style, struct], dim=1)

        _, indices = vq.encode_to_code(field)
        indices = indices.cpu().reshape(-1, code_map_hw * code_map_hw)

        all_codes.append(indices)
        all_conds.append(cond)

    codes = torch.cat(all_codes, dim=0)
    conds = torch.cat(all_conds, dim=0)
    return codes.numpy(), conds.numpy()


def mask_tokens(code_tokens, mask_token_id, min_mask=0.10, max_mask=0.90):
    """Randomly mask tokens with per-batch variable ratio."""
    B, S = code_tokens.shape
    labels = code_tokens.clone()
    mask_ratio = torch.empty(1).uniform_(min_mask, max_mask).item()
    mask = torch.rand(B, S, device=code_tokens.device) < mask_ratio

    masked_tokens = code_tokens.clone()
    masked_tokens[mask] = mask_token_id
    labels[~mask] = -100

    return masked_tokens, labels, mask_ratio


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Masked Code Transformer training")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg = load_config(args.config)

    CONFIG.resolution = cfg.get("resolution", CONFIG.resolution)
    CONFIG.code_map_size = cfg.get("code_map_size", CONFIG.code_map_size)
    if data_dir := cfg.get("data_dir"):
        CONFIG.train_split_path = os.path.join(data_dir, "train.parquet")
        CONFIG.val_split_path = os.path.join(data_dir, "val.parquet")

    resolution = CONFIG.resolution
    code_map_hw = CONFIG.code_map_size
    seq_len = code_map_hw * code_map_hw
    num_codes = cfg.get("num_codes", 1024)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = cfg.get("output_dir", "runtimes/transformer_5km")
    cache_dir = cfg.get("cache_dir", "cache/masked_code_maps_5km")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    # ── Frozen VQ-VAE ──
    print("[Train] Loading frozen VQ-VAE...")
    vq = VQVAE(
        resolution=resolution,
        num_codes=num_codes,
        code_map_size=code_map_hw,
        embed_dim=cfg.get("embed_dim", 64),
    ).to(device)
    vq.eval()
    for p in vq.parameters():
        p.requires_grad = False
    state = torch.load(cfg["vq_checkpoint"], map_location=device, weights_only=True)
    vq.load_state_dict(state["model_state_dict"], strict=False)

    # ── Extract or load cached code maps ──
    train_cache = os.path.join(cache_dir, "train.npz")
    val_cache = os.path.join(cache_dir, "val.npz")

    if os.path.exists(train_cache) and os.path.exists(val_cache):
        print("[Train] Loading cached code maps from cache/...")
        train_ds = CachedCodeMapDataset(train_cache)
        val_ds = CachedCodeMapDataset(val_cache)
    else:
        print("[Train] Extracting code maps with VQ-VAE...")
        train_loader = make_field_dataloader(
            "train",
            batch_size=64,
            shuffle=False,
            num_workers=cfg.get("num_workers", 8),
            limit_samples=cfg.get("limit_train"),
            resolution=resolution,
        )
        val_loader = make_field_dataloader(
            "val",
            batch_size=64,
            shuffle=False,
            num_workers=cfg.get("num_workers", 8),
            limit_samples=cfg.get("limit_val", 200),
            resolution=resolution,
        )

        train_codes, train_conds = extract_code_maps(
            vq, train_loader, device, code_map_hw=code_map_hw
        )
        val_codes, val_conds = extract_code_maps(vq, val_loader, device, code_map_hw=code_map_hw)

        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(train_cache, code_maps=train_codes, conditions=train_conds)
        np.savez_compressed(val_cache, code_maps=val_codes, conditions=val_conds)
        print(f"  Cached to {train_cache} ({len(train_codes)} samples)")

        train_ds = CachedCodeMapDataset(train_cache)
        val_ds = CachedCodeMapDataset(val_cache)

    if cfg.get("balanced_sampling", False) and len(train_ds) > 0:
        # Compute class weights from style_vector in conditions
        classes = train_ds.conditions.argmax(dim=1)  # (N,) style class
        counts = Counter(classes.tolist())
        weights = [1.0 / counts[c] for c in classes.tolist()]
        sampler = WeightedRandomSampler(weights, len(train_ds), replacement=True)
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.get("batch_size", 128),
            sampler=sampler,
            num_workers=cfg.get("num_workers", 8),
            pin_memory=True,
        )
        print(f"  Balanced sampling enabled: {dict(counts)}")
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.get("batch_size", 128),
            shuffle=True,
            num_workers=cfg.get("num_workers", 8),
            pin_memory=True,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.get("batch_size", 128),
        shuffle=False,
        num_workers=cfg.get("num_workers", 8),
        pin_memory=True,
    )

    # ── Model ──
    model = MaskedCodeModel(
        vocab_size=num_codes + 1,
        d_model=cfg.get("d_model", 512),
        num_layers=cfg.get("num_layers", 6),
        nhead=cfg.get("nhead", 8),
        max_seq_len=seq_len,
    ).to(device)
    print(
        f"[Train] Transformer: {cfg.get('num_layers',6)} layers, "
        f"{cfg.get('nhead',8)} heads, {cfg.get('d_model',512)} dim"
    )
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Sequence length: {seq_len} ({code_map_hw}x{code_map_hw})")

    # ── Wandb ──
    wandb_project = cfg.get("wandb_project", "roadweaver-transformer").strip()
    if wandb_project:
        wandb.init(project=wandb_project, config=cfg, settings=wandb.Settings(console="off"))
        print(f"  [wandb] Logging to {wandb_project}")

    opt = optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.get("epochs", 50) - 5, eta_min=cfg.get("lr", 1e-3) * 0.01
    )

    train_log = []
    best_val_ce = float("inf")
    total_start = time.time()

    for epoch in range(cfg.get("epochs", 50)):
        t0 = time.time()

        # ── Train ──
        model.train()
        losses = defaultdict(float)
        n_tokens = 0

        for batch in tqdm(train_loader, desc=f"E{epoch}", leave=False):
            codes = batch["code_tokens"].to(device, non_blocking=True)
            cond = batch["condition"].to(device, non_blocking=True)

            masked_codes, labels, mask_ratio = mask_tokens(codes, mask_token_id=model.mask_token_id)

            logits = model(masked_codes, cond)
            loss = F.cross_entropy(
                logits.view(-1, model.vocab_size), labels.view(-1), ignore_index=-100
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            losses["ce"] += loss.item() * (labels != -100).sum().item()
            n_tokens += (labels != -100).sum().item()

        train_ce = losses["ce"] / max(n_tokens, 1)
        if epoch >= 5:
            scheduler.step()

        # ── Validation ──
        model.eval()
        val_ce = 0.0
        val_tokens = 0
        val_acc = 0.0
        val_n = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False):
                codes = batch["code_tokens"].to(device, non_blocking=True)
                cond = batch["condition"].to(device, non_blocking=True)

                masked_codes, labels, _ = mask_tokens(codes, mask_token_id=model.mask_token_id)

                logits = model(masked_codes, cond)
                loss = F.cross_entropy(
                    logits.view(-1, model.vocab_size), labels.view(-1), ignore_index=-100
                )

                val_ce += loss.item() * (labels != -100).sum().item()
                val_tokens += (labels != -100).sum().item()

                preds = logits.argmax(dim=-1)
                correct = (preds == codes) & (labels != -100)
                val_acc += correct.sum().item()
                val_n += (labels != -100).sum().item()

        val_ce /= max(val_tokens, 1)
        val_acc /= max(val_n, 1)

        # ── Sample quality ──
        sample_iou = 0.0
        if epoch % 5 == 0 or epoch == cfg.get("epochs", 50) - 1:
            with torch.no_grad():
                cond_sample = cond[:4]
                sampled_codes = model.sample(cond_sample, num_steps=8)
                sampled_field = vq.decode_from_code(sampled_codes)
                gt_code_map = codes[:4].reshape(-1, code_map_hw, code_map_hw)
                gt_field = vq.decode_from_code(gt_code_map)
                for b in range(4):
                    pred_bin = torch.sigmoid(sampled_field[b, 0]) > 0.5
                    gt_bin = torch.sigmoid(gt_field[b, 0]) > 0.5
                    tp = (pred_bin & gt_bin).sum()
                    fp = (pred_bin & ~gt_bin).sum()
                    fn = (~pred_bin & gt_bin).sum()
                    sample_iou += (tp / (tp + fp + fn + 1e-8)).item()
                sample_iou /= 4

        print(
            f"E{epoch:2d} | train_ce={train_ce:.4f} val_ce={val_ce:.4f} "
            f"val_acc={val_acc:.4f} sample_iou={sample_iou:.4f} | "
            f"{time.time()-t0:.0f}s"
        )

        metrics = {
            "epoch": epoch,
            "train_ce": train_ce,
            "val_ce": val_ce,
            "val_acc": val_acc,
            "sample_iou": sample_iou,
            "lr": opt.param_groups[0]["lr"],
        }
        train_log.append(metrics)
        if wandb_project:
            wandb.log(metrics)

        if val_ce < best_val_ce:
            best_val_ce = val_ce
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_ce": best_val_ce,
                },
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
    wall = time.time() - total_start
    print(f"\nDone. {wall/60:.1f} min | Best val_ce={best_val_ce:.4f}")


if __name__ == "__main__":
    main()

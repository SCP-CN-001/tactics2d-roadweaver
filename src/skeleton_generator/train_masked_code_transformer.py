"""
Masked Code Transformer Training.

Frozen VQ-VAE. Trains a Transformer to predict masked code tokens
given condition + visible tokens.

Usage:
    # Precompute code maps first (if not cached)
    # Then train:
    PYTHONPATH=src:$PYTHONPATH conda run -n road-weaver \
        python -m src.skeleton_generator.train_masked_code_transformer
"""

import argparse
import json
import os
import time
from collections import defaultdict

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .vq_vae import VQVAE
from .masked_transformer import MaskedCodeModel
from .skeleton_dataset import make_field_dataloader, RESOLUTION


class CachedCodeMapDataset(Dataset):
    """Load pre-computed code maps + conditions from a cached file."""

    def __init__(self, cache_path: str):
        import numpy as np
        data = np.load(cache_path)
        self.code_maps = torch.from_numpy(data["code_maps"])  # (N, 1024)
        self.conditions = torch.from_numpy(data["conditions"])  # (N, 11)
        print(f"  Loaded {len(self.code_maps)} samples from {cache_path}")

    def __len__(self):
        return len(self.code_maps)

    def __getitem__(self, idx):
        return {
            "code_tokens": self.code_maps[idx].long(),
            "condition": self.conditions[idx].float(),
        }


@torch.no_grad()
def extract_code_maps(vq, loader, device, max_samples=None):
    """Extract code maps and conditions from a dataloader."""
    all_codes, all_conds = [], []
    count = 0
    for batch in tqdm(loader, desc="Extracting"):
        field = batch["field"].to(device)
        style = batch["style_vector"]
        struct = batch["structural_priors"]
        cond = torch.cat([style, struct], dim=1)

        _, indices = vq.encode_to_code(field)  # (B, 32, 32)
        indices = indices.cpu().reshape(-1, 1024)  # (B, 1024)

        all_codes.append(indices)
        all_conds.append(cond)
        count += indices.shape[0]
        if max_samples and count >= max_samples:
            break

    codes = torch.cat(all_codes, dim=0)
    conds = torch.cat(all_conds, dim=0)
    return codes.numpy(), conds.numpy()


def mask_tokens(code_tokens, mask_token_id=MaskedCodeModel.MASK_TOKEN_ID,
                min_mask=0.10, max_mask=0.90):
    """Randomly mask tokens with per-batch variable ratio.

    Each batch gets a random mask ratio sampled uniformly from [min_mask, max_mask].
    This ensures the model sees both lightly-masked and heavily-masked sequences,
    making iterative decoding from full mask possible.

    Returns:
        masked_tokens: code_tokens with MASK at selected positions
        mask: boolean mask of masked positions
        labels: original code IDs (set to -100 for non-masked)
    """
    B, S = code_tokens.shape
    labels = code_tokens.clone()

    # Per-batch variable mask ratio
    mask_ratio = torch.empty(1).uniform_(min_mask, max_mask).item()

    prob = torch.rand(B, S, device=code_tokens.device)
    mask = prob < mask_ratio

    masked_tokens = code_tokens.clone()
    masked_tokens[mask] = mask_token_id

    labels[~mask] = -100

    return masked_tokens, mask, labels, mask_ratio


def parse_args():
    p = argparse.ArgumentParser(description="Masked Code Transformer Training")
    p.add_argument("--vq-checkpoint", default="checkpoints/skeleton_generator/vq_vae.pth")
    p.add_argument("--cache-dir", default="cache/masked_code_maps",
                   help="Directory for precomputed code map cache")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", default="runtimes/masked_code_transformer")
    p.add_argument("--limit-train", type=int, default=0,
                   help="0 = use all")
    p.add_argument("--limit-val", type=int, default=200)
    p.add_argument("--mask-prob", type=float, default=0.15)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--wandb-project", type=str, default="roadweaver-ae",
                   help="Wandb project name. Empty string = no logging.")
    p.add_argument("--num-codes", type=int, default=512,
                   help="Number of codes (vocab_size = num_codes + 1)")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.limit_train = args.limit_train if args.limit_train > 0 else None

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    # ── Frozen VQVAE ──
    print("[Train] Loading frozen VQVAE...")
    vq = VQVAE(resolution=RESOLUTION, num_codes=args.num_codes).to(device)
    vq.eval()
    for p in vq.parameters():
        p.requires_grad = False
    state = torch.load(args.vq_checkpoint, map_location=device, weights_only=True)
    vq_state = vq.state_dict()
    for k, v in state["model_state_dict"].items():
        if k in vq_state and v.shape == vq_state[k].shape:
            vq_state[k] = v
    vq.load_state_dict(vq_state, strict=False)

    # ── Extract or load cached code maps ──
    train_cache = os.path.join(args.cache_dir, "train.npz")
    val_cache = os.path.join(args.cache_dir, "val.npz")

    if os.path.exists(train_cache) and os.path.exists(val_cache):
        print("[Train] Loading cached code maps...")
        train_ds = CachedCodeMapDataset(train_cache)
        val_ds = CachedCodeMapDataset(val_cache)
    else:
        print("[Train] Extracting code maps (this may take a while)...")
        train_loader = make_field_dataloader(
            "train", batch_size=64, shuffle=False,
            num_workers=args.num_workers, limit_samples=args.limit_train,
            resolution=RESOLUTION,
        )
        val_loader = make_field_dataloader(
            "val", batch_size=64, shuffle=False,
            num_workers=args.num_workers, limit_samples=args.limit_val,
            resolution=RESOLUTION,
        )

        train_codes, train_conds = extract_code_maps(vq, train_loader, device,
                                                     max_samples=args.limit_train)
        val_codes, val_conds = extract_code_maps(vq, val_loader, device,
                                                 max_samples=args.limit_val)

        os.makedirs(args.cache_dir, exist_ok=True)
        import numpy as np
        np.savez_compressed(train_cache, code_maps=train_codes, conditions=train_conds)
        np.savez_compressed(val_cache, code_maps=val_codes, conditions=val_conds)
        print(f"  Cached to {train_cache}, {val_cache}")

        train_ds = CachedCodeMapDataset(train_cache)
        val_ds = CachedCodeMapDataset(val_cache)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # ── Model ──
    model = MaskedCodeModel(vocab_size=args.num_codes + 1,
                            d_model=args.d_model, num_layers=args.num_layers,
                            nhead=args.num_heads).to(device)
    print(f"[Train] Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Transformer: {args.num_layers} layers, {args.num_heads} heads, {args.d_model} dim")

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs - 5, eta_min=args.lr * 0.01)

    train_log = []
    best_val_ce = float("inf")
    total_start = time.time()

    for epoch in range(args.epochs):
        t0 = time.time()

        # ── Train ──
        model.train()
        losses = defaultdict(float)
        n_tokens = 0

        pbar = tqdm(train_loader, desc=f"E{epoch}", leave=False)
        for batch in pbar:
            codes = batch["code_tokens"].to(device, non_blocking=True)  # (B, 1024)
            cond = batch["condition"].to(device, non_blocking=True)     # (B, 11)

            # Random mask (variable ratio)
            masked_codes, mask, labels, mask_ratio = mask_tokens(
                codes, mask_token_id=MaskedCodeModel.MASK_TOKEN_ID)

            # Forward
            logits = model(masked_codes, mask, cond)

            # Loss on masked positions only
            loss = F.cross_entropy(
                logits.view(-1, model.vocab_size), labels.view(-1), ignore_index=-100)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            losses["ce"] += loss.item() * mask.sum().item()
            n_tokens += mask.sum().item()

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

                masked_codes, mask, labels, _ = mask_tokens(
                    codes, mask_token_id=MaskedCodeModel.MASK_TOKEN_ID)

                logits = model(masked_codes, mask, cond)
                loss = F.cross_entropy(
                    logits.view(-1, model.vocab_size), labels.view(-1), ignore_index=-100)

                val_ce += loss.item() * mask.sum().item()
                val_tokens += mask.sum().item()

                preds = logits.argmax(dim=-1)
                correct = (preds == codes) & mask
                val_acc += correct.sum().item()
                val_n += mask.sum().item()

        val_ce /= max(val_tokens, 1)
        val_acc /= max(val_n, 1)

        # ── Sample quality evaluation ──
        sample_iou = 0.0
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            with torch.no_grad():
                # Take first 4 val conditions, sample code maps
                cond_sample = cond[:4].to(device)
                sampled_codes = model.sample_code_map(cond_sample, num_steps=8)
                sampled_field = vq.decode_from_code(sampled_codes)
                # Compare to GT for reference
                gt_code_map = codes[:4].reshape(-1, 32, 32).to(device)
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

        train_log.append({
            "epoch": epoch, "train_ce": train_ce, "val_ce": val_ce,
            "val_acc": val_acc, "sample_iou": sample_iou,
        })

        if val_ce < best_val_ce:
            best_val_ce = val_ce
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "best_val_ce": best_val_ce},
                       os.path.join(args.output_dir, "checkpoints", "best.pth"))

        torch.save({"epoch": epoch, "model_state_dict": model.state_dict()},
                   os.path.join(args.output_dir, "checkpoints", "latest.pth"))

        with open(os.path.join(args.output_dir, "train_log.json"), "w") as f:
            json.dump(train_log, f, indent=2)

    wall = time.time() - total_start
    print(f"\nDone. {wall/60:.1f} min | Best val_ce={best_val_ce:.4f}")


if __name__ == "__main__":
    main()

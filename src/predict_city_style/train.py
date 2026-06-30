#!/usr/bin/env python3
"""
Train the city style encoder on CRHD images.

Usage:
    python -m src.predict_city_style.train \
        --manifest data/crhd/manifest.json \
        --output-dir runtime/train \
        --style-dim 6 --epochs 3 --batch-size 8 --device cpu
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Ensure project root is on sys.path so src.* imports work
_PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.predict_city_style.crhd_dataset import CRHDDataset
from src.predict_city_style.style_encoder import StyleEncoder
from src.utils.io_utils import save_json


def parse_args():
    parser = argparse.ArgumentParser(description='Train style encoder')
    parser.add_argument('--manifest', type=str, required=True,
                        help='Path to training manifest JSON')
    parser.add_argument('--output-dir', type=str,
                        default='runtime/train',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--style-dim', type=int, default=6)
    parser.add_argument('--backbone', type=str, default='resnet34')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--backbone-weights', type=str, default=None,
                        help='Local safetensors/.pth file for backbone init')
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--loss-type', type=str, default='soft_label',
                        choices=['soft_label', 'hard_label'])
    return parser.parse_args()


def mse_loss(pred, target):
    return nn.functional.mse_loss(pred, target)


def main():
    args = parse_args()

    # Validate manifest
    with open(args.manifest) as f:
        manifest_data = json.load(f)
    has_labels = any(
        sum(entry.get('label', [0.0])) > 0
        for entry in manifest_data[:20]
    )
    if not has_labels:
        print('[ERROR] Manifest has no valid labels.')
        print('  Each entry needs "label": [p1, p2, ..., pN]')
        print('  Use --loss-type soft_label for probability vectors.')
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    save_json(vars(args), os.path.join(args.output_dir, 'config.json'))

    # Dataset
    print(f'[train] Loading dataset: {args.manifest}')
    dataset = CRHDDataset(
        manifest=args.manifest,
        image_size=(args.image_size, args.image_size),
    )
    print(f'[train] Total samples: {len(dataset)}')

    # Split
    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Model
    device = torch.device(args.device)
    model = StyleEncoder(
        style_dim=args.style_dim,
        backbone_name=args.backbone,
        pretrained=args.pretrained,
        backbone_weights=args.backbone_weights,
    )
    model.to(device)
    print(f'[train] Model: {args.backbone}, style_dim={args.style_dim}')

    # Loss & optimizer
    criterion = mse_loss if args.loss_type == 'soft_label' else nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    # Train loop
    best_val_loss = float('inf')
    log = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            if args.loss_type == 'hard_label':
                loss = criterion(outputs, labels.argmax(dim=1))
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                if args.loss_type == 'hard_label':
                    loss = criterion(outputs, labels.argmax(dim=1))
                else:
                    loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f'[train] Epoch {epoch}/{args.epochs} | '
              f'train: {avg_train:.6f} | val: {avg_val:.6f} | '
              f'{time.time()-t0:.1f}s')
        log.append({
            'epoch': epoch, 'train_loss': round(avg_train, 6),
            'val_loss': round(avg_val, 6),
        })

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': avg_val,
            }, os.path.join(args.output_dir, 'best.pth'))
            print(f'  -> Saved best.pth (val_loss={avg_val:.6f})')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }, os.path.join(args.output_dir, 'latest.pth'))
        scheduler.step(avg_val)

    save_json(log, os.path.join(args.output_dir, 'train_log.json'))
    print(f'[train] Done! Best val_loss: {best_val_loss:.6f}')


if __name__ == '__main__':
    main()

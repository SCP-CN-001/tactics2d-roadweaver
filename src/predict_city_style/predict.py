#!/usr/bin/env python3
"""
Predict city styles from CRHD images using a trained style encoder.

Usage:
    python -m src.predict_city_style.predict \
        --input data/crhd \
        --checkpoint checkpoints/style_encoder/best.pth \
        --output outputs/predictions.json \
        --device cpu
"""

import argparse
import os
import sys

import numpy as np
import torch

# Ensure project root is on sys.path so src.* imports work
_PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.predict_city_style.style_encoder import build_encoder
from src.predict_city_style.crhd_dataset import load_image, to_tensor
from src.utils.io_utils import find_images, save_json

# Label names matching the original project
PATTERN_NAMES = [
    'Gridiron', 'Linear', 'No pattern',
    'Organic', 'Radial', 'Tributary',
]

# Model architecture constants (must match training config)
STYLE_DIM = 6
IMAGE_SIZE = 224


def parse_args():
    parser = argparse.ArgumentParser(description='Predict city style vectors')
    parser.add_argument('--input', type=str, required=True,
                        help='Image file or directory')
    parser.add_argument('--checkpoint', type=str, default='none',
                        help='Model checkpoint path ("none" for random init)')
    parser.add_argument('--output', type=str,
                        default='runtime/predictions.json',
                        help='Output JSON path')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cpu')
    return parser.parse_args()


def resolve_checkpoint(checkpoint_arg: str) -> str:
    """Resolve checkpoint path: if it's a directory, auto-detect the .pth file."""
    if checkpoint_arg.lower() in ('none', ''):
        return checkpoint_arg

    if os.path.isfile(checkpoint_arg):
        return checkpoint_arg

    if os.path.isdir(checkpoint_arg):
        # Priority order of checkpoint files
        candidates = ['best.pth', 'style_encoder_best.pth', 'latest.pth',
                      'checkpoint.pth']
        available = [f for f in os.listdir(checkpoint_arg) if f.endswith('.pth')]
        if not available:
            print(f'[error] No .pth files found in checkpoint directory: '
                  f'{checkpoint_arg}', file=sys.stderr)
            print(f'  Directory contents: {os.listdir(checkpoint_arg)}',
                  file=sys.stderr)
            sys.exit(1)
        for name in candidates:
            path = os.path.join(checkpoint_arg, name)
            if os.path.isfile(path):
                return path
        # Fallback: use the first .pth found (sorted alphabetically)
        fallback = os.path.join(checkpoint_arg, sorted(available)[0])
        print(f'[warn] No preferred checkpoint found, using: {fallback}')
        return fallback

    # Path does not exist at all
    print(f'[error] Checkpoint path does not exist: {checkpoint_arg}',
          file=sys.stderr)
    sys.exit(1)


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Resolve checkpoint path (handle file or directory)
    ckpt_path = resolve_checkpoint(args.checkpoint)
    checkpoint_loaded = ckpt_path.lower() not in ('none', '')
    print(f'[predict] Checkpoint: {ckpt_path if checkpoint_loaded else "none (random init)"}')

    model = build_encoder(
        style_dim=STYLE_DIM,
        checkpoint_path=ckpt_path,
        device=device,
    )

    image_paths = find_images(args.input)
    if not image_paths:
        print(f'[error] No images found at: {args.input}', file=sys.stderr)
        sys.exit(1)

    print(f'[predict] Found {len(image_paths)} image(s)')
    results = []

    for i in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[i:i + args.batch_size]
        batch_tensors = []
        for p in batch_paths:
            img = load_image(p, size=(IMAGE_SIZE, IMAGE_SIZE))
            batch_tensors.append(torch.from_numpy(to_tensor(img)))
        batch = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            vectors = model(batch)

        for j, p in enumerate(batch_paths):
            vec = vectors[j].cpu().numpy().tolist()
            vec_arr = np.array(vec)
            top_idx = int(vec_arr.argmax())
            results.append({
                'image_path': p,
                'style_vector': vec,
                'style_dim': STYLE_DIM,
                'top_pattern': top_idx,
                'top_pattern_name': (
                    PATTERN_NAMES[top_idx]
                    if top_idx < len(PATTERN_NAMES) else 'Unknown'
                ),
                'confidence': round(float(vec_arr[top_idx]), 6),
                'checkpoint_loaded': checkpoint_loaded,
            })

    save_json(results, args.output)
    print(f'[predict] Results saved to: {args.output}')
    print(f'[predict] Processed {len(results)} image(s). Done!')


if __name__ == '__main__':
    main()

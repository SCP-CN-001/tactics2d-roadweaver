#!/usr/bin/env python3
"""
Predict city style vectors from CRHD images using a trained style encoder.

Usage:
    python -m style_predictor.predict \
        --input data/crhd_2km \
        --checkpoint checkpoints/style_encoder/best.pth \
        --output data/crhd_2km_style_predictions.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .dataset import load_image, to_tensor
from .encoder import build_encoder

PATTERN_NAMES = ["Gridiron", "Linear", "No pattern", "Organic", "Radial", "Tributary"]
STYLE_DIM = 6
IMAGE_SIZE = 224


def _find_images(path: str | list[str], exts=(".png", ".jpg", ".jpeg")) -> list[str]:
    """Find image files from a path (file, dir, or list)."""
    if isinstance(path, list):
        return sorted(p for p in path if p.lower().endswith(exts))
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        return sorted(os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(exts))
    raise FileNotFoundError(f"Path not found: {path}")


def _save_json(data, path: str):
    """Save data as pretty-printed JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Predict city style vectors")
    parser.add_argument("--input", type=str, required=True, help="Image file or directory")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/style_predictor/best.pth",
        help='Model checkpoint path ("none" for random init)',
    )
    parser.add_argument(
        "--output", type=str, default="runtime/predictions.json", help="Output JSON path"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    model = build_encoder(style_dim=STYLE_DIM, checkpoint_path=args.checkpoint, device=device)
    print(f"[predict] Checkpoint: {args.checkpoint}")

    image_paths = _find_images(args.input)
    if not image_paths:
        print(f"[error] No images found at: {args.input}", file=__import__("sys").stderr)
        return

    print(f"[predict] Found {len(image_paths)} image(s)")
    results = []

    for i in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[i : i + args.batch_size]
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
            results.append(
                {
                    "image_path": p,
                    "style_vector": vec,
                    "style_dim": STYLE_DIM,
                    "top_pattern": top_idx,
                    "top_pattern_name": (
                        PATTERN_NAMES[top_idx] if top_idx < len(PATTERN_NAMES) else "Unknown"
                    ),
                    "confidence": round(float(vec_arr[top_idx]), 6),
                }
            )

    _save_json(results, args.output)
    print(f"[predict] Results saved to: {args.output}")
    print(f"[predict] Processed {len(results)} image(s). Done!")


if __name__ == "__main__":
    main()

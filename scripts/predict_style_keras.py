#!/usr/bin/env python3
"""
Classify CRHD images using the Chen et al. 2024 ResNet34 checkpoint.

Usage:
    CUDA_VISIBLE_DEVICES="" python scripts/predict_style_keras.py \
        --input data/crhd_2km \
        --output data/crhd_2km_style_keras.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob

import cv2
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


CKPT_DIR = "/home/rowena/Documents/Global-road-network-patterns"
os.chdir(CKPT_DIR)
sys.path.insert(0, os.path.join(CKPT_DIR, "train"))

# Ensure dataset directory exists for config.py (relative to CWD after chdir)
os.makedirs(os.path.join(CKPT_DIR, "dataset", "tmp"), exist_ok=True)

from Build_model import Build_model
from config import config

PATTERN_NAMES = ["Gridiron", "Linear", "Nopattern", "Organic", "Radial", "Tributary"]


def load_model():
    config.classNumber = 6
    builder = Build_model(config)
    model = builder.build_model()
    model.load_weights(os.path.join(CKPT_DIR, "ResNet-34-6class-aug5.h5"))
    return model


def preprocess(path: str) -> np.ndarray:
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thre = cv2.threshold(gray, 200, 255, cv2.THRESH_TRUNC)[1]
    mask = np.where(thre > 190)
    img[mask] = 255
    img = cv2.resize(img, (224, 224))
    return img.astype(np.float32) / 255.0


def main():
    parser = argparse.ArgumentParser(description="Classify CRHD with Keras model")
    parser.add_argument("--input", type=str, required=True, help="CRHD image directory")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    model = load_model()
    print(f"[predict] Model loaded from {CKPT_DIR}/ResNet-34-6class-aug5.h5")

    paths = sorted(glob(os.path.join(args.input, "*.png")))
    print(f"[predict] Found {len(paths)} images")

    results = []
    for i in range(0, len(paths), args.batch_size):
        batch_paths = paths[i : i + args.batch_size]
        batch = np.array([preprocess(p) for p in batch_paths])
        preds = model.predict(batch, verbose=0)

        for j, p in enumerate(batch_paths):
            vec = preds[j].tolist()
            top_idx = int(np.argmax(preds[j]))
            results.append(
                {
                    "image_path": p,
                    "style_vector": vec,
                    "style_dim": 6,
                    "top_pattern": top_idx,
                    "top_pattern_name": PATTERN_NAMES[top_idx],
                    "confidence": round(float(preds[j][top_idx]), 6),
                }
            )

        if (i // args.batch_size) % 10 == 0:
            print(f"  {i + len(batch_paths)}/{len(paths)}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[predict] Saved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()

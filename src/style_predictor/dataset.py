"""CRHD image dataset with soft/hard label support."""

from __future__ import annotations

import json

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def load_image(path: str, size: tuple | None = None) -> np.ndarray:
    """Load an RGB image, optionally resize, normalise to [0, 1]."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    return img.astype(np.float32) / 255.0


def to_tensor(img: np.ndarray) -> np.ndarray:
    """Convert HWC to CHW."""
    return np.transpose(img, (2, 0, 1))


class CRHDDataset(Dataset):
    """Dataset for CRHD images with soft labels.

    Manifest format (JSON list):
        [{"image_path": "...", "label": [p1, p2, ...]}, ...]
    """

    def __init__(self, manifest: str, image_size: tuple[int, int] = (224, 224)):
        if isinstance(manifest, str):
            with open(manifest) as f:
                data = json.load(f)
        else:
            data = manifest

        self.samples = []
        for entry in data:
            img_path = entry.get("image_path", "")
            label = np.array(entry.get("label", [0.0] * 6), dtype=np.float32)
            self.samples.append((img_path, label))

        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.samples[idx]
        img = load_image(img_path, size=self.image_size)
        return torch.from_numpy(to_tensor(img)), torch.from_numpy(label)

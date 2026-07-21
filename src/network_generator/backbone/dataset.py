"""
Field dataset — renders GT skeleton graphs as raster field tensors.

Each sample calls graph_to_raster() on the fly and returns:
  - condition: style_vector, structural_priors, map_size
  - field: (6, H, W) tensor:
      [0] road_prob (binary centerline)
      [1] sin_2theta, [2] cos_2theta
      [3] junction_hm, [4] endpoint_hm
      [5] soft_distance (auxiliary)
"""

from __future__ import annotations

import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from network_generator.topology.graph_to_raster import graph_to_raster

from .config import CONFIG


class SkeletonFieldDataset(Dataset):
    """Loads skeleton graphs from Parquet and renders them as raster fields."""

    def __init__(
        self, split: str = "train", limit_samples: int | None = None, resolution: int | None = None
    ):
        super().__init__()
        path = CONFIG.train_split_path if split == "train" else CONFIG.val_split_path
        import pandas as pd

        self.df = pd.read_parquet(path)

        mask = (self.df["skeleton_node_count"] >= 2) & (
            self.df["skeleton_node_count"] <= CONFIG.max_nodes_in_training
        )
        self.df = self.df[mask].reset_index(drop=True)
        if limit_samples is not None and limit_samples < len(self.df):
            self.df = self.df.iloc[:limit_samples].reset_index(drop=True)

        self._style_cols = [f"style_vector_{i}" for i in range(CONFIG.style_dim)]
        self._structural_cols = [
            "road_density_km_per_km2",
            "gridness_score",
            "radialness_score",
            "organic_score",
            "bearing_entropy",
        ]
        self.resolution = resolution or CONFIG.resolution

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # style_vector may be absent (no-style splits)
        if all(c in self.df.columns for c in self._style_cols):
            style_vector = torch.tensor([row[c] for c in self._style_cols], dtype=torch.float)
        else:
            style_vector = torch.zeros(CONFIG.style_dim, dtype=torch.float)
        structural_priors = torch.tensor([row[c] for c in self._structural_cols], dtype=torch.float)
        map_size = torch.tensor([CONFIG.map_size_scale, CONFIG.map_size_scale], dtype=torch.float)

        # Parse graph
        sg = json.loads(row["skeleton_graph_json"])
        raw_nodes: list[dict] = sg.get("nodes", []) or []
        raw_edges: list[dict] = sg.get("edges", []) or []

        if not raw_nodes:
            return self.__getitem__((idx + 1) % len(self.df))

        # Build arrays
        id_to_idx = {n["id"]: i for i, n in enumerate(raw_nodes)}
        N = len(raw_nodes)
        coords = np.zeros((N, 2), dtype=np.float32)
        edge_list = []

        for n in raw_nodes:
            i = id_to_idx[n["id"]]
            coords[i] = [float(n.get("x", 0)), float(n.get("y", 0))]

        for e in raw_edges:
            src = id_to_idx.get(e.get("source", -1))
            tgt = id_to_idx.get(e.get("target", -1))
            if src is None or tgt is None or src == tgt:
                continue
            edge_list.append((src, tgt))

        edge_index = (
            np.array(edge_list, dtype=np.int64) if edge_list else np.zeros((0, 2), dtype=np.int64)
        )

        # Render field
        field = graph_to_raster(
            coords, edge_index, resolution=self.resolution, binary_centerline=True
        )

        field_tensor = torch.stack(
            [
                torch.from_numpy(field["road_prob"]),
                torch.from_numpy(field["sin_2theta"]),
                torch.from_numpy(field["cos_2theta"]),
                torch.from_numpy(field["junction_hm"]),
                torch.from_numpy(field["endpoint_hm"]),
                torch.from_numpy(field["soft_distance"]),
            ],
            dim=0,
        )

        return {
            "style_vector": style_vector,
            "structural_priors": structural_priors,
            "map_size": map_size,
            "field": field_tensor,
        }


def collate_fields(batch: list[dict]) -> dict:
    """Collate field samples (all same size, just stack)."""
    out = {}
    for k in batch[0].keys():
        if isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            out[k] = [b[k] for b in batch]
    return out


def make_field_dataloader(
    split: str = "train",
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 2,
    limit_samples: int | None = None,
    resolution: int | None = None,
) -> DataLoader:
    dataset = SkeletonFieldDataset(split=split, limit_samples=limit_samples, resolution=resolution)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fields,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

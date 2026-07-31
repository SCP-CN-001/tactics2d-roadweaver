"""Skeleton field dataset and dataloader."""

from __future__ import annotations

import json
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from network_generator.topology.graph_to_raster import graph_to_raster

from .config import CONFIG


class SkeletonFieldDataset(Dataset):
    """Loads skeleton graphs from Parquet and renders them as raster fields.

    If *cache_fields* is True, rendered fields are cached to disk at
    ``cache/dataset_fields/{resolution}/{split}/`` for fast reloading.
    """

    def __init__(
        self,
        split: str = "train",
        limit_samples: int | None = None,
        resolution: int | None = None,
        cache_fields: bool = True,
    ):
        super().__init__()
        path = CONFIG.train_split_path if split == "train" else CONFIG.val_split_path

        self.df = pd.read_parquet(path)
        self.split = split
        self.resolution = resolution or CONFIG.resolution

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

        # ── Disk cache ────────────────────────────────────────────────
        self._fields_cache = None  # (N, 6, H, W) numpy array
        self._conditions_cache = None  # (N, 11) numpy array
        self._map_sizes_cache = None  # (N, 2) numpy array

        if cache_fields:
            self._load_or_build_cache()

    def _cache_dir(self) -> Path:
        """Path like ``cache/dataset_fields/{resolution}/{split}/``."""
        return Path("cache") / "dataset_fields" / str(self.resolution) / self.split

    def _load_or_build_cache(self):
        cache_dir = self._cache_dir()
        cache_path = cache_dir / "fields.npy"
        cond_path = cache_dir / "conditions.npy"
        map_path = cache_dir / "map_sizes.npy"

        if cache_path.exists() and cond_path.exists():
            self._fields_cache = np.load(str(cache_path), mmap_mode="r")
            self._conditions_cache = np.load(str(cond_path), mmap_mode="r")
            self._map_sizes_cache = np.load(str(map_path), mmap_mode="r")
            print(f"  [Cache] Loaded {len(self)} fields from {cache_dir}")
            return

        print(f"  [Cache] Building field cache at {cache_dir} ...")
        os.makedirs(cache_dir, exist_ok=True)

        N = len(self)
        H = W = self.resolution
        fields = np.zeros((N, 6, H, W), dtype=np.float32)
        conditions = np.zeros((N, 11), dtype=np.float32)
        map_sizes = np.zeros((N, 2), dtype=np.float32)

        has_style = all(c in self.df.columns for c in self._style_cols)
        rows = []
        for idx in range(N):
            row = self.df.iloc[idx]
            style_vec = (
                [row[c] for c in self._style_cols] if has_style else [0.0] * CONFIG.style_dim
            )
            struct_priors = [row[c] for c in self._structural_cols]
            skeleton_json = row["skeleton_graph_json"]
            rows.append((idx, style_vec, struct_priors, skeleton_json))

        n_workers = min(os.cpu_count() or 4, 32)
        render_fn = partial(
            _render_one_field,
            resolution=self.resolution,
            style_dim=CONFIG.style_dim,
            map_size_scale=CONFIG.map_size_scale,
        )
        with Pool(n_workers) as pool:
            for result in tqdm(
                pool.imap_unordered(render_fn, rows), total=N, desc=f"Caching {self.split}"
            ):
                idx, cond, map_sz, fld = result
                conditions[idx] = cond
                map_sizes[idx] = map_sz
                if fld is not None:
                    fields[idx] = fld

        np.save(str(cache_path), fields)
        np.save(str(cond_path), conditions)
        np.save(str(map_path), map_sizes)
        print(f"  [Cache] Saved {N} fields to {cache_dir}")
        self._fields_cache = np.load(str(cache_path), mmap_mode="r")
        self._conditions_cache = np.load(str(cond_path), mmap_mode="r")
        self._map_sizes_cache = np.load(str(map_path), mmap_mode="r")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        if self._fields_cache is not None:
            return {
                "style_vector": torch.from_numpy(self._conditions_cache[idx, :6]),
                "structural_priors": torch.from_numpy(self._conditions_cache[idx, 6:11]),
                "map_size": torch.from_numpy(self._map_sizes_cache[idx]),
                "field": torch.from_numpy(self._fields_cache[idx]),
            }

        # Fallback: render on the fly (no cache)
        row = self.df.iloc[idx]

        if all(c in self.df.columns for c in self._style_cols):
            style_vector = torch.tensor([row[c] for c in self._style_cols], dtype=torch.float)
        else:
            style_vector = torch.zeros(CONFIG.style_dim, dtype=torch.float)
        structural_priors = torch.tensor([row[c] for c in self._structural_cols], dtype=torch.float)
        map_size = torch.tensor([CONFIG.map_size_scale, CONFIG.map_size_scale], dtype=torch.float)

        sg = json.loads(row["skeleton_graph_json"])
        raw_nodes: list[dict] = sg.get("nodes", []) or []
        raw_edges: list[dict] = sg.get("edges", []) or []
        if not raw_nodes:
            return self.__getitem__((idx + 1) % len(self.df))

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


def _render_one_field(args, resolution=256, style_dim=6, map_size_scale=5000.0):
    """Module-level function for multiprocessing: render one field from JSON."""
    idx, sv, sp, sg_json = args
    cond = np.array(sv + sp, dtype=np.float32)
    map_sz = np.array([map_size_scale, map_size_scale], dtype=np.float32)

    sg = json.loads(sg_json)
    raw_nodes = sg.get("nodes", []) or []
    raw_edges = sg.get("edges", []) or []
    if not raw_nodes:
        return idx, cond, map_sz, None

    id_to_idx = {n["id"]: i for i, n in enumerate(raw_nodes)}
    coords = np.zeros((len(raw_nodes), 2), dtype=np.float32)
    edge_list = []
    for n in raw_nodes:
        i = id_to_idx[n["id"]]
        coords[i] = [float(n.get("x", 0)), float(n.get("y", 0))]
    for e in raw_edges:
        src = id_to_idx.get(e.get("source", -1))
        tgt = id_to_idx.get(e.get("target", -1))
        if src is not None and tgt is not None and src != tgt:
            edge_list.append((src, tgt))

    edge_index = (
        np.array(edge_list, dtype=np.int64) if edge_list else np.zeros((0, 2), dtype=np.int64)
    )

    field = graph_to_raster(coords, edge_index, resolution=resolution, binary_centerline=True)
    fld = np.stack(
        [
            field["road_prob"],
            field["sin_2theta"],
            field["cos_2theta"],
            field["junction_hm"],
            field["endpoint_hm"],
            field["soft_distance"],
        ],
        axis=0,
    )
    return idx, cond, map_sz, fld.astype(np.float32)


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
    cache_fields: bool = True,
) -> DataLoader:
    """Build a DataLoader over skeleton field samples."""
    dataset = SkeletonFieldDataset(
        split=split, limit_samples=limit_samples, resolution=resolution, cache_fields=cache_fields
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fields,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

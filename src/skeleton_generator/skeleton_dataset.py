"""
Field Dataset — renders GT skeleton graphs as raster field tensors.

Each sample calls graph_to_field() on the fly and returns:
  - condition: style_vector, structural_priors, map_size, complexity
  - field: (6, H, W) tensor:
      [0] road_prob (binary centerline, near-0/1)
      [1] sin_2theta, [2] cos_2theta
      [3] junction_hm, [4] endpoint_hm
      [5] soft_distance (auxiliary, for training stability)
"""

from typing import List, Optional
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .config import CONFIG
from .field_config import FIELD_CONFIG as cfg
from .bfs_ordering import BFSOrdering
from .graph_to_field import graph_to_field

RESOLUTION = CONFIG.resolution  # mirrors config; change via --resolution CLI


class SkeletonFieldDataset(Dataset):
    """Loads skeleton graphs from Parquet and renders them as raster fields."""

    def __init__(self, split: str = "train", limit_samples: Optional[int] = None,
                 resolution: int = RESOLUTION):
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
            "road_density_km_per_km2", "gridness_score", "radialness_score",
            "organic_score", "bearing_entropy",
        ]
        self.resolution = resolution

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # Condition
        style_vector = torch.tensor([row[c] for c in self._style_cols], dtype=torch.float)
        structural_priors = torch.tensor(
            [row[c] for c in self._structural_cols], dtype=torch.float
        )
        map_size = torch.tensor([CONFIG.map_size_scale, CONFIG.map_size_scale], dtype=torch.float)
        complexity = torch.zeros(1, dtype=torch.float)

        # Parse graph
        sg = json.loads(row["skeleton_graph_json"])
        raw_nodes: List[dict] = sg.get("nodes", []) or []
        raw_edges: List[dict] = sg.get("edges", []) or []

        if not raw_nodes:
            return self.__getitem__((idx + 1) % len(self.df))

        # Detect roundabouts (from dataset.py pattern)
        adj = {n["id"]: set() for n in raw_nodes}
        for e in raw_edges:
            s, t = e.get("source"), e.get("target")
            if s in adj and t in adj:
                adj[s].add(t); adj[t].add(s)
        roundabout_ids = set()
        for n in raw_nodes:
            nid, nt = n["id"], n.get("node_type")
            if nt in ("intersection", "major_intersection") and len(adj.get(nid, set())) >= 3:
                nbr_deg2 = sum(1 for nb in adj[nid] if len(adj.get(nb, set())) >= 2)
                if nbr_deg2 >= 2:
                    roundabout_ids.add(nid)

        # Build arrays
        id_to_idx = {n["id"]: i for i, n in enumerate(raw_nodes)}
        N = len(raw_nodes)
        coords = np.zeros((N, 2), dtype=np.float32)
        node_connect = np.zeros((N, N), dtype=np.float32)

        NODE_TYPE_TO_INT = {"skeleton_waypoint": 0, "waypoint": 0, "intersection": 1,
                            "roundabout": 2, "major_intersection": 3, "dead_end": 4, "isolated": 0}
        node_types = np.zeros(N, dtype=np.int64)

        for n in raw_nodes:
            i = id_to_idx[n["id"]]
            coords[i] = [float(n.get("x", 0)), float(n.get("y", 0))]
            nt = n.get("node_type", "waypoint")
            if n["id"] in roundabout_ids:
                nt = "roundabout"
            node_types[i] = NODE_TYPE_TO_INT.get(nt, 0)

        for e in raw_edges:
            src = id_to_idx.get(e.get("source", -1))
            tgt = id_to_idx.get(e.get("target", -1))
            if src is None or tgt is None or src == tgt:
                continue
            node_connect[src, tgt] = 1.0
            node_connect[tgt, src] = 1.0

        # BFS ordering for consistency
        orderer = BFSOrdering(neighbor_sort_by=CONFIG.bfs_neighbor_sort_by)
        order, _ = orderer.build_order(coords, node_connect)
        ord_coords = coords[order]
        reordered_adj = node_connect[order][:, order]

        # Extract edges from reordered adjacency
        edge_list = []
        for i in range(N):
            for j in range(i + 1, N):
                if reordered_adj[i, j] > 0.5:
                    edge_list.append((i, j))
        edge_index = np.array(edge_list, dtype=np.int64) if edge_list else np.zeros((0, 2), dtype=np.int64)

        # Render field (binary centerline mode: road_prob ≈ 0/1 for stable vectorization)
        field = graph_to_field(ord_coords, edge_index, resolution=self.resolution,
                               binary_centerline=True)

        # Combine into tensor (C, H, W)
        # Channels: 0=road_prob, 1=sin_2theta, 2=cos_2theta,
        #           3=junction_hm, 4=endpoint_hm, 5=soft_distance (auxiliary)
        field_tensor = torch.stack([
            torch.from_numpy(field["road_prob"]),
            torch.from_numpy(field["sin_2theta"]),
            torch.from_numpy(field["cos_2theta"]),
            torch.from_numpy(field["junction_hm"]),
            torch.from_numpy(field["endpoint_hm"]),
            torch.from_numpy(field["soft_distance"]),
        ], dim=0)  # (6, H, W)

        return {
            "style_vector": style_vector,
            "structural_priors": structural_priors,
            "map_size": map_size,
            "complexity": complexity,
            "field": field_tensor,
        }


def collate_fields(batch: List[dict]) -> dict:
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
    limit_samples: Optional[int] = None,
    resolution: int = RESOLUTION,
) -> DataLoader:
    dataset = SkeletonFieldDataset(split=split, limit_samples=limit_samples, resolution=resolution)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=collate_fields,
        pin_memory=True, persistent_workers=(num_workers > 0),
    )

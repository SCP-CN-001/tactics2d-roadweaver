"""Anchor code map retrieval and sampling."""

from __future__ import annotations

import pickle

import numpy as np
import torch


class AnchorSampler:
    """
    Sample anchor tokens for partial code map generation.

    Given an 11-dim condition, retrieve nearest neighbours from cached
    training code maps using stratified bucketing (pattern × density),
    and sample a subset of their tokens as anchors.
    """

    def __init__(
        self,
        cache_path: str = "cache/masked_code_maps/train.npz",
        cluster_cache_path: str | None = None,
        n_neighbors: int = 5,
        anchor_ratio: float = 0.1,
        seed: int = 42,
        code_map_hw: int = 32,
        n_density_bins: int = 5,
    ):
        self.n_neighbors = n_neighbors
        self.anchor_ratio = anchor_ratio
        self.base_seed = seed
        self.rng = np.random.default_rng(seed)
        self.code_map_hw = code_map_hw
        self.S = code_map_hw * code_map_hw
        self.use_clusters = cluster_cache_path is not None
        self.mask_token_id = 511  # safe default, overridden by Generator

        if self.use_clusters:
            data = np.load(cluster_cache_path + "/centroids.npz")
            self.cond_db = torch.from_numpy(data["centroids"])
            with open(cluster_cache_path + "/sources.pkl", "rb") as f:
                self.centroid_sources = pickle.load(f)
            try:
                src_data = np.load(cluster_cache_path + "/source_code_maps.npz")
                self.code_map_db = torch.from_numpy(src_data["code_maps"])
                self._use_compact_cache = True
            except FileNotFoundError:
                full_data = np.load(cache_path)
                self.code_map_db = torch.from_numpy(full_data["code_maps"].copy())
                self._use_compact_cache = False
            self.n_centroids = self.cond_db.shape[0]
            sv = self.cond_db[:, :6].numpy()
            sp = self.cond_db[:, 6:].numpy()
            self._build_bucket_indices(sv, sp, n_density_bins)
        else:
            data = np.load(cache_path)
            self.code_map_db = torch.from_numpy(data["code_maps"])
            self.cond_db = torch.from_numpy(data["conditions"])
            sv = self.cond_db[:, :6].numpy()
            sp = self.cond_db[:, 6:].numpy()
            self._build_bucket_indices(sv, sp, n_density_bins)

    def _build_bucket_indices(
        self, style_vectors: np.ndarray, structural_priors: np.ndarray, n_bins: int = 5
    ):
        """Build per-pattern and per-density-bin lookups for stratified retrieval."""
        patterns = style_vectors.argmax(axis=1)
        densities = structural_priors[:, 0]

        if densities.std() > 0:
            bin_edges = np.percentile(densities, np.linspace(0, 100, n_bins + 1))
            bin_edges[-1] += 1e-6
            density_bins = np.digitize(densities, bin_edges) - 1
        else:
            density_bins = np.zeros(len(densities), dtype=np.int64)
            bin_edges = np.linspace(0, 1, n_bins + 1)

        self._pattern_ids = {i: [] for i in range(6)}
        for i, p in enumerate(patterns):
            self._pattern_ids[int(p)].append(i)

        self._density_bin_ids = {i: [] for i in range(n_bins)}
        for i, b in enumerate(density_bins):
            self._density_bin_ids[int(b)].append(i)

        self._pattern_density_ids = {}
        for i, (p, b) in enumerate(zip(patterns, density_bins)):
            key = (int(p), int(b))
            self._pattern_density_ids.setdefault(key, []).append(i)

        self._n_density_bins = n_bins
        self._density_bin_edges = bin_edges

    def _query_bucket_indices(self, condition: np.ndarray) -> dict:
        sv = condition[:6]
        density = float(condition[6])
        bin_id = np.searchsorted(self._density_bin_edges, density, side="right") - 1
        bin_id = int(np.clip(bin_id, 0, self._n_density_bins - 1))
        return {"pattern": int(sv.argmax()), "density_bin": bin_id}

    def retrieve_similar(self, condition: torch.Tensor, k: int | None = None) -> list[torch.Tensor]:
        """Retrieve K code maps using pattern-aware + density-aware stratification.

        Bucket strategy (for k=5):
          - 2 from same pattern (any density)
          - 2 from same density bin (any pattern)
          - 1 from same pattern + same density bin
        Falls back to L2 KNN if any bucket is empty.
        """
        if k is None:
            k = self.n_neighbors

        if self.use_clusters:
            dist = torch.cdist(condition.cpu(), self.cond_db, p=2)
            centroid_idx = dist[0].argsort()[:k]
            sources = []
            for ci in centroid_idx.tolist():
                if self._use_compact_cache:
                    for j in range(self.code_map_db.shape[1]):
                        sources.append(self.code_map_db[ci, j])
                        if len(sources) >= k:
                            break
                else:
                    for src_idx in self.centroid_sources[ci][:2]:
                        sources.append(self.code_map_db[src_idx])
                        if len(sources) >= k:
                            break
                if len(sources) >= k:
                    break
            return sources

        cond_np = condition.cpu().numpy().ravel()
        meta = self._query_bucket_indices(cond_np)
        p, b = meta["pattern"], meta["density_bin"]

        all_candidates = []
        seen = set()
        pattern_has_samples = len(self._pattern_ids.get(p, [])) > 0

        if not pattern_has_samples:
            n_per = max(1, k // 3)
            for pool, key in [(self._density_bin_ids, b), (self._pattern_ids, p)]:
                pool_b = pool.get(key, [])
                if pool_b:
                    bucket_conds = self.cond_db[pool_b]
                    dists = torch.cdist(condition.cpu(), bucket_conds, p=2)[0]
                    for idx in [pool_b[int(i)] for i in dists.argsort().tolist()]:
                        if idx not in seen:
                            seen.add(idx)
                            all_candidates.append(self.code_map_db[idx])
                            if len(all_candidates) >= n_per:
                                break
                if len(all_candidates) >= n_per:
                    break
            if len(all_candidates) < k:
                dist = torch.cdist(condition.cpu(), self.cond_db, p=2)
                for idx in dist[0].argsort().tolist():
                    if idx not in seen:
                        seen.add(idx)
                        all_candidates.append(self.code_map_db[idx])
                        if len(all_candidates) >= k:
                            break
            return all_candidates[:k]

        n_per = max(1, k // 3)
        buckets = [("both", (p, b)), ("pattern", p), ("density", b)]
        for bucket_type, key in buckets:
            pool = {
                "both": self._pattern_density_ids,
                "pattern": self._pattern_ids,
                "density": self._density_bin_ids,
            }[bucket_type].get(key, [])
            if not pool:
                continue
            if len(pool) > 1:
                bucket_conds = self.cond_db[pool]
                dists = torch.cdist(condition.cpu(), bucket_conds, p=2)[0]
                ranked = [pool[int(i)] for i in dists.argsort().tolist()]
            else:
                ranked = list(pool)
            for idx in ranked:
                if idx not in seen:
                    seen.add(idx)
                    all_candidates.append(self.code_map_db[idx])
                    if len(all_candidates) >= k:
                        break
            if len(all_candidates) >= k:
                break

        if len(all_candidates) < k:
            dist = torch.cdist(condition.cpu(), self.cond_db, p=2)
            for idx in dist[0].argsort().tolist():
                if idx not in seen:
                    seen.add(idx)
                    all_candidates.append(self.code_map_db[idx])
                    if len(all_candidates) >= k:
                        break

        return all_candidates[:k]

    def sample_anchors(
        self, condition: torch.Tensor, anchor_ratio: float | None = None, seed: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (partial_code_map, mask) for a given condition.

        Anchors are sampled from multiple retrieved neighbours
        (not just the nearest) to improve diversity.
        """
        MASK_ID = self.mask_token_id
        S = self.S

        ratio = anchor_ratio if anchor_ratio is not None else self.rng.uniform(0.05, 0.15)
        call_seed = seed if seed is not None else int(self.rng.integers(0, 100000))
        rng = np.random.default_rng(call_seed)

        neighbors = self.retrieve_similar(condition)
        n_anchors = max(2, int(S * ratio))
        n_sources = min(len(neighbors), max(2, n_anchors // 20))

        partial = torch.full((S,), MASK_ID, dtype=torch.long)
        mask = torch.ones(S, dtype=torch.bool)

        anchors_per_source = max(1, n_anchors // n_sources)
        all_positions = np.arange(S)
        rng.shuffle(all_positions)
        pos = 0

        for source_idx in range(min(n_sources, n_anchors)):
            source = neighbors[source_idx].flatten()
            n_from = min(anchors_per_source, n_anchors - pos)
            if n_from <= 0:
                break
            anchor_pos = all_positions[pos : pos + n_from]
            pos += n_from
            anchor_pos_t = torch.from_numpy(anchor_pos).long()
            partial[anchor_pos_t] = source[anchor_pos_t].long()
            mask[anchor_pos_t] = False

        return partial, mask

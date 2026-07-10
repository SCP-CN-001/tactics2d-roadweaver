"""
Anchor-based Code Map Generator

Generates road skeleton graphs from weak 11-dim condition:

  condition
    → retrieve K similar code maps from training set
    → sample 5-15% anchor tokens
    → iterative masked completion (cosine schedule) → full code map
    → VQ decoder → road field → graph

Uses top-p nucleus sampling with temperature-adaptive confidence thresholding
for diversity while maintaining structural plausibility.
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn

from .masked_transformer import MaskedCodeModel
from .vq_vae import VQVAE
from .config import CONFIG


class AnchorSampler:
    """
    Sample anchor tokens for partial code map generation.

    Given an 11-dim condition, retrieve K nearest neighbors from cached
    training code maps and sample a subset of their tokens as anchors.

    Supports two modes:
      - full cache:  retrieve from full training set (53k samples)
      - cluster:     retrieve from 256 KMeans centroids (faster, deployable)
    """

    def __init__(self, cache_path: str = "cache/masked_code_maps/train.npz",
                 cluster_cache_path: Optional[str] = None,
                 n_neighbors: int = 5, anchor_ratio: float = 0.1,
                 seed: int = 42):
        self.n_neighbors = n_neighbors
        self.anchor_ratio = anchor_ratio
        self.base_seed = seed
        self.rng = np.random.default_rng(seed)
        self.use_clusters = cluster_cache_path is not None

        if self.use_clusters:
            import pickle
            data = np.load(cluster_cache_path + "/centroids.npz")
            self.cond_db = torch.from_numpy(data["centroids"])  # (256, 11)
            with open(cluster_cache_path + "/sources.pkl", "rb") as f:
                self.centroid_sources = pickle.load(f)
            # Compact source code maps: (256, 5, 1024) — per centroid, up to 5 anchors
            try:
                src_data = np.load(cluster_cache_path + "/source_code_maps.npz")
                self.code_map_db = torch.from_numpy(src_data["code_maps"])  # (256, 5, 1024)
                self._use_compact_cache = True
            except FileNotFoundError:
                # Fallback: load full cache
                full_data = np.load(cache_path)
                self.code_map_db = torch.from_numpy(full_data["code_maps"].copy())
                self._use_compact_cache = False
            self.n_centroids = self.cond_db.shape[0]
            print(f"  [AnchorSampler] Loaded {self.n_centroids} centroids + sources from {cluster_cache_path}")
            print(f"    Full code map cache: {len(self.code_map_db)} samples")
        else:
            data = np.load(cache_path)
            self.code_map_db = torch.from_numpy(data["code_maps"])
            self.cond_db = torch.from_numpy(data["conditions"])
            print(f"  [AnchorSampler] Loaded {len(self.code_map_db)} samples from {cache_path}")

    def retrieve_similar(self, condition: torch.Tensor, k: Optional[int] = None) -> list[torch.Tensor]:
        if k is None:
            k = self.n_neighbors

        if self.use_clusters:
            dist = torch.cdist(condition.cpu(), self.cond_db, p=2)  # (1, 256)
            centroid_idx = dist[0].argsort()[:k]
            sources = []
            for ci in centroid_idx.tolist():
                if self._use_compact_cache:
                    # code_map_db is (256, 5, 1024) — pick all 5 per centroid
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
        else:
            dist = torch.cdist(condition.cpu(), self.cond_db, p=2)
            idx = dist[0].argsort()[:k]
            return [self.code_map_db[i] for i in idx]

    def sample_anchors(self, condition: torch.Tensor,
                       anchor_ratio: Optional[float] = None,
                       seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given condition, return (partial_code_map, mask).

        Key improvements for diversity:
          - anchor_ratio randomized 5-15% per call (if not explicitly set)
          - anchors sampled from multiple neighbors (not just nearest)
          - seed-based determinism for reproducibility
        """
        MASK_ID = MaskedCodeModel.MASK_TOKEN_ID
        S = 1024

        # Randomize anchor ratio per call (5-15%) if not explicitly set
        if anchor_ratio is not None:
            ratio = anchor_ratio
        else:
            ratio = self.rng.uniform(0.05, 0.15)

        # Set seed for reproducibility
        call_seed = seed if seed is not None else int(self.rng.integers(0, 100000))
        rng = np.random.default_rng(call_seed)

        # Retrieve similar code maps
        neighbors = self.retrieve_similar(condition)

        # Sample tokens from MULTIPLE neighbors (not just one)
        n_anchors = max(2, int(S * ratio))
        n_sources = min(len(neighbors), max(2, n_anchors // 20))  # 5% per source

        partial = torch.full((S,), MASK_ID, dtype=torch.long)
        mask = torch.ones(S, dtype=torch.bool)

        # Allocate anchors across multiple source samples
        anchors_per_source = max(1, n_anchors // n_sources)
        all_positions = np.arange(S)
        rng.shuffle(all_positions)
        pos_cursor = 0

        for source_idx in range(min(n_sources, n_anchors)):
            source = neighbors[source_idx].flatten()
            n_from_source = min(anchors_per_source, n_anchors - pos_cursor)
            if n_from_source <= 0:
                break

            anchor_pos = all_positions[pos_cursor:pos_cursor + n_from_source]
            pos_cursor += n_from_source

            anchor_pos_t = torch.from_numpy(anchor_pos).long()
            partial[anchor_pos_t] = source[anchor_pos_t].long()
            mask[anchor_pos_t] = False

        return partial, mask


class AnchorGenerator(nn.Module):
    """
    Full pipeline: condition → anchor → masked completion → graph.

    Usage:
        generator = AnchorGenerator()
        graph = generator.generate(condition)  # returns graph dict
    """

    def __init__(self, vq_checkpoint: str = "checkpoints/skeleton_generator/vq_vae.pth",
                 model_checkpoint: str = "checkpoints/skeleton_generator/code_transformer.pth",
                 cache_path: str = "cache/masked_code_maps/train.npz",
                 cluster_cache_path: Optional[str] = "cache/anchor_clusters",
                 device: str = "cuda",
                 d_model: int = 256, num_layers: int = 4, nhead: int = 4,
                 num_codes: int = 512,
                 resolution: Optional[int] = None,
                 code_map_size: Optional[int] = None):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.num_layers = num_layers
        self.nhead = nhead
        self.num_codes = num_codes

        # Resolution: default to CONFIG, override with explicit param
        self.resolution = resolution if resolution is not None else CONFIG.resolution
        self.code_map_size = code_map_size if code_map_size is not None else CONFIG.code_map_size

        # VQ decoder
        self.vq = VQVAE(resolution=self.resolution, num_codes=num_codes,
                        code_map_size=self.code_map_size).to(device)
        self.vq.eval()
        for p in self.vq.parameters():
            p.requires_grad = False
        state = torch.load(vq_checkpoint, map_location=device, weights_only=True)
        vs = self.vq.state_dict()
        for k, v in state["model_state_dict"].items():
            if k in vs and v.shape == vs[k].shape:
                vs[k] = v
        self.vq.load_state_dict(vs, strict=False)

        # Masked code model
        code_map_hw = self.resolution // (4 if self.code_map_size == 32 else 2)
        self.model = MaskedCodeModel(vocab_size=num_codes + 1,
                                      d_model=self.d_model,
                                      num_layers=self.num_layers,
                                      nhead=self.nhead,
                                      max_seq_len=code_map_hw * code_map_hw).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        state = torch.load(model_checkpoint, map_location=device, weights_only=True)
        self.model.load_state_dict(state["model_state_dict"])

        # Anchor sampler
        self.sampler = AnchorSampler(cache_path=cache_path,
                                     cluster_cache_path=cluster_cache_path)

        print(f"  [AnchorGenerator] VQ + Transformer loaded, anchors via {'clusters' if cluster_cache_path else 'full cache'}")

    def set_cache(self, cache_path: str, cluster_cache_path: Optional[str] = None):
        """Replace anchor cache."""
        self.sampler = AnchorSampler(cache_path=cache_path,
                                     cluster_cache_path=cluster_cache_path)

    @torch.no_grad()
    def generate_code_map(self, condition: torch.Tensor, anchor_ratio: Optional[float] = None,
                          seed: Optional[int] = None, num_steps: int = 8,
                          temperature: float = 0.75, top_p: float = 0.65) -> torch.Tensor:
        """
        Generate code map via iterative masked completion (MaskGIT-style).

        Starts from anchor tokens + MASK, then iteratively:
          - forward through Transformer
          - sample masked positions with top-p + temperature
          - keep highest-confidence tokens, re-mask rest
          - repeat with cosine schedule

        Args:
            condition: (B, 11)
            anchor_ratio: None = randomized 5-15% per sample
            seed: random seed
            num_steps: number of iterative decoding steps
            temperature: sampling temperature
            top_p: nucleus sampling threshold
        Returns:
            code_map: (B, 32, 32)
        """
        B = condition.shape[0]
        S = 1024
        MASK_ID = MaskedCodeModel.MASK_TOKEN_ID

        # Get anchors for each sample
        all_tokens, all_masks = [], []
        for b in range(B):
            s = seed + b if seed is not None else None
            partial, mask = self.sampler.sample_anchors(
                condition[b:b+1], anchor_ratio=anchor_ratio, seed=s)
            all_tokens.append(partial)
            all_masks.append(mask)

        tokens = torch.stack(all_tokens).to(self.device)
        mask = torch.stack(all_masks).to(self.device)

        n_masked_init = mask.sum().item()

        for step in range(num_steps):
            if not mask.any():
                break

            # --- Forward through Transformer ---
            logits = self.model(tokens, mask, condition)  # (B, S, 257)

            # --- Sample masked positions ---
            logits_masked = logits[mask]  # (N, 257)
            logits_scaled = logits_masked / max(temperature, 1e-8)

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    logits_scaled, descending=True, dim=-1)
                cum_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                indices_to_remove = torch.zeros_like(
                    logits_scaled, dtype=torch.bool)
                indices_to_remove.scatter_(1, sorted_indices, remove)
                logits_scaled[indices_to_remove] = float('-inf')

            probs = torch.softmax(logits_scaled, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
            conf = probs.max(dim=-1).values

            tokens[mask] = sampled

            # --- Decide which to keep unmasked vs re-mask ---
            if step < num_steps - 1:
                cos_alpha = math.cos(
                    math.pi / 2 * (step + 1) / num_steps)
                n_target_masked = max(1, int(n_masked_init * cos_alpha))

                # Build confidence map:
                #   mask positions → conf  (freshly sampled)
                #   ~mask           → inf   (anchors & previously unmasked, never re-mask)
                conf_map = torch.zeros((B, S), device=self.device)
                conf_map[mask] = conf
                conf_map[~mask] = float('inf')

                n_masked = mask.sum().item()
                if n_masked > n_target_masked:
                    # Keep lowest-confidence tokens masked
                    masked_confs = conf_map[mask]
                    threshold = masked_confs.kthvalue(
                        n_target_masked).values
                    # Only re-mask if threshold is finite
                    if math.isfinite(threshold.item()):
                        mask = mask & (conf_map <= threshold)
                    else:
                        # All remaining confs are inf — keep as is
                        pass
            else:
                # Final step: unmask everything
                mask = torch.zeros_like(mask)

        # Safety fallback: ensure no MASK tokens remain
        if (tokens == MASK_ID).any():
            remaining = (tokens == MASK_ID)
            logits = self.model(tokens, remaining, condition)
            tokens[remaining] = logits[remaining].argmax(dim=-1)

        return tokens.reshape(-1, 32, 32)

    @torch.no_grad()
    def generate_adaptive(self, condition: torch.Tensor, seed: Optional[int] = None,
                          density_low: float = 10, density_high: float = 25,
                          num_steps: int = 8) -> Dict:
        """
        Generate a graph with density-adaptive sampling parameters.

        Args:
            condition: (B, 11) — [0:6] style_vector, [6:11] structural_priors
            seed: random seed
            density_low: threshold below which = low density
            density_high: threshold above which = high density
            num_steps: iterative decoding steps
        Returns:
            graph dict
        """
        density = condition[0, 6].item()

        # Linear interpolation for parameters
        if density <= density_low:
            temp, top_p, anchor = 0.75, 0.65, 0.15
        elif density >= density_high:
            temp, top_p, anchor = 0.65, 0.55, 0.25
        else:
            t = (density - density_low) / (density_high - density_low)
            temp = 0.75 - t * 0.10     # 0.75 → 0.65
            top_p = 0.65 - t * 0.10    # 0.65 → 0.55
            anchor = 0.15 + t * 0.10   # 0.15 → 0.25

        return self.generate(condition, anchor_ratio=anchor, seed=seed,
                             num_steps=num_steps, temperature=temp, top_p=top_p)

    @torch.no_grad()
    def generate(self, condition: torch.Tensor, anchor_ratio: Optional[float] = None,
                 seed: Optional[int] = None, num_steps: int = 8,
                 temperature: float = 0.75, top_p: float = 0.65) -> Dict:
        """
        Generate a complete graph from condition.

        Returns dict with 'coords', 'edge_index', 'node_types', 'road_field'.
        """
        from .field_to_graph import field_to_graph

        # Uniform cleanup: morphological opening/closing to remove speckle noise
        opening_r, closing_r = 2, 2
        # Density-adaptive pruning: lower density = keep shorter branches
        density = condition[0, 6].item() if condition.shape[1] >= 7 else 30

        code_map = self.generate_code_map(
            condition, anchor_ratio=anchor_ratio, seed=seed,
            num_steps=num_steps, temperature=temperature, top_p=top_p)

        field = self.vq.decode_from_code(code_map)

        # Use soft_distance (ch5) rather than binary centerline (ch0) for graph extraction.
        # The new EMA VQ decoder produces sparse ch0 activations; ch5 has broader response
        # that preserves connectivity after morphological cleanup.
        road = torch.sigmoid(field[0, 5]).cpu().numpy()
        junct = torch.sigmoid(field[0, 3]).cpu().numpy()
        endpt = torch.sigmoid(field[0, 4]).cpu().numpy()

        graph = field_to_graph(
            road, junct, endpt, road_threshold=0.10,
            resolution=self.resolution, prune_short_branches=True,
            cleanup=True, opening_radius=opening_r, closing_radius=closing_r,
            min_edge_len=0.004 if density < 20 else 0.008,
            keep_all_nodes=True,  # let graph_refiner handle simplification
        )

        return {
            "coords": graph["coords"],
            "edge_index": graph["edge_index"],
            "node_types": graph["node_types"],
            "road_field": road,
        }


def reject_noisy_graph(graph: Dict, max_n: int = 120, max_e: int = 180) -> bool:
    """Rejection filter for generated graphs."""
    n = len(graph["coords"])
    if n < 2 or n > max_n:
        return False
    e = len(graph["edge_index"])
    if e > max_e:
        return False
    return True

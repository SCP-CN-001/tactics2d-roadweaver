"""
Anchor-based Code Map Generator

Generates road skeleton graphs from weak 11-dim condition:

  condition
    → retrieve K similar code maps from training set
    → sample 5-15% anchor tokens
    → masked completion → full code map
    → VQ decoder → road field → graph

This leverages the masked transformer's completion ability (15-90% mask acc=62-78%)
while providing the spatial anchor needed to avoid code collapse.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from .masked_transformer import MaskedCodeModel
from .vq_vae import VQVAE
from .skeleton_dataset import RESOLUTION


class AnchorSampler:
    """
    Sample anchor tokens for partial code map generation.

    Given an 11-dim condition, retrieve K nearest neighbors from cached
    training code maps and sample a subset of their tokens as anchors.
    """

    def __init__(self, cache_path: str = "cache/masked_code_maps/train.npz",
                 n_neighbors: int = 5, anchor_ratio: float = 0.1,
                 seed: int = 42):
        self.n_neighbors = n_neighbors
        self.anchor_ratio = anchor_ratio
        self.base_seed = seed
        self.rng = np.random.default_rng(seed)

        data = np.load(cache_path)
        self.code_map_db = torch.from_numpy(data["code_maps"])
        self.cond_db = torch.from_numpy(data["conditions"])
        print(f"  [AnchorSampler] Loaded {len(self.code_map_db)} samples from {cache_path}")

    def retrieve_similar(self, condition: torch.Tensor, k: int = None) -> List[torch.Tensor]:
        if k is None:
            k = self.n_neighbors
        # Add small jitter to condition for diversity (seed-based)
        dist = torch.cdist(condition.cpu().unsqueeze(0), self.cond_db, p=2)
        idx = dist[0].argsort()[:k]
        return [self.code_map_db[i] for i in idx]

    def sample_anchors(self, condition: torch.Tensor,
                       anchor_ratio: float = None,
                       seed: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given condition, return (partial_code_map, mask).

        Key improvements for diversity:
          - anchor_ratio randomized 5-15% per call (if not explicitly set)
          - anchors sampled from multiple neighbors (not just nearest)
          - seed-based determinism for reproducibility
        """
        MASK_ID = 256
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

    def __init__(self, vq_checkpoint: str = "runtimes/vq_vae/checkpoints/best.pth",
                 model_checkpoint: str = "runtimes/masked_code_transformer/checkpoints/best.pth",
                 cache_path: str = "cache/masked_code_maps/train.npz",
                 device: str = "cuda"):
        super().__init__()
        self.device = device

        # VQ decoder
        self.vq = VQVAE(resolution=RESOLUTION).to(device)
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
        self.model = MaskedCodeModel(d_model=256, num_layers=4, nhead=4).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        state = torch.load(model_checkpoint, map_location=device, weights_only=True)
        self.model.load_state_dict(state["model_state_dict"])

        # Anchor sampler
        self.sampler = AnchorSampler(cache_path=cache_path)

        print(f"  [AnchorGenerator] VQ + Transformer loaded, anchors from {cache_path}")

    def set_cache(self, cache_path: str):
        """Replace anchor cache."""
        self.sampler = AnchorSampler(cache_path=cache_path)

    @torch.no_grad()
    def generate_code_map(self, condition: torch.Tensor, anchor_ratio: float = None,
                          num_steps: int = 8, temperature: float = 1.2,
                          top_p: float = 0.9, seed: int = None) -> torch.Tensor:
        """
        Generate a code map from condition + random anchors.

        Args:
            condition: (B, 11)
            anchor_ratio: None = randomized 5-15% per sample
            num_steps: iterative decoding steps (8-16 recommended)
            temperature: sampling temperature (higher = more random)
            top_p: nucleus sampling threshold (0.9 recommended)
            seed: random seed (None = each call different)
        Returns:
            code_map: (B, 32, 32)
        """
        B = condition.shape[0]
        MASK_ID = 256
        S = 1024
        import math

        # Get anchors for each sample (with seed for reproducibility)
        all_tokens, all_masks = [], []
        for b in range(B):
            s = seed + b if seed is not None else None
            partial, mask = self.sampler.sample_anchors(
                condition[b:b+1], anchor_ratio=anchor_ratio, seed=s)
            all_tokens.append(partial)
            all_masks.append(mask)

        tokens = torch.stack(all_tokens).to(self.device)
        masks = torch.stack(all_masks).to(self.device)

        # Iterative decoding
        for step in range(num_steps):
            curr_mask = (tokens == MASK_ID)
            if not curr_mask.any():
                break

            logits = self.model(tokens, curr_mask, condition)

            # Top-p (nucleus) sampling
            probs = torch.softmax(logits / temperature, dim=-1)  # (B, S, V)
            sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
            cumsum = sorted_probs.cumsum(dim=-1)

            # Mask out tokens beyond top-p
            keep = cumsum < top_p
            # Always keep at least one token
            keep[..., 0] = True

            # Set excluded to -inf so they never get sampled
            filtered_logits = logits.clone()
            filtered_logits[~keep] = float('-inf')
            filtered_probs = torch.softmax(filtered_logits / temperature, dim=-1)

            # Sample from filtered distribution
            flat_probs = filtered_probs.reshape(-1, 257)
            flat_samples = torch.multinomial(flat_probs, 1).reshape(B, S)
            confidence = filtered_probs.gather(-1, flat_samples.unsqueeze(-1)).squeeze(-1)

            # Cosine schedule for unmasking
            ratio = math.cos(math.pi / 2 * (step + 1) / num_steps)
            n_keep = max(1, int(S * ratio))

            for b in range(B):
                b_mask = curr_mask[b]
                if not b_mask.any():
                    continue
                b_conf = confidence[b].clone()
                b_conf[~b_mask] = float('inf')
                threshold = b_conf.flatten().kthvalue(n_keep + 1).values
                unmask = (confidence[b] > threshold) & b_mask
                tokens[b, unmask] = flat_samples[b, unmask]

        # Fallback: unmask any remaining with argmax
        remaining = (tokens == MASK_ID)
        if remaining.any():
            logits = self.model(tokens, remaining, condition)
            tokens[remaining] = logits[remaining].argmax(dim=-1)

        return tokens.reshape(-1, 32, 32)

    @torch.no_grad()
    def generate_adaptive(self, condition: torch.Tensor, seed: int = None,
                          density_low: float = 15, density_high: float = 35) -> Dict:
        """
        Generate a graph with density-adaptive sampling parameters.

        Args:
            condition: (B, 11) — [0:6] style_vector, [6:11] structural_priors
            seed: random seed
            density_low: threshold below which = low density
            density_high: threshold above which = high density
        Returns:
            graph dict
        """
        density = condition[0, 6].item()

        # Linear interpolation for parameters
        # Parameters tightened across the board (model is noisy by nature)
        if density <= density_low:
            temp, top_p, anchor = 0.85, 0.75, 0.12
        elif density >= density_high:
            temp, top_p, anchor = 0.70, 0.65, 0.25
        else:
            t = (density - density_low) / (density_high - density_low)
            temp = 0.85 - t * 0.15     # 0.85 → 0.70
            top_p = 0.75 - t * 0.10    # 0.75 → 0.65
            anchor = 0.12 + t * 0.13   # 0.12 → 0.25

        return self.generate(condition, anchor_ratio=anchor, num_steps=8,
                             temperature=temp, top_p=top_p, seed=seed)

    @torch.no_grad()
    def generate(self, condition: torch.Tensor, anchor_ratio: float = None,
                 num_steps: int = 8, temperature: float = 1.2,
                 top_p: float = 0.9, seed: int = None) -> Dict:
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
            condition, anchor_ratio=anchor_ratio, num_steps=num_steps,
            temperature=temperature, top_p=top_p, seed=seed)

        field = self.vq.decode_from_code(code_map)
        road = torch.sigmoid(field[0, 0]).cpu().numpy()
        junct = torch.sigmoid(field[0, 3]).cpu().numpy()
        endpt = torch.sigmoid(field[0, 4]).cpu().numpy()

        graph = field_to_graph(
            road, junct, endpt, road_threshold=0.5,
            resolution=RESOLUTION, prune_short_branches=True,
            cleanup=True, opening_radius=opening_r, closing_radius=closing_r,
            min_edge_len=0.004 if density < 20 else 0.008,
        )

        return {
            "coords": graph["coords"],
            "edge_index": graph["edge_index"],
            "node_types": graph["node_types"],
            "road_field": road,
        }




def reject_noisy_graph(graph: Dict, max_n: int = 120, max_e: int = 180,
                       max_cc: int = 3) -> bool:
    """Rejection filter for generated graphs."""
    n = len(graph["coords"])
    if n < 2 or n > max_n:
        return False
    e = len(graph["edge_index"])
    if e > max_e:
        return False
    return True

"""Backbone generation pipeline entry point."""

from __future__ import annotations

import math

import torch
from torch import nn

from network_generator.topology.raster_to_graph import field_to_graph

from .config import CONFIG
from .sampler import AnchorSampler
from .transformer import MaskedCodeModel
from .vq_vae import VQVAE


def make_generator(
    vq_checkpoint: str,
    model_checkpoint: str,
    *,
    cache_path: str = "cache/masked_code_maps/train.npz",
    device: str = "cuda",
    num_codes: int = 512,
    resolution: int = 128,
    code_map_size: int = 32,
    d_model: int = 256,
    num_layers: int = 6,
    nhead: int = 4,
    use_adaln: bool = True,
    cond_dim: int = 11,
) -> Generator:
    """Build a :class:`Generator` with the standard 2km style-checkpoint defaults.

    Shared factory so scripts don't each repeat the same 12-line instantiation
    block.  ``num_codes=512 / resolution=128 / code_map_size=32 / d_model=256 /
    num_layers=6 / nhead=4 / use_adaln=True / cond_dim=11`` are the values used
    by all current generation scripts; pass explicit overrides to change them.
    """
    return Generator(
        vq_checkpoint=vq_checkpoint,
        model_checkpoint=model_checkpoint,
        cache_path=cache_path,
        device=device,
        num_codes=num_codes,
        resolution=resolution,
        code_map_size=code_map_size,
        d_model=d_model,
        num_layers=num_layers,
        nhead=nhead,
        use_adaln=use_adaln,
        cond_dim=cond_dim,
    )


class Generator(nn.Module):
    """
    Full pipeline: condition → anchors → masked completion → graph.

    Usage:
        gen = Generator(vq_checkpoint="...", model_checkpoint="...")
        code_map = gen.generate_code_map(condition)
        field = gen.vq.decode_from_code(code_map)
    """

    def __init__(
        self,
        vq_checkpoint: str,
        model_checkpoint: str,
        cache_path: str = "cache/masked_code_maps/train.npz",
        cluster_cache_path: str | None = None,
        device: str = "cuda",
        d_model: int = 512,
        num_layers: int = 6,
        nhead: int = 8,
        num_codes: int = 512,
        resolution: int | None = None,
        code_map_size: int | None = None,
        use_adaln: bool = False,
        cond_dim: int = 11,
    ):
        super().__init__()

        self.device = device
        self.num_codes = num_codes
        self.resolution = resolution or CONFIG.resolution
        self.code_map_size = code_map_size or CONFIG.code_map_size

        # ── VQ-VAE decoder ──────────────────────────────────────────────
        self.vq = VQVAE(
            resolution=self.resolution, num_codes=num_codes, code_map_size=self.code_map_size
        ).to(device)
        self.vq.eval()
        for p in self.vq.parameters():
            p.requires_grad = False
        state = torch.load(vq_checkpoint, map_location=device, weights_only=True)
        vs = self.vq.state_dict()
        for k, v in state["model_state_dict"].items():
            if k in vs and v.shape == vs[k].shape:
                vs[k] = v
        self.vq.load_state_dict(vs, strict=False)

        # ── Masked code model ───────────────────────────────────────────
        seq_len = self.code_map_size * self.code_map_size
        self.model = MaskedCodeModel(
            vocab_size=num_codes + 1,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            cond_dim=cond_dim,
            max_seq_len=seq_len,
            use_adaln=use_adaln,
        ).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        state = torch.load(model_checkpoint, map_location=device, weights_only=True)
        self.model.load_state_dict(state["model_state_dict"])

        # ── Anchor sampler ──────────────────────────────────────────────
        self.sampler = AnchorSampler(
            cache_path=cache_path,
            cluster_cache_path=cluster_cache_path,
            code_map_hw=self.code_map_size,
        )
        self.sampler.mask_token_id = self.model.mask_token_id

    def set_cache(self, cache_path: str, cluster_cache_path: str | None = None):
        """Replace anchor cache, preserving current code map size."""
        self.sampler = AnchorSampler(
            cache_path=cache_path,
            cluster_cache_path=cluster_cache_path,
            code_map_hw=self.code_map_size,
        )

    @torch.no_grad()
    def generate_code_map(
        self,
        condition: torch.Tensor,
        anchor_ratio: float | None = None,
        seed: int | None = None,
        num_steps: int = 8,
        temperature: float = 0.75,
        top_p: float = 0.65,
    ) -> torch.Tensor:
        """Generate code map via iterative masked completion.

        Args:
            condition: (B, 11) style vector + structural priors.
            anchor_ratio: fraction of anchor tokens (None = randomised 5-15%).
            seed: random seed for anchor sampling.
            num_steps: number of iterative decoding steps.
            temperature: sampling temperature.
            top_p: nucleus sampling threshold.
        Returns:
            code_map: (B, H, W) code IDs.
        """
        B = condition.shape[0]
        S = self.sampler.S
        MASK_ID = self.model.mask_token_id

        # Get anchors for each sample in the batch
        all_tokens, all_masks = [], []
        for b in range(B):
            s = seed + b if seed is not None else None
            partial, mask = self.sampler.sample_anchors(
                condition[b : b + 1], anchor_ratio=anchor_ratio, seed=s
            )
            all_tokens.append(partial)
            all_masks.append(mask)

        tokens = torch.stack(all_tokens).to(self.device)
        mask = torch.stack(all_masks).to(self.device)
        n_masked_init = mask.sum().item()

        for step in range(num_steps):
            if not mask.any():
                break

            logits = self.model(tokens, condition)
            logits_masked = logits[mask] / max(temperature, 1e-8)

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits_masked, descending=True, dim=-1)
                cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                indices_to_remove = torch.zeros_like(logits_masked, dtype=torch.bool)
                indices_to_remove.scatter_(1, sorted_indices, remove)
                logits_masked[indices_to_remove] = float("-inf")

            probs = torch.softmax(logits_masked, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
            conf = probs.max(dim=-1).values
            tokens[mask] = sampled

            # Decide which to keep unmasked vs re-mask
            if step < num_steps - 1:
                cos_alpha = math.cos(math.pi / 2 * (step + 1) / num_steps)
                n_target = max(1, int(n_masked_init * cos_alpha))

                conf_map = torch.zeros((B, S), device=self.device)
                conf_map[mask] = conf
                conf_map[~mask] = float("inf")

                if mask.sum().item() > n_target:
                    threshold = conf_map[mask].kthvalue(n_target).values
                    if math.isfinite(threshold.item()):
                        mask = mask & (conf_map <= threshold)
            else:
                mask = torch.zeros_like(mask)

        # Safety fallback: unmask any remaining MASK positions
        if (tokens == MASK_ID).any():
            remaining = tokens == MASK_ID
            logits = self.model(tokens, condition)
            tokens[remaining] = logits[remaining].argmax(dim=-1)

        return tokens.reshape(-1, self.code_map_size, self.code_map_size)

    @torch.no_grad()
    def generate(
        self,
        condition: torch.Tensor,
        anchor_ratio: float | None = None,
        seed: int | None = None,
        num_steps: int = 8,
        temperature: float = 0.75,
        top_p: float = 0.65,
    ) -> dict:
        """Generate a complete road graph from condition.

        Returns a dict with 'coords', 'edge_index', 'node_types', 'road_field'.
        """
        code_map = self.generate_code_map(
            condition,
            anchor_ratio=anchor_ratio,
            seed=seed,
            num_steps=num_steps,
            temperature=temperature,
            top_p=top_p,
        )

        field = self.vq.decode_from_code(code_map)

        density = condition[0, 6].item() if condition.shape[1] >= 7 else 30
        road = torch.sigmoid(field[0, 5]).cpu().numpy()

        graph = field_to_graph(
            road,
            road_threshold=0.10,
            resolution=self.resolution,
            prune_short_branches=True,
            cleanup=True,
            opening_radius=2,
            closing_radius=2,
            min_edge_len=0.004 if density < 20 else 0.008,
        )

        return {
            "coords": graph["coords"],
            "edge_index": graph["edge_index"],
            "node_types": graph["node_types"],
            "road_field": road,
        }

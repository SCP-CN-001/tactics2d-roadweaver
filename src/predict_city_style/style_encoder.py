"""ResNet-based style encoder for CRHD images."""

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class StyleEncoder(nn.Module):
    """ResNet backbone + style head -> softmax-normalized style vector.

    Args:
        style_dim: Output dimension (default 6 for the 6 road patterns).
        backbone_name: timm backbone name (default 'resnet34').
        pretrained: Load ImageNet weights for backbone.
        backbone_weights: Path to local safetensors/.pth file for backbone init.
        dropout_rate: Dropout in style head.
    """

    def __init__(
        self,
        style_dim: int = 6,
        backbone_name: str = 'resnet34',
        pretrained: bool = False,
        backbone_weights: Optional[str] = None,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.style_dim = style_dim
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained,
            features_only=False, num_classes=0,
        )

        # Load backbone weights from local file (safetensors or .pth)
        if backbone_weights and os.path.exists(backbone_weights):
            self._load_backbone_weights(backbone_weights)

        # Infer backbone feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            self._feat_dim = self.backbone(dummy).shape[-1]

        self.head = nn.Sequential(
            nn.Linear(self._feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, style_dim),
        )

    def _load_backbone_weights(self, path: str):
        """Load backbone-only weights from a safetensors or .pth file."""
        ext = os.path.splitext(path)[1]
        if ext == '.safetensors':
            from safetensors import safe_open
            imported_state = {}
            with safe_open(path, framework='pt', device='cpu') as f:
                for key in f.keys():
                    imported_state[key] = f.get_tensor(key)
        else:
            imported_state = torch.load(path, map_location='cpu', weights_only=False)
            if 'state_dict' in imported_state:
                imported_state = imported_state['state_dict']
            if 'model_state_dict' in imported_state:
                imported_state = imported_state['model_state_dict']

        # Filter to only keys present in the backbone, removing classifier/head keys
        backbone_state = {}
        for key in imported_state:
            # Skip classifier/head keys from the original timm model
            if key.startswith('head.') or key.startswith('fc.'):
                continue
            if key in self.backbone.state_dict():
                backbone_state[key] = imported_state[key]

        if backbone_state:
            self.backbone.load_state_dict(backbone_state, strict=False)
            print(f'  [StyleEncoder] Loaded {len(backbone_state)} backbone weight tensors from {path}')
        else:
            print(f'  [StyleEncoder] WARNING: No matching backbone keys found in {path}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: (B,3,H,W) -> (B,style_dim) with softmax."""
        features = self.backbone(x)
        logits = self.head(features)
        return F.softmax(logits, dim=-1)


def build_encoder(
    style_dim: int = 6,
    checkpoint_path: Optional[str] = None,
    device: str = 'cpu',
) -> 'StyleEncoder':
    """Build encoder and optionally load checkpoint."""
    model = StyleEncoder(style_dim=style_dim)
    model.to(device)
    model.eval()

    if checkpoint_path and checkpoint_path.lower() not in ('none', ''):
        state = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
        print(f'  [build] Loaded checkpoint: {checkpoint_path}')

    return model

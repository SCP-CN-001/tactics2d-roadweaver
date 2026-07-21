"""
Field reconstruction loss: BCE + Dice + orientation L1 + focal + distance L1.

Uses built-in PyTorch losses where available (BCEWithLogitsLoss, L1Loss)
and torchvision's sigmoid_focal_loss for junction/endpoint heatmaps.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.ops import sigmoid_focal_loss as _sigmoid_focal_loss


class DiceLoss(nn.Module):
    """Smooth Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1e-8):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.flatten(1)
        target = target.flatten(1)
        num = 2 * (pred * target).sum(dim=1) + self.smooth
        den = pred.sum(dim=1) + target.sum(dim=1) + self.smooth
        return (1 - num / den).mean()


class FieldLoss(nn.Module):
    """Multi-channel road field reconstruction loss.

    Channels:
      [0] road_prob     — BCEWithLogitsLoss + DiceLoss
      [1-2] sin/cos_2theta — L1Loss (masked to road pixels)
      [3] junction_hm   — sigmoid_focal_loss
      [4] endpoint_hm   — sigmoid_focal_loss
      [5] soft_distance — L1Loss
    """

    def __init__(self, focal_gamma: float = 2.0, road_weight: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.l1 = nn.L1Loss()
        self.focal_gamma = focal_gamma
        self.road_weight = road_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> dict:
        pred = torch.sigmoid(logits)
        road_pred = pred[:, 0]
        road_tgt = target[:, 0]

        # Road: BCE + Dice
        bce = self.bce(logits[:, 0], road_tgt)
        dice = self.dice(road_pred, road_tgt)

        # Orientation (ch 1, 2): L1 on road pixels only
        orient_mask = road_tgt > 0.5
        if orient_mask.any():
            orient_loss = (
                self.l1(pred[:, 1][orient_mask], target[:, 1][orient_mask])
                + self.l1(pred[:, 2][orient_mask], target[:, 2][orient_mask])
            ) / 2.0
        else:
            orient_loss = torch.tensor(0.0, device=logits.device)

        # Junction + endpoint: focal loss
        junct_loss = _sigmoid_focal_loss(
            logits[:, 3], target[:, 3], gamma=self.focal_gamma, reduction="mean"
        )
        endpt_loss = _sigmoid_focal_loss(
            logits[:, 4], target[:, 4], gamma=self.focal_gamma, reduction="mean"
        )

        # Distance: L1
        dist_loss = self.l1(pred[:, 5], target[:, 5])

        # IoU for monitoring
        inter = (road_pred * road_tgt).sum(dim=(1, 2))
        union = road_pred.sum(dim=(1, 2)) + road_tgt.sum(dim=(1, 2)) - inter
        road_iou = (inter / (union + 1e-8)).mean()

        total = self.road_weight * (bce + dice) + orient_loss + junct_loss + endpt_loss + dist_loss

        return {
            "total": total,
            "bce": bce,
            "dice": dice,
            "orient_loss": orient_loss,
            "junct_loss": junct_loss,
            "endpt_loss": endpt_loss,
            "dist_loss": dist_loss,
            "road_iou": road_iou,
        }

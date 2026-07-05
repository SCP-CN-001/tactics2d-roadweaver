"""
Shared loss functions for field reconstruction.

FieldLoss: BCE + Dice + orientation L1 + junction/endpoint Focal + soft distance L1.
FocalLoss: gamma-weighted BCE.
"""

import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        return (1 - pt) ** self.gamma * bce


class FieldLoss(nn.Module):
    """
    Multi-channel field reconstruction loss.

    Channels:
      [0] road_prob — BCE + Dice
      [1-2] sin/cos_2theta — L1 masked by road
      [3] junction_hm — Focal
      [4] endpoint_hm — Focal
      [5] soft_distance — L1 (auxiliary, low weight)
    """

    def __init__(self):
        super().__init__()
        self.focal = FocalLoss(gamma=2.0)
        self.l1 = nn.L1Loss(reduction='none')

    def forward(self, pred_logits, target):
        losses = {}

        # 0: Road BCE
        losses["road_bce"] = F.binary_cross_entropy_with_logits(
            pred_logits[:, 0], target[:, 0], reduction='mean')

        # 0: Road Dice
        road_probs = torch.sigmoid(pred_logits[:, 0])
        inter = (road_probs * target[:, 0]).sum(dim=(1, 2))
        union = road_probs.sum(dim=(1, 2)) + target[:, 0].sum(dim=(1, 2))
        losses["road_dice"] = (1 - (2 * inter + 1) / (union + 1)).mean()

        # 1-2: Orientation (masked)
        road_mask = (target[:, 0] > 0.5).float().detach()
        orient_loss = self.l1(torch.tanh(pred_logits[:, 1]), target[:, 1]) * road_mask
        orient_loss = orient_loss.sum() / (road_mask.sum() + 1e-8)
        orient_loss_cos = self.l1(torch.tanh(pred_logits[:, 2]), target[:, 2]) * road_mask
        orient_loss = orient_loss + orient_loss_cos.sum() / (road_mask.sum() + 1e-8)
        losses["orient"] = orient_loss / 2.0

        # 3: Junction (Focal)
        losses["junction"] = self.focal(pred_logits[:, 3], target[:, 3]).mean()

        # 4: Endpoint (Focal)
        losses["endpoint"] = self.focal(pred_logits[:, 4], target[:, 4]).mean()

        # 5: Soft distance (auxiliary L1)
        if target.shape[1] >= 6 and pred_logits.shape[1] >= 6:
            losses["soft_distance"] = F.l1_loss(
                torch.sigmoid(pred_logits[:, 5]), target[:, 5])
        else:
            losses["soft_distance"] = torch.tensor(0.0, device=pred_logits.device)

        # Metrics
        with torch.no_grad():
            road_pred = (torch.sigmoid(pred_logits[:, 0]) > 0.5).float()
            road_gt = (target[:, 0] > 0.5).float()
            tp = (road_pred * road_gt).sum(dim=(1, 2))
            fp = (road_pred * (1 - road_gt)).sum(dim=(1, 2))
            fn = ((1 - road_pred) * road_gt).sum(dim=(1, 2))
            losses["road_iou"] = (tp / (tp + fp + fn + 1e-8)).mean()

        losses["total"] = (
            losses["road_dice"] * 2.0
            + losses["road_bce"] * 1.0
            + losses["orient"] * 0.2
            + losses["junction"] * 1.0
            + losses["endpoint"] * 0.5
            + losses["soft_distance"] * 0.1
        )
        return losses

"""Losses for distance regression and the leaderboard's F1@2m decision."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def focal_cross_entropy(
    logits: Tensor, targets: Tensor, gamma: float = 2.0
) -> Tensor:
    """Per-sample focal cross-entropy for ordinal bin logits."""
    log_probability = F.log_softmax(logits, dim=-1)
    log_target = log_probability.gather(1, targets[:, None]).squeeze(1)
    probability = log_target.exp()
    return -((1.0 - probability) ** gamma) * log_target


def obstacle_loss(
    distance: Tensor,
    bin_logits: Tensor,
    near_logit: Tensor,
    target: Tensor,
    bin_edges: Tensor,
    near_threshold: float = 2.0,
    near_weight: float = 8.0,
    regression_weight: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    valid = torch.isfinite(target) & (target <= bin_edges[-1])
    count = valid.sum().clamp(min=1)
    clipped = target.clamp(bin_edges[0], bin_edges[-1] - 1e-4)
    classes = torch.bucketize(clipped, bin_edges[1:-1])
    ordinal = (focal_cross_entropy(bin_logits, classes) * valid).sum() / count
    regression = (
        F.smooth_l1_loss(
            torch.log1p(distance),
            torch.log1p(clipped),
            beta=0.1,
            reduction="none",
        )
        * valid
    ).sum() / count
    near = (
        F.binary_cross_entropy_with_logits(
            near_logit, (target < near_threshold).float(), reduction="none"
        )
        * valid
    ).sum() / count
    total = ordinal + regression_weight * regression + near_weight * near
    return total, {"ordinal": ordinal, "regression": regression, "near": near}

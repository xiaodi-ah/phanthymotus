"""Leaderboard-compatible distance metrics."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def obstacle_metrics(
    prediction: Tensor, target: Tensor, threshold: float = 2.0
) -> dict[str, float]:
    predicted_near = prediction < threshold
    actual_near = target < threshold
    true_positive = (predicted_near & actual_near).sum().item()
    false_positive = (predicted_near & ~actual_near).sum().item()
    false_negative = (~predicted_near & actual_near).sum().item()
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    rmse = math.sqrt(torch.mean((prediction - target) ** 2).item())
    return {"precision": precision, "recall": recall, "f1": f1, "rmse": rmse}

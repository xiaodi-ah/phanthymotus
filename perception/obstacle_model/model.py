"""Compact ConvNeXt model with ordinal-distance and near-obstacle heads."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def build_bin_edges(
    num_bins: int = 64,
    dense_limit: float = 1.5,
    dense_step: float = 0.05,
    max_distance: float = 655.0,
) -> Tensor:
    """Build dense near-field bins followed by logarithmic far-field bins."""
    dense_bins = round(dense_limit / dense_step)
    if dense_bins >= num_bins:
        raise ValueError("num_bins must leave room for far-field bins")
    near = torch.linspace(0.0, dense_limit, dense_bins + 1)
    far = torch.logspace(
        torch.log10(torch.tensor(dense_limit)),
        torch.log10(torch.tensor(max_distance)),
        num_bins - dense_bins + 1,
    )[1:]
    return torch.cat((near, far))


def bin_centers(edges: Tensor) -> Tensor:
    """Use arithmetic centers nearby and geometric centers after 1.5 m."""
    left, right = edges[:-1], edges[1:]
    centers = (left + right) * 0.5
    far = left >= 1.5
    centers[far] = torch.sqrt(left[far] * right[far])
    return centers


def decode_distance(logits: Tensor, centers: Tensor) -> Tensor:
    """Decode a categorical distance distribution to its expected distance."""
    return (logits.softmax(dim=-1) * centers).sum(dim=-1)


def calibrate_near_threshold(
    distance: Tensor,
    near_logit: Tensor,
    threshold: float = 2.0,
    margin: float = 0.01,
) -> Tensor:
    """Make the published distance agree with the dedicated near-obstacle head."""
    near_value = torch.minimum(
        distance, torch.full_like(distance, threshold - margin)
    )
    far_value = torch.maximum(
        distance, torch.full_like(distance, threshold + margin)
    )
    return torch.where(near_logit > 0.0, near_value, far_value)


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.probability == 0.0:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep)
        return x * mask / keep


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int, drop_path: float = 0.0) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=7, padding=3, groups=channels
        )
        self.norm = LayerNorm2d(channels)
        self.expand = nn.Conv2d(channels, channels * 4, kernel_size=1)
        self.act = nn.GELU()
        self.project = nn.Conv2d(channels * 4, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1) * 1e-6)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.norm(x)
        x = self.expand(x)
        x = self.act(x)
        x = self.project(x)
        return residual + self.drop_path(self.gamma * x)


class ConvNeXtFemto(nn.Module):
    """Five-channel ConvNeXt-Femto sized for Jetson TensorRT FP16."""

    def __init__(
        self,
        num_bins: int = 64,
        dims: tuple[int, ...] = (48, 96, 192, 384),
        depths: tuple[int, ...] = (2, 2, 6, 2),
        drop_path_rate: float = 0.1,
        max_distance: float = 655.0,
    ) -> None:
        super().__init__()
        if len(dims) != 4 or len(depths) != 4:
            raise ValueError("ConvNeXtFemto requires four stages")

        edges = build_bin_edges(num_bins=num_bins, max_distance=max_distance)
        self.register_buffer("bin_edges", edges)
        self.register_buffer("bin_centers", bin_centers(edges))

        self.stem = nn.Sequential(
            nn.Conv2d(5, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0]),
        )
        rates = torch.linspace(0.0, drop_path_rate, sum(depths)).tolist()
        cursor = 0
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for stage, (channels, depth) in enumerate(zip(dims, depths)):
            blocks = [
                ConvNeXtBlock(channels, rates[cursor + index])
                for index in range(depth)
            ]
            cursor += depth
            self.stages.append(nn.Sequential(*blocks))
            if stage < len(dims) - 1:
                self.downsamples.append(
                    nn.Sequential(
                        LayerNorm2d(channels),
                        nn.Conv2d(channels, dims[stage + 1], kernel_size=2, stride=2),
                    )
                )

        self.head_norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.ordinal_head = nn.Linear(dims[-1], num_bins)
        self.near_head = nn.Linear(dims[-1], 1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = self.stem(x)
        for index, stage in enumerate(self.stages):
            x = stage(x)
            if index < len(self.downsamples):
                x = self.downsamples[index](x)
        features = self.head_norm(x.mean(dim=(-2, -1)))
        bin_logits = self.ordinal_head(features)
        near_logit = self.near_head(features).squeeze(-1)
        distance = decode_distance(bin_logits, self.bin_centers)
        return distance, bin_logits, near_logit


VARIANTS: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "femto": ((48, 96, 192, 384), (2, 2, 6, 2)),
    "nano": ((64, 128, 256, 512), (2, 2, 6, 2)),
}


def build_convnext(variant: str = "femto", **kwargs) -> ConvNeXtFemto:
    """Build a ConvNeXtFemto from a named width/depth variant."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown model variant {variant!r}; use {sorted(VARIANTS)}")
    dims, depths = VARIANTS[variant]
    return ConvNeXtFemto(dims=dims, depths=depths, **kwargs)


class ObstacleInferenceModel(nn.Module):
    """Deployment wrapper that applies the leaderboard F1 threshold calibration."""

    def __init__(self, model: ConvNeXtFemto, near_threshold: float = 2.0) -> None:
        super().__init__()
        self.model = model
        self.near_threshold = near_threshold

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        distance, bin_logits, near_logit = self.model(x)
        distance = calibrate_near_threshold(distance, near_logit, self.near_threshold)
        return distance, bin_logits, near_logit

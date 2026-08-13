"""Train ConvNeXt-Femto on the EI_dataset indoor/outdoor splits."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader

from .data import (
    EIOutdoorDepthDataset,
    EIOutdoorSegVk2Dataset,
    EIIndoorDataset,
)
from .losses import obstacle_loss
from .metrics import obstacle_metrics
from .model import ConvNeXtFemto, calibrate_near_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/mnt/contest_ceph/i-wangaodi/datasets/EI_dataset"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/obstacle/cache"))
    parser.add_argument(
        "--obb-root",
        type=Path,
        default=Path("/tmp/vkitti2_textgt"),
        help="VKITTI2 pose.txt/bbox.txt root for bumper-to-OBB outdoor labels",
    )
    parser.add_argument(
        "--fov-aug",
        action="store_true",
        help="Enable random FOV crop augmentation for outdoor samples (experimental)",
    )
    parser.add_argument("--output", type=Path, default=Path("runs/obstacle"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-distance", type=float, default=50.0)
    parser.add_argument("--near-threshold", type=float, default=2.0)
    parser.add_argument("--smoke-samples", type=int, default=0)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_datasets(
    root: Path,
    cache_dir: Path,
    smoke_samples: int = 0,
    obb_root: Path | None = None,
    fov_aug: bool = False,
) -> tuple[ConcatDataset, ConcatDataset]:
    if not root.is_dir():
        raise FileNotFoundError(f"EI_dataset root not found: {root}")
    train = ConcatDataset(
        [
            EIIndoorDataset(
                root / "train" / "indoor",
                augment=True,
                cache_dir=cache_dir,
                limit=smoke_samples,
            ),
            EIOutdoorDepthDataset(
                root / "train" / "outdoor_depth",
                augment=True,
                cache_dir=cache_dir,
                limit=smoke_samples,
                obb_root=obb_root,
                fov_aug=fov_aug,
            ),
            EIOutdoorSegVk2Dataset(
                root / "train" / "outdoor_seg",
                augment=True,
                cache_dir=cache_dir,
                limit=smoke_samples,
                obb_root=obb_root,
                fov_aug=fov_aug,
            ),
        ]
    )
    validation = ConcatDataset(
        [
            EIIndoorDataset(
                root / "val" / "indoor",
                cache_dir=cache_dir,
                limit=smoke_samples,
            ),
            EIOutdoorDepthDataset(
                root / "val" / "outdoor_depth",
                cache_dir=cache_dir,
                limit=smoke_samples,
                obb_root=obb_root,
                fov_aug=fov_aug,
            ),
            EIOutdoorSegVk2Dataset(
                root / "val" / "outdoor_seg",
                cache_dir=cache_dir,
                limit=smoke_samples,
                obb_root=obb_root,
                fov_aug=fov_aug,
            ),
        ]
    )
    return train, validation


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    near_threshold: float = 2.0,
) -> dict[str, float]:
    model.eval()
    predictions, targets = [], []
    for images, target, _domain in loader:
        prediction, _, near_logit = model(images.to(device, non_blocking=True))
        predictions.append(
            calibrate_near_threshold(prediction, near_logit, near_threshold).cpu()
        )
        targets.append(target)
    return obstacle_metrics(
        torch.cat(predictions), torch.cat(targets), threshold=near_threshold
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, validation_set = build_datasets(
        args.dataset_root,
        args.cache_dir,
        args.smoke_samples,
        args.obb_root,
        fov_aug=args.fov_aug,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_options)
    validation_loader = DataLoader(validation_set, shuffle=False, **loader_options)

    model = ConvNeXtFemto(max_distance=args.max_distance).to(device)
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda", init_scale=1024.0
    )
    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, target, _domain in train_loader:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                distance, bin_logits, near_logit = model(images)
                loss, _ = obstacle_loss(
                    distance,
                    bin_logits,
                    near_logit,
                    target,
                    model.bin_edges,
                    near_threshold=args.near_threshold,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        scheduler.step()

        metrics = evaluate(
            model, validation_loader, device, args.near_threshold
        )
        summary = {
            "epoch": epoch,
            "loss": running_loss / max(len(train_loader), 1),
            "lr": scheduler.get_last_lr()[0],
            **metrics,
        }
        print(json.dumps(summary, sort_keys=True), flush=True)
        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "max_distance": args.max_distance,
            "near_threshold": args.near_threshold,
        }
        torch.save(checkpoint, args.output / "last.pt")
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(checkpoint, args.output / "best.pt")


if __name__ == "__main__":
    main()

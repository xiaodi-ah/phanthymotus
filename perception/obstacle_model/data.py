"""NYUv2 and Virtual KITTI 2 data adapters."""

from __future__ import annotations

import csv
import json
import math
import random
import re
from pathlib import Path
from typing import Iterable

import cv2
import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

IMAGE_SIZE = (320, 240)
RGB_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
RGB_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
OUTDOOR_MAX_DISTANCE = 50.0
# VKITTI2 classSeg vehicle obstacle colors given as RGB
# (255,127,80)/(210,0,200)/(255,130,0); stored here as BGR tuples.
_OBSTACLE_BGR_COLORS = ((80, 127, 255), (200, 0, 210), (0, 130, 255))
_VKITTI_NAME_RE = re.compile(r"vk2_(Scene\d+)_(.+?)_(\d+)_(?:rgb|depth|seg)\.")


def _parse_vkitti_name(name: str) -> tuple[str, str, int] | None:
    match = _VKITTI_NAME_RE.match(name)
    if match is None:
        return None
    return match.group(1), match.group(2), int(match.group(3))


def _random_fov_crop(rgb: np.ndarray) -> np.ndarray:
    """Random center crop to a narrower FOV, mimicking a robot camera."""
    height, width = rgb.shape[:2]
    aspect = random.uniform(4.0 / 3.0, 16.0 / 9.0)
    crop_w = min(width, int(round(aspect * height)))
    crop_h = min(height, int(round(crop_w / aspect)))
    crop_w = min(crop_w, int(crop_h * aspect))
    max_x = max(width - crop_w, 0)
    max_y = max(height - crop_h, 0)
    x0 = random.randint(0, max_x) if max_x else 0
    y0 = random.randint(0, max_y) if max_y else 0
    return rgb[y0 : y0 + crop_h, x0 : x0 + crop_w]


def _letterbox_rgb(rgb: np.ndarray, size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Resize preserving aspect ratio and pad to `size=(width, height)`."""
    target_w, target_h = size
    scale = min(target_h / rgb.shape[0], target_w / rgb.shape[1])
    resized = cv2.resize(
        rgb,
        (max(1, round(rgb.shape[1] * scale)), max(1, round(rgb.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y0 = (target_h - resized.shape[0]) // 2
    x0 = (target_w - resized.shape[1]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return canvas


def preprocess_rgb(rgb: np.ndarray, augment: bool = False) -> Tensor:
    """Letterbox RGB and append normalized horizontal/vertical coordinates."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected HWC RGB image, got {rgb.shape}")
    image = _letterbox_rgb(rgb)
    image = image.astype(np.float32) / 255.0
    if augment:
        contrast = random.uniform(0.85, 1.15)
        brightness = random.uniform(-0.08, 0.08)
        image = np.clip(image * contrast + brightness, 0.0, 1.0)
        if random.random() < 0.5:
            image = image[:, ::-1].copy()
    image = (image - RGB_MEAN) / RGB_STD
    height, width = image.shape[:2]
    u = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    v = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    features = np.concatenate((image, uu[..., None], vv[..., None]), axis=2)
    return torch.from_numpy(features.transpose(2, 0, 1).copy())


def indoor_p1_target(depth: np.ndarray) -> float:
    """Apply the benchmark center-third/top-five-eighths P1 rule."""
    height, width = depth.shape
    roi = depth[: round(height * 5 / 8), width // 3 : (width * 2) // 3]
    valid = roi[np.isfinite(roi) & (roi > 0.0)]
    if valid.size == 0:
        raise ValueError("depth map contains no valid pixels in the benchmark ROI")
    return float(np.percentile(valid, 1.0))


def outdoor_depth_target(
    depth: np.ndarray, top_fraction: float = 5 / 8
) -> float:
    """Nearest valid depth above the bottom ground strip, capped at 50 m."""
    height = depth.shape[0]
    region = depth[: round(height * top_fraction)]
    valid = (
        np.isfinite(region) & (region > 0.0) & (region <= OUTDOOR_MAX_DISTANCE)
    )
    if not valid.any():
        raise ValueError("no valid depth <= 50 m in the upper region (sentinel)")
    return float(region[valid].min())


def vkitti_seg_obstacle_mask(seg: np.ndarray) -> np.ndarray:
    """Boolean obstacle mask from VKITTI2 classSeg vehicle colors."""
    if seg.ndim != 3 or seg.shape[2] != 3:
        raise ValueError(f"expected BGR classSeg image, got {seg.shape}")
    mask = np.zeros(seg.shape[:2], dtype=bool)
    for blue, green, red in _OBSTACLE_BGR_COLORS:
        mask |= (
            (seg[:, :, 0] == blue)
            & (seg[:, :, 1] == green)
            & (seg[:, :, 2] == red)
        )
    return mask


def obstacle_mask_target(depth: np.ndarray, mask: np.ndarray) -> float:
    """Nearest valid depth inside an obstacle mask, capped at 50 m."""
    values = depth[mask]
    valid = np.isfinite(values) & (values > 0.0) & (values <= OUTDOOR_MAX_DISTANCE)
    if not valid.any():
        raise ValueError("no valid obstacle depth <= 50 m")
    return float(values[valid].min())


def _read_nyu(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix == ".npz":
        with np.load(path) as sample:
            rgb, depth = sample["rgb"], sample["depth"]
    else:
        with h5py.File(path, "r") as sample:
            rgb, depth = sample["rgb"][()], sample["depth"][()]
    if rgb.shape[0] == 3:
        rgb = rgb.transpose(1, 2, 0)
    return rgb, depth


def _load_or_build_label_cache(
    cache_path: Path, builder
) -> dict[str, float]:
    """Load a JSON label cache or build it once and persist it."""
    if cache_path.is_file():
        payload = json.loads(cache_path.read_text())
        if payload.get("version") == 1:
            return {name: float(label) for name, label in payload["labels"].items()}
    print(f"building label cache {cache_path.name}", flush=True)
    labels = builder()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps({"version": 1, "labels": labels}, sort_keys=True))
    tmp_path.replace(cache_path)
    print(f"wrote label cache {cache_path.name} ({len(labels)} labels)", flush=True)
    return labels


class EIIndoorDataset(Dataset):
    """EI_dataset indoor NPZ with benchmark ROI P1 and >50 m filter."""

    def __init__(
        self,
        directory: str | Path,
        augment: bool = False,
        cache_dir: str | Path | None = None,
        limit: int = 0,
        max_distance: float = OUTDOOR_MAX_DISTANCE,
    ) -> None:
        self.directory = Path(directory)
        self.augment = augment
        self.max_distance = max_distance
        self.files = sorted(self.directory.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"no EI indoor NPZ under {self.directory}")

        def build(num: int) -> dict[str, float]:
            labels: dict[str, float] = {}
            files = self.files[:num] if num else self.files
            total = len(files)
            for index, path in enumerate(files, 1):
                try:
                    _rgb, depth = _read_nyu(path)
                    labels[path.name] = indoor_p1_target(depth)
                except (OSError, ValueError):
                    continue
                if index % 1000 == 0 or index == total:
                    print(f"[indoor] labels {index}/{total}", flush=True)
            return labels

        if limit:
            labels = build(limit)
        elif cache_dir:
            cache_path = Path(cache_dir) / f"ei_indoor_{self.directory.parent.name}.json"
            labels = _load_or_build_label_cache(cache_path, lambda: build(0))
        else:
            labels = build(0)

        self.samples: list[tuple[Path, float]] = []
        for path in self.files:
            label = labels.get(path.name)
            if label is not None and np.isfinite(label) and 0.0 < label <= self.max_distance:
                self.samples.append((path, float(label)))
        if not self.samples:
            raise RuntimeError(f"no valid EI indoor samples under {self.directory}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        path, target = self.samples[index]
        rgb, _depth = _read_nyu(path)
        return (
            preprocess_rgb(rgb, self.augment),
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(0, dtype=torch.long),
        )


class EIOutdoorDepthDataset(Dataset):
    """EI_dataset outdoor_depth RGB + depth NPZ with sentinel filtering."""

    def __init__(
        self,
        directory: str | Path,
        augment: bool = False,
        cache_dir: str | Path | None = None,
        limit: int = 0,
        max_distance: float = OUTDOOR_MAX_DISTANCE,
        obb_root: str | Path | None = None,
        fov_aug: bool = False,
    ) -> None:
        self.directory = Path(directory)
        self.augment = augment
        self.max_distance = max_distance
        self.obb_root = Path(obb_root) if obb_root else None
        self.fov_aug = fov_aug
        depth_files = sorted(self.directory.glob("vk2_*_depth.npz"))
        self.pairs: list[tuple[Path, Path]] = []
        for depth_path in depth_files:
            rgb_path = depth_path.with_name(
                depth_path.name.replace("_depth.npz", "_rgb.jpg")
            )
            self.pairs.append((depth_path, rgb_path))
        if not self.pairs:
            raise FileNotFoundError(f"no EI outdoor_depth pairs under {self.directory}")

        self._obb_targets: dict[tuple[str, str], dict[int, float]] = {}
        if self.obb_root is not None:
            for depth_path, _rgb in self.pairs:
                parsed = _parse_vkitti_name(depth_path.name)
                if parsed is None:
                    continue
                scene, variation = parsed[:2]
                if (scene, variation) in self._obb_targets:
                    continue
                sequence = self.obb_root / scene / variation
                if sequence.is_dir():
                    self._obb_targets[(scene, variation)] = vkitti_obb_targets(sequence)

        def build(num: int) -> dict[str, float]:
            labels: dict[str, float] = {}
            pairs = self.pairs[:num] if num else self.pairs
            total = len(pairs)
            for index, (depth_path, _rgb) in enumerate(pairs, 1):
                try:
                    if self.obb_root is not None:
                        parsed = _parse_vkitti_name(depth_path.name)
                        if parsed is None:
                            continue
                        label = self._obb_targets.get(
                            parsed[:2], {}
                        ).get(parsed[2])
                        if label is None:
                            continue
                    else:
                        depth = np.load(depth_path)["depth"].astype(np.float32)
                        label = outdoor_depth_target(depth)
                    labels[depth_path.name] = label
                except (OSError, KeyError, ValueError):
                    continue
                if index % 1000 == 0 or index == total:
                    print(f"[outdoor_depth] labels {index}/{total}", flush=True)
            return labels

        if limit:
            labels = build(limit)
        elif cache_dir:
            tag = "obb_" if self.obb_root is not None else ""
            cache_path = (
                Path(cache_dir) / f"ei_outdoor_depth_{tag}{self.directory.parent.name}.json"
            )
            labels = _load_or_build_label_cache(cache_path, lambda: build(0))
        else:
            labels = build(0)

        self.samples: list[tuple[Path, float]] = []
        for depth_path, rgb_path in self.pairs:
            label = labels.get(depth_path.name)
            if label is not None and np.isfinite(label) and 0.0 < label <= self.max_distance:
                self.samples.append((rgb_path, float(label)))
        if not self.samples:
            raise RuntimeError(f"no valid EI outdoor_depth samples under {self.directory}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        path, target = self.samples[index]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise OSError(f"failed to read {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.augment and self.fov_aug:
            rgb = _random_fov_crop(rgb)
        return (
            preprocess_rgb(rgb, self.augment),
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(1, dtype=torch.long),
        )


class EIOutdoorSegVk2Dataset(Dataset):
    """EI_dataset outdoor_seg VKITTI2 RGB + depth + obstacle classSeg."""

    def __init__(
        self,
        directory: str | Path,
        augment: bool = False,
        cache_dir: str | Path | None = None,
        limit: int = 0,
        max_distance: float = OUTDOOR_MAX_DISTANCE,
        obb_root: str | Path | None = None,
        fov_aug: bool = False,
    ) -> None:
        self.directory = Path(directory)
        self.augment = augment
        self.max_distance = max_distance
        self.obb_root = Path(obb_root) if obb_root else None
        self.fov_aug = fov_aug
        self.triples: list[tuple[Path, Path, Path]] = []
        for seg_path in sorted(self.directory.glob("vk2_*_seg.png")):
            base = seg_path.name[: -len("_seg.png")]
            depth_path = seg_path.with_name(f"{base}_depth.npz")
            rgb_path = seg_path.with_name(f"{base}_rgb.jpg")
            self.triples.append((rgb_path, depth_path, seg_path))
        if not self.triples:
            raise FileNotFoundError(f"no EI outdoor_seg vk2 triples under {self.directory}")

        self._obb_targets: dict[tuple[str, str], dict[int, float]] = {}
        if self.obb_root is not None:
            for rgb_path, depth_path, _seg in self.triples:
                parsed = _parse_vkitti_name(depth_path.name)
                if parsed is None:
                    continue
                scene, variation = parsed[:2]
                if (scene, variation) in self._obb_targets:
                    continue
                sequence = self.obb_root / scene / variation
                if sequence.is_dir():
                    self._obb_targets[(scene, variation)] = vkitti_obb_targets(sequence)

        def build(num: int) -> dict[str, float]:
            labels: dict[str, float] = {}
            triples = self.triples[:num] if num else self.triples
            total = len(triples)
            for index, (rgb_path, depth_path, seg_path) in enumerate(triples, 1):
                try:
                    if self.obb_root is not None:
                        parsed = _parse_vkitti_name(depth_path.name)
                        if parsed is None:
                            continue
                        label = self._obb_targets.get(
                            parsed[:2], {}
                        ).get(parsed[2])
                        if label is None:
                            continue
                    else:
                        depth = np.load(depth_path)["depth"].astype(np.float32)
                        seg = cv2.imread(str(seg_path), cv2.IMREAD_COLOR)
                        if seg is None:
                            continue
                        label = obstacle_mask_target(
                            depth, vkitti_seg_obstacle_mask(seg)
                        )
                    labels[depth_path.name] = label
                except (OSError, KeyError, ValueError):
                    continue
                if index % 500 == 0 or index == total:
                    print(f"[outdoor_seg] labels {index}/{total}", flush=True)
            return labels

        if limit:
            labels = build(limit)
        elif cache_dir:
            tag = "obb_" if self.obb_root is not None else ""
            cache_path = (
                Path(cache_dir)
                / f"ei_outdoor_seg_vk2_{tag}{self.directory.parent.name}.json"
            )
            labels = _load_or_build_label_cache(cache_path, lambda: build(0))
        else:
            labels = build(0)

        self.samples: list[tuple[Path, float]] = []
        for rgb_path, depth_path, _seg in self.triples:
            label = labels.get(depth_path.name)
            if label is not None and np.isfinite(label) and 0.0 < label <= self.max_distance:
                self.samples.append((rgb_path, float(label)))
        if not self.samples:
            raise RuntimeError(f"no valid EI outdoor_seg vk2 samples under {self.directory}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        path, target = self.samples[index]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise OSError(f"failed to read {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.augment and self.fov_aug:
            rgb = _random_fov_crop(rgb)
        return (
            preprocess_rgb(rgb, self.augment),
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(2, dtype=torch.long),
        )


class NYUv2Dataset(Dataset):
    def __init__(self, root: str | Path, augment: bool = False) -> None:
        self.root = Path(root)
        self.augment = augment
        self.files = sorted(self.root.rglob("*.npz"))
        if not self.files:
            self.files = sorted(self.root.rglob("*.h5"))
        if not self.files:
            raise FileNotFoundError(f"no NYUv2 .npz or .h5 samples under {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        rgb, depth = _read_nyu(self.files[index])
        target = indoor_p1_target(depth)
        return (
            preprocess_rgb(rgb, self.augment),
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(0, dtype=torch.long),
        )


def point_to_obb_distance(
    center_x: float,
    center_z: float,
    width: float,
    length: float,
    yaw: float,
    bumper_z: float = 3.412,
) -> float:
    """Distance from the ego front-bumper point to an oriented x-z rectangle."""
    delta_x, delta_z = -center_x, bumper_z - center_z
    along_length = delta_x * math.sin(yaw) + delta_z * math.cos(yaw)
    along_width = delta_x * math.cos(yaw) - delta_z * math.sin(yaw)
    outside_length = max(abs(along_length) - length * 0.5, 0.0)
    outside_width = max(abs(along_width) - width * 0.5, 0.0)
    return math.hypot(outside_length, outside_width)


def _visible_tracks(sequence: Path) -> dict[int, set[int]]:
    visible: dict[int, set[int]] = {}
    with (sequence / "bbox.txt").open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter=" ", skipinitialspace=True):
            if int(row["cameraID"]) != 0 or int(row["number_pixels"]) <= 0:
                continue
            visible.setdefault(int(row["frame"]), set()).add(int(row["trackID"]))
    return visible


def vkitti_obb_targets(sequence: str | Path) -> dict[int, float]:
    """Compute per-frame nearest visible vehicle OBB distance."""
    sequence = Path(sequence)
    visible = _visible_tracks(sequence)
    targets: dict[int, float] = {}
    with (sequence / "pose.txt").open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter=" ", skipinitialspace=True):
            frame, camera = int(row["frame"]), int(row["cameraID"])
            track = int(row["trackID"])
            if camera != 0 or track not in visible.get(frame, set()):
                continue
            distance = point_to_obb_distance(
                center_x=float(row["camera_space_X"]),
                center_z=float(row["camera_space_Z"]),
                width=float(row["width"]),
                length=float(row["length"]),
                yaw=float(row["rotation_camera_space_y"]),
            )
            targets[frame] = min(targets.get(frame, math.inf), distance)
    return targets


class VirtualKITTI2Dataset(Dataset):
    def __init__(
        self,
        image_root: str | Path,
        annotation_root: str | Path,
        scenes: Iterable[str] | None = None,
        augment: bool = False,
        empty_distance: float = 655.0,
    ) -> None:
        self.image_root = Path(image_root)
        self.annotation_root = Path(annotation_root)
        self.augment = augment
        selected = set(scenes or ())
        self.samples: list[tuple[Path, float]] = []
        for sequence in sorted(self.image_root.glob("Scene*_*")):
            scene, variation = sequence.name.split("_", 1)
            if selected and scene not in selected:
                continue
            annotations = self.annotation_root / scene / variation
            if not annotations.is_dir():
                continue
            targets = vkitti_obb_targets(annotations)
            info_path = sequence / "scene_info.json"
            if info_path.is_file():
                frame_count = int(json.loads(info_path.read_text())["num_frames"])
            else:
                frame_count = len(list((sequence / "rgbs").glob("rgb_*.jpg")))
            for frame in range(frame_count):
                image_path = sequence / "rgbs" / f"rgb_{frame:05d}.jpg"
                self.samples.append((image_path, targets.get(frame, empty_distance)))
        if not self.samples:
            raise FileNotFoundError("no paired VKITTI2 RGB/OBB samples were found")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        path, target = self.samples[index]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise OSError(f"failed to read {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return (
            preprocess_rgb(rgb, self.augment),
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(1, dtype=torch.long),
        )

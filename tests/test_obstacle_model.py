import pytest
import numpy as np
import torch

from perception.obstacle_model.data import (
    EIIndoorDataset,
    RGB_MEAN,
    RGB_STD,
    _parse_vkitti_name,
    _random_fov_crop,
    indoor_p1_target,
    obstacle_mask_target,
    outdoor_depth_target,
    point_to_obb_distance,
    preprocess_rgb,
    vkitti_seg_obstacle_mask,
)
from perception.obstacle_model.losses import obstacle_loss
from perception.obstacle_model.metrics import obstacle_metrics
from perception.obstacle_model.model import (
    ConvNeXtFemto,
    build_bin_edges,
    calibrate_near_threshold,
)


def test_bin_layout() -> None:
    edges = build_bin_edges()
    assert edges.shape == (65,)
    torch.testing.assert_close(edges[30], torch.tensor(1.5))
    torch.testing.assert_close(edges[:31].diff(), torch.full((30,), 0.05))
    torch.testing.assert_close(edges[-1], torch.tensor(655.0), rtol=1e-5, atol=1e-5)


def test_preprocess_and_indoor_target() -> None:
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.full((480, 640), 4.0, dtype=np.float32)
    depth[:300, 213:426] = 2.0
    depth[100:103, 300:303] = 0.5
    tensor = preprocess_rgb(rgb)
    assert tensor.shape == (5, 240, 320)
    assert 0.5 <= indoor_p1_target(depth) <= 2.0


def test_preprocess_letterboxes_wide_outdoor() -> None:
    rgb = np.full((375, 1242, 3), 255, dtype=np.uint8)
    tensor = preprocess_rgb(rgb)
    assert tensor.shape == (5, 240, 320)
    padded_black = ((0.0 - RGB_MEAN) / RGB_STD).astype(np.float32)
    for channel in range(3):
        torch.testing.assert_close(
            tensor[channel, :71, :], torch.full((71, 320), padded_black[channel])
        )
        torch.testing.assert_close(
            tensor[channel, -72:, :], torch.full((72, 320), padded_black[channel])
        )


def test_outdoor_depth_target_excludes_sentinel() -> None:
    depth = np.full((480, 640), 655.5, dtype=np.float32)
    depth[:300, 100] = 4.2
    depth[:300, 101] = 60.0
    assert outdoor_depth_target(depth) == pytest.approx(4.2)
    with pytest.raises(ValueError):
        outdoor_depth_target(np.full((480, 640), 655.5, dtype=np.float32))


def test_vkitti_seg_obstacle_mask_and_target() -> None:
    seg = np.zeros((10, 12, 3), dtype=np.uint8)
    seg[1, 1] = (80, 127, 255)
    seg[2, 2] = (200, 0, 210)
    seg[3, 3] = (0, 130, 255)
    seg[4, 4] = (0, 199, 0)
    mask = vkitti_seg_obstacle_mask(seg)
    assert mask.sum() == 3
    depth = np.full((10, 12), 3.0, dtype=np.float32)
    depth[1, 1] = 0.7
    depth[0, 0] = 655.5
    assert obstacle_mask_target(depth, mask) == pytest.approx(0.7)
    with pytest.raises(ValueError):
        obstacle_mask_target(depth, np.zeros_like(mask))


def test_ei_indoor_dataset_filters_sentinel_and_caches(tmp_path) -> None:
    directory = tmp_path / "train" / "indoor"
    directory.mkdir(parents=True)
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    valid_depth = np.full((480, 640), 2.5, dtype=np.float32)
    sentinel_depth = np.full((480, 640), 655.5, dtype=np.float32)
    np.savez(directory / "a.npz", rgb=rgb.transpose(2, 0, 1), depth=valid_depth)
    np.savez(directory / "b.npz", rgb=rgb.transpose(2, 0, 1), depth=sentinel_depth)

    dataset = EIIndoorDataset(directory, cache_dir=tmp_path / "cache")
    assert len(dataset) == 1
    tensor, target, domain = dataset[0]
    assert tensor.shape == (5, 240, 320)
    assert float(target) == pytest.approx(2.5)
    assert int(domain) == 0
    assert (tmp_path / "cache" / "ei_indoor_train.json").is_file()


def test_parse_vkitti_name() -> None:
    assert _parse_vkitti_name("vk2_Scene01_15-deg-left_00042_depth.npz") == (
        "Scene01",
        "15-deg-left",
        42,
    )
    assert _parse_vkitti_name("not_a_vkitti_file.png") is None


def test_random_fov_crop_preserves_rgb() -> None:
    rgb = np.zeros((375, 1242, 3), dtype=np.uint8)
    cropped = _random_fov_crop(rgb)
    assert cropped.ndim == 3 and cropped.shape[2] == 3
    assert 4.0 / 3.0 - 1e-6 <= cropped.shape[1] / cropped.shape[0] <= 16.0 / 9.0 + 1e-6


def test_point_to_axis_aligned_obb() -> None:
    distance = point_to_obb_distance(
        center_x=0.0,
        center_z=10.0,
        width=2.0,
        length=4.0,
        yaw=0.0,
    )
    assert abs(distance - (10.0 - 2.0 - 3.412)) < 1e-6


def test_near_threshold_calibration() -> None:
    distance = torch.tensor([3.0, 0.5, 0.5, 3.0])
    near_logit = torch.tensor([1.0, 1.0, -1.0, -1.0])
    calibrated = calibrate_near_threshold(distance, near_logit)
    torch.testing.assert_close(calibrated, torch.tensor([1.99, 0.5, 2.01, 3.0]))


def test_model_and_loss_smoke() -> None:
    model = ConvNeXtFemto(drop_path_rate=0.0)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert 4_000_000 < parameter_count < 7_000_000
    images = torch.randn(2, 5, 240, 320)
    distance, logits, near_logit = model(images)
    assert distance.shape == (2,)
    assert logits.shape == (2, 64)
    assert near_logit.shape == (2,)
    loss, parts = obstacle_loss(
        distance, logits, near_logit, torch.tensor([0.7, 6.2]), model.bin_edges
    )
    assert torch.isfinite(loss)
    assert set(parts) == {"ordinal", "regression", "near"}


def test_obstacle_metrics_uses_two_meter_threshold() -> None:
    prediction = torch.tensor([1.5, 2.5])
    target = torch.tensor([1.5, 1.5])
    metrics = obstacle_metrics(prediction, target, threshold=2.0)
    assert metrics["f1"] == pytest.approx(2.0 / 3.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(0.5)

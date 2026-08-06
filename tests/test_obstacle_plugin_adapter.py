"""Tests for the obstacle plugin's local ONNX adapter (ROS2-free)."""

import sys
import types

import numpy as np


def _install_ros_stubs() -> None:
    def make_class(**defaults):
        class Stub:
            def __init__(self, **kwargs):
                self.__dict__.update(defaults)
                self.__dict__.update(kwargs)

        return Stub

    policies = types.SimpleNamespace(RELIABLE=1, KEEP_LAST=1, VOLATILE=1)
    rclpy = types.ModuleType("rclpy")
    rclpy_node = types.ModuleType("rclpy.node")
    rclpy_node.Node = type("Node", (), {})
    rclpy_qos = types.ModuleType("rclpy.qos")
    rclpy_qos.QoSProfile = make_class()
    rclpy_qos.ReliabilityPolicy = policies
    rclpy_qos.HistoryPolicy = policies
    rclpy_qos.DurabilityPolicy = policies
    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.CompressedImage = type("CompressedImage", (), {})
    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.String = type("String", (), {})
    for name, module in [
        ("rclpy", rclpy),
        ("rclpy.node", rclpy_node),
        ("rclpy.qos", rclpy_qos),
        ("sensor_msgs", sensor_msgs),
        ("sensor_msgs.msg", sensor_msgs_msg),
        ("std_msgs", std_msgs),
        ("std_msgs.msg", std_msgs_msg),
    ]:
        sys.modules[name] = module


_install_ros_stubs()

from perception.obstacle_model.data import preprocess_rgb  # noqa: E402
from perception.plugins.obstacle import (  # noqa: E402
    LocalDistanceAdapter,
    ObstacleDistancePlugin,
    _build_distance_adapter,
)


def test_preprocess_matches_training_indoor_aspect() -> None:
    rgb = np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    np.testing.assert_allclose(
        LocalDistanceAdapter.preprocess(rgb)[0],
        preprocess_rgb(rgb).numpy(),
        rtol=1e-5,
        atol=1e-5,
    )


def test_preprocess_matches_training_wide_outdoor() -> None:
    rgb = np.random.default_rng(1).integers(0, 256, (375, 1242, 3), dtype=np.uint8)
    np.testing.assert_allclose(
        LocalDistanceAdapter.preprocess(rgb)[0],
        preprocess_rgb(rgb).numpy(),
        rtol=1e-5,
        atol=1e-5,
    )


def test_decode_rgb_jpeg() -> None:
    import cv2

    yy, xx = np.mgrid[0:120, 0:160]
    rgb = np.stack([xx % 256, yy % 256, (xx + yy) % 256], axis=-1).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr)
    assert ok
    decoded = LocalDistanceAdapter._decode_rgb(encoded.tobytes())
    assert decoded.shape == rgb.shape
    assert decoded.dtype == np.uint8
    assert np.abs(decoded.astype(int) - rgb.astype(int)).mean() < 4.0


def test_build_local_adapter_keeps_model_path() -> None:
    adapter = _build_distance_adapter(
        {
            "provider": "local",
            "model_path": "/models/obstacle.onnx",
            "model_url": "http://example.invalid/obstacle.onnx",
        }
    )
    assert isinstance(adapter, LocalDistanceAdapter)
    assert adapter.model_path == "/models/obstacle.onnx"
    assert adapter.model_url == "http://example.invalid/obstacle.onnx"


def test_config_rebuild_preserves_model_url() -> None:
    executor = types.SimpleNamespace(add_node=lambda node: None, remove_node=lambda node: None)
    plugin = ObstacleDistancePlugin(
        {
            "provider": "local",
            "model_path": "/models/obstacle.onnx",
            "model_url": "http://example.invalid/obstacle.onnx",
        },
        executor,
    )
    result = plugin.dispatch("obstacle", {"action": "config", "provider": "local"})
    assert result["status"] == "configured"
    assert plugin._adapter.model_path == "/models/obstacle.onnx"
    assert plugin._adapter.model_url == "http://example.invalid/obstacle.onnx"
    plugin.dispatch(
        "obstacle", {"action": "config", "model_url": "http://example.invalid/new.onnx"}
    )
    assert plugin._adapter.model_url == "http://example.invalid/new.onnx"


def test_estimate_falls_back_when_model_missing() -> None:
    adapter = LocalDistanceAdapter("/nonexistent/obstacle.onnx")
    result = adapter.estimate(b"not an image")
    assert result == {"pred_distance": 10.0}

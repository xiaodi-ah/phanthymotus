"""Export a trained checkpoint for TensorRT FP16 conversion."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from .model import ConvNeXtFemto, ObstacleInferenceModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = ConvNeXtFemto(max_distance=checkpoint.get("max_distance", 655.0))
    model.load_state_dict(checkpoint["model"])
    model = ObstacleInferenceModel(
        model, near_threshold=checkpoint.get("near_threshold", 2.0)
    ).eval()
    example = torch.randn(1, 5, 240, 320)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        example,
        args.output,
        input_names=["image"],
        output_names=["distance", "bin_logits", "near_logit"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(args.output))
    with torch.no_grad():
        expected = model(example)[0].numpy()
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    actual = session.run(["distance"], {"image": example.numpy()})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
    print(f"exported and verified {args.output}")


if __name__ == "__main__":
    main()

# Obstacle Distance Model

Training code for the EI_dataset layout:

```text
EI_dataset/
  train|val/
    indoor/          rgb + depth NPZ (NYUv2)
    outdoor_depth/   RGB JPG + depth NPZ (VKITTI2 geometric variants)
    outdoor_seg/     VKITTI2 RGB + depth + classSeg, nuScenes, SSCBench
```

Input preprocessing letterboxes RGB to a fixed `320x240` tensor while keeping
the aspect ratio, then appends normalized `(u, v)` coordinate channels. The
model is a five-channel ConvNeXt-Femto with 64 distance bins (5 cm bins to
1.5 m, then logarithmic bins to 50 m) and an auxiliary `<2m` head aligned with
the leaderboard F1@2m metric.

Labels:

- indoor: benchmark ROI (center third, top five eighths) P1 depth in meters;
- outdoor_depth: nearest valid depth above the bottom ground strip, full width;
- outdoor_seg VKITTI2: nearest valid depth inside the classSeg vehicle mask
  (RGB colors 255,127,80 / 210,0,200 / 255,130,0).

Sentinel handling: depth values that are non-finite or above 50 m are ignored
when computing labels, and any sample whose label exceeds 50 m is excluded from
the dataset. The loss additionally masks per-sample contributions for targets
that are non-finite or beyond the last bin, so sentinel frames never contribute
gradients. The nuScenes and SSCBench masks contain no depth and therefore do
not provide distance labels; they are not loaded by the distance datasets.

Labels are cached as JSON under `--cache-dir` on first run; delete that
directory if the dataset is regenerated.

```bash
conda activate obstacle-distance
cd perception
python -m obstacle_model.train \
  --dataset-root /mnt/contest_ceph/i-wangaodi/datasets/EI_dataset \
  --output /mnt/disk1/wangaodi/obstacle-runs/convnext-femto

python -m obstacle_model.export_onnx \
  /mnt/disk1/wangaodi/obstacle-runs/convnext-femto/best.pt \
  /mnt/disk1/wangaodi/obstacle-runs/convnext-femto/obstacle.onnx
```

For a quick pipeline check, add `--epochs 1 --smoke-samples 16 --workers 0`
and run from a directory writable for `runs/` or pass `--output /tmp/...`.
Convert the verified ONNX file to a fixed-shape TensorRT FP16 engine on the
Jetson target; do not commit checkpoints or model artifacts to Git.

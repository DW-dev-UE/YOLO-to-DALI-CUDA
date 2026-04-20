# YOLO-to-DALI-CUDA

**NVIDIA DALI accelerated data loading for Ultralytics YOLO**

Drop-in DALI data loader for Ultralytics YOLO that offloads image decoding to the GPU, cutting CPU load significantly.

<p align="center">
  <img src="https://img.shields.io/badge/DALI-Accelerated-00A4EF" alt="DALI">
  <img src="https://img.shields.io/badge/CPU%20Load-Reduced-red" alt="Reduced CPU">
</p>

---

## What this is

The default YOLO training pipeline decodes and preprocesses images entirely on the CPU via OpenCV. On GPU-heavy training rigs this becomes the bottleneck fast.

This library plugs NVIDIA DALI into the Ultralytics pipeline so that image decoding happens on the GPU instead. You flip one flag (`use_dali=True`) and everything else stays the same -- augmentations, collate, label parsing, task formatting all remain untouched.

**Heads up:**
- Built on top of Ultralytics' open-source code. Unauthorized resale or any violation of Ultralytics' terms is prohibited.
- This library fully complies with Ultralytics' license.
- DALI won't hit full performance under **WSL** (Windows Subsystem for Linux).

---

## Features

- CUDA-based DALI pipeline for GPU image decoding
- Significant CPU load reduction during training
- Supports YOLOv8, YOLOv10, YOLOv11, YOLOv26
- Minimal code changes -- just add `use_dali=True`
- Automatic cv2 fallback if DALI fails on any batch

---

## Benchmarks

We benchmarked the standard cv2 dataloader against DALI across Detection, OBB, Segmentation, and Classification tasks.

Full results, throughput numbers, and analysis are in the dedicated report:

**[View the Performance Benchmark Report](performance.md)**

---

## Quick Start

### Install

```bash
pip install nvidia-dali-cuda120
pip install git+https://github.com/DW-dev-UE/YOLO-to-DALI-CUDA.git
```

### Usage

```py
from ultralytics_dali import YOLO

# Detect
model_det = YOLO("yolo11n.pt")
model_det.train(data="/path/to/detect.yaml", task="detect", epochs=10, imgsz=1024, batch=16, device=0, workers=24, use_dali=True)

# Segment
model_seg = YOLO("yolo11n-seg.pt")
model_seg.train(data="/path/to/segment.yaml", task="segment", epochs=10, imgsz=1024, batch=16, device=0, workers=24, use_dali=True)

# Pose
model_pose = YOLO("yolo8n-pose.pt")
model_pose.train(data="/path/to/pose.yaml", task="pose", epochs=10, imgsz=1024, batch=16, device=0, workers=24, use_dali=True)

# OBB
model_obb = YOLO("yolo26n-obb.pt")
model_obb.train(data="/path/to/obb.yaml", task="obb", epochs=10, imgsz=1024, batch=16, device=0, workers=24, use_dali=True)

# Classification (pass the dataset root directory directly)
model_cls = YOLO("yolo11n-cls.pt")
model_cls.train(data="/path/to/classification_root", task="classify", epochs=10, imgsz=1024, batch=16, device=0, workers=24, use_dali=True)
```

---

## How it works

### The core change

The key modification is in `BaseDataset.load_image()`. Instead of always calling cv2 to decode images, the dataset now checks for a pluggable image provider. If DALI is enabled, decoding routes through the DALI backend. If not, it falls back to cv2 as usual.

This means all existing YOLO logic -- augmentations, collate functions, label parsing -- stays completely untouched. DALI only replaces the decode step.

### What was added

**DALI Image Provider** -- sits between the dataset and the decoder. Handles batch-aware decoding, manages the worker-local decoder lifecycle, caches decoded images for the current batch, and stays compatible with the existing dataset indexing.

**DALI Batch Decoder** -- uses DALI's `external_source` + mixed decode to decode raw image bytes on the GPU. If anything goes wrong (corrupt image, DALI error), it falls back to cv2 automatically so training doesn't crash.

### How the flag propagates

When you set `use_dali=True`:

1. The dataloader builder picks up the flag and attaches the DALI provider to the dataset
2. Cache behavior and worker handling get adjusted for DALI's requirements
3. The flag propagates through all dataset builders, so it works consistently across `detect`, `segment`, `pose`, `obb`, and `grounding` tasks

### The pipeline flow

```
use_dali=True
  -> build_yolo_dataset / build_grounding
    -> dataset created with DALI flag
      -> build_dataloader attaches DALI provider
        -> load_image() delegates to DALI provider
          -> DALI decodes on GPU
            -> decoded images go back into standard YOLO pipeline
```

Everything after the decode step is identical to the default pipeline.

> **Note:** DALI currently accelerates decode only. It does not replace YOLO's augmentation logic, collate semantics, label parsing, or task-specific formatting.

---

## Acknowledgments

- Ultralytics Team
- NVIDIA DALI Team

## License

Built on Ultralytics' open-source code. Unauthorized resale and any violation of Ultralytics' terms of use is strictly prohibited. This library fully complies with Ultralytics' license.

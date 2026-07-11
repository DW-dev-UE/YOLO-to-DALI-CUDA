# YOLO-to-DALI-CUDA

**NVIDIA DALI accelerated data loading for Ultralytics YOLO**

Drop-in DALI data loader for Ultralytics YOLO that offloads **image decode + initial resize** to the GPU.

<p align="center">
  <img src="https://img.shields.io/badge/DALI-Accelerated-00A4EF" alt="DALI">
  <img src="https://img.shields.io/badge/CPU%20Load-Reduced-red" alt="Reduced CPU">
  <img src="https://img.shields.io/badge/version-0.2%20(V2)-green" alt="V2">
</p>

---

## What's new in V2

V1 shipped a `use_dali=True` flag, but the integration was incomplete and the DALI pipeline was configured for correctness over speed. **V2 fixes both.**

| Area | V1 | V2 |
| :--- | :--- | :--- |
| Ultralytics patch | `build_yolo_dataset` only | **`build_yolo_dataset` + `build_grounding` + `build_dataloader`** |
| DALI attach | Often never ran (trainer used stock `build_dataloader`) | Provider attaches on every train/val loader |
| Executor | `exec_async=False`, `exec_pipelined=False` | **`exec_async=True`, `exec_pipelined=True`** |
| Resize | Full-res decode → CPU `cv2.resize` | **GPU `resize_longer=imgsz` inside DALI** |
| Label geometry | N/A / fragile | Provider returns **original shape** with the resized image |
| Dead code | Vendored Ultralytics tree, unused `DaliPrefetchLoader`, unused `Gpu*` classes | Removed (~71k lines) |
| Benchmark | Manual only | **`python -m ultralytics_dali.benchmark`** ON/OFF A/B tool |

**In short:** DALI actually hooks into training, the pipeline is async and resizes on GPU, and you can measure cv2 vs DALI with one command.

> Expect speedups when the **data pipeline** is the bottleneck (I/O + CPU decode). Large `imgsz` multi-GPU runs that are already GPU-compute bound may see little epoch-time change even with a faster loader.

---

## What this is

The default YOLO training pipeline decodes and resizes images on the CPU via OpenCV. On machines where the GPU waits on the loader, that becomes the bottleneck.

This library monkey-patches three Ultralytics builders at import time so that, with `use_dali=True`, decode + aspect-preserving resize run on GPU through NVIDIA DALI. Augmentations (mosaic, mixup, …), collate, labels, and task formatting stay on the standard YOLO path.

**Heads up:**
- Built on Ultralytics open-source code. Unauthorized resale or license violations are prohibited.
- Complies with Ultralytics' license (AGPL-3.0).
- DALI will not reach full performance under **WSL**.

---

## Features

- CUDA DALI pipeline: **GPU decode + GPU resize** (`resize_longer=imgsz`)
- Async / pipelined DALI executor
- Full monkeypatch: dataset **and** dataloader builders (so DALI really attaches)
- Original image shape preserved for correct label scaling after GPU resize
- `use_dali=True` flag — minimal code change
- Automatic cv2 fallback on decode failure (with warning log)
- Built-in A/B benchmark: `python -m ultralytics_dali.benchmark`
- YOLOv8 / v10 / v11 / v26 detect, segment, pose, obb, grounding

---

## Quick Start

### Install

```bash
pip install nvidia-dali-cuda120
pip install git+https://github.com/DW-dev-UE/YOLO-to-DALI-CUDA.git
```

Optional extras: `pip install "ultralytics-dali-cuda[cuda12]"` or `[cuda11]`.

### Usage

```python
from ultralytics_dali import YOLO

model = YOLO("yolo11n.pt")
model.train(
    data="/path/to/data.yaml",
    task="detect",
    epochs=10,
    imgsz=1024,
    batch=16,
    device=0,
    workers=8,
    use_dali=True,   # ← enable GPU decode + resize
)
```

Same flag works for `segment`, `pose`, `obb`. Classification (`task=classify`) still uses the stock cv2 path.

### OFF (baseline)

```python
model.train(..., use_dali=False)  # or omit the flag
```

---

## Run your own A/B benchmark

Compare **cv2 vs DALI** on the same dataset and settings (decode + augment + collate throughput only):

```bash
# both backends
python -m ultralytics_dali.benchmark --data /path/to/data.yaml --imgsz 1024 --batch 16 --workers 8

# one side only
python -m ultralytics_dali.benchmark --data /path/to/data.yaml --backends cv2
python -m ultralytics_dali.benchmark --data /path/to/data.yaml --backends dali
```

Example output:

```
backend   provider                   img/s   s/batch   images
cv2       _Cv2ImageProvider          208.5     0.077      800
dali      DaliImageProvider          512.3     0.031      800

dali vs cv2: 2.46x throughput
```

Useful flags: `--task detect|segment|pose|obb`, `--num-batches`, `--warmup`, `--split train|val`.

> This measures the **dataloader**, not full train epochs. If `nvidia-smi` already shows ~100% GPU util, a faster loader will not shorten wall-clock epochs much.

Historical multi-task numbers from an earlier setup: **[performance.md](performance.md)** (pre-V2; re-run the benchmark on your box for current numbers).

---

## Verifying DALI is active

When training starts you should see:

```
DALI: GPU decode backend enabled (workers=..., batch=..., imgsz=..., gpu_aug=...)
```

If that line is missing, DALI did not attach (`nvidia-dali` missing, no CUDA, or `use_dali=False`) and the run is on the cv2 path.

---

## How it works (V2)

### Import-time patch

Importing `ultralytics_dali` registers `use_dali` on the default cfg and replaces three symbols in stock `ultralytics`:

1. `build_yolo_dataset`
2. `build_grounding`
3. **`build_dataloader`** ← this is what calls `_attach_dali_provider`

Without (3), `use_dali=True` was effectively a no-op in V1.

### Data path

```
use_dali=True
  → patched build_yolo_dataset / build_grounding
      → dataset with use_dali flag (cache disabled for DALI)
  → patched build_dataloader
      → attaches DaliImageProvider
  → load_image()
      → DALI: read bytes → GPU decode → GPU resize_longer=imgsz
      → return (image, original_shape) for label parity
  → standard YOLO aug / collate / train
```

### What DALI does **not** replace

Mosaic, MixUp, LetterBox (post-decode geometry beyond the initial resize), collate, label parsing, and task formatting remain YOLO/CPU. That is intentional: label parity with the cv2 path is preserved.

---

## When to use DALI

| Situation | Expectation |
| :--- | :--- |
| CPU-bound decode / I/O, GPU underutilized | Good candidate for V2 |
| Single GPU, moderate `imgsz`, many workers still not feeding GPU | Often benefits |
| Very large `imgsz` + multi-GPU, GPU already ~100% | Little epoch speedup (compute-bound) |
| WSL | Reduced DALI gains |

---

## V2 change list (files)

| File / area | Change |
| :--- | :--- |
| `ultralytics_dali/__init__.py` | Patch `build_dataloader` + `build_grounding` (not only dataset) |
| `ultralytics_dali/data/dali_pipeline.py` | Async/pipelined executor, GPU resize, fixed batch + chunking, original shape, fallback warnings |
| `ultralytics_dali/data/build.py` | Attach provider; log `GPU decode backend enabled` |
| `ultralytics_dali/data/base.py` / `dataset.py` | Provider contract (shape-aware load path) |
| `ultralytics_dali/benchmark.py` | **New** — cv2/DALI A/B CLI |
| Removed | `dali_loader.py`, unused `Gpu*` augment blocks, vendored `cfg/engine/hub/models/...` tree, broken `yolo` console script entry |

---

## Acknowledgments

- Ultralytics Team
- NVIDIA DALI Team

## License

Built on Ultralytics' open-source code. Unauthorized resale and any violation of Ultralytics' terms of use is strictly prohibited. This library fully complies with Ultralytics' license.

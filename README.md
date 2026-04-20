# YOLO-to-DALI-CUDA

**NVIDIA DALI accelerated data loading for Ultralytics YOLO**

This is a DALI data loader designed specifically for Ultralytics YOLO that significantly reduces CPU load.

<p align="center">
  <img src="https://img.shields.io/badge/DALI-Accelerated-00A4EF" alt="DALI">
  <img src="https://img.shields.io/badge/CPU%20Load-Reduced-red" alt="Reduced CPU">
</p>

---

## 📌 Project Overview

This is a GitHub open-source project developed to assist users who train YOLO models on images.  <br><br>
The current YOLO implementation does not utilize NVIDIA’s CUDA DALI library, so image preprocessing is performed on the CPU. <br>This library was created to help users reduce CPU load.

🔊 **Important Notes**
- This project is built upon Ultralytics’ open-source code.
- Unauthorized sales or any actions that violate Ultralytics’ terms of use are strictly prohibited.
- This library fully complies with Ultralytics’ license and terms of use.
- In **WSL (Windows Subsystem for Linux)** environments, the DALI library cannot achieve its full performance.

---

## ✨ Features

- CUDA-based DALI pipeline for fast image preprocessing
- Dramatically reduced CPU load
- Full support for Ultralytics YOLOv8, YOLOv10, YOLOv11
- Minimal code changes required (drop-in compatible)
- High-performance data loading on NVIDIA GPUs

---

## 📊 Performance (Benchmarks)

We have conducted extensive performance benchmarks comparing the standard OpenCV (`cv2`) dataloader with the high-performance **NVIDIA DALI** backend across various vision tasks (Detection, OBB, Segmentation, Classification).

For the full benchmark results, throughput analysis, and a deep dive into why DALI significantly accelerates certain tasks, please refer to our dedicated performance document:

**👉 [View the Detailed Performance Benchmark Report](performance.md)**

## 💡 Quick Start

> INSTALL <br>

```bash
pip install nvidia-dali-cuda120
pip install git+https://github.com/DW-dev-UE/YOLO-to-DALI-CUDA.git
```

> CODE EXAMPLE

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

# Classification (Pass the dataset root directory directly)
model_cls = YOLO("yolo11n-cls.pt")
model_cls.train(data="/path/to/classification_root", task="classify", epochs=10, imgsz=1024, batch=16, device=0, workers=24, use_dali=True)
```

---

# 🏗️ Architecture Modifications for NVIDIA DALI Integration

This document outlines the specific architectural changes made to the Ultralytics YOLO pipeline to integrate NVIDIA DALI as a pluggable image decode/backend provider.

## 1. The Core Modification: Image-Provider Seam
The fundamental architectural change was refactoring `BaseDataset.load_image()` to support a pluggable image provider. This creates a "seam" in the existing pipeline.

Instead of hardcoding OpenCV (`cv2`) for decoding, the dataset can now dynamically switch between:
* The **default `cv2` provider**
* The **DALI-backed provider**

This ensures that the existing YOLO task semantics (augmentations, collate, label parsing) remain completely unchanged while allowing the underlying decode backend to be swapped out.

## 2. New Architectural Components Added

### A. DALI-Backed Image Provider
A new image provider class was introduced to handle DALI-specific operations within the dataset pipeline.
* **Responsibilities:** * Batch-aware decoding.
    * Worker-local decoder lifecycle management.
    * Decoded image caching for the current batch.
    * Maintaining compatibility with the existing dataset indexing path.

### B. DALI Batch Decoder
A dedicated DALI batch decoder utilizing `external_source` and mixed decode was integrated.
* **Input:** Raw image bytes.
* **Output:** Decoded image tensors/arrays.
* **Backend:** DALI mixed decode (GPU-accelerated).
* **Fallback Mechanism:** Automatically falls back to standard `cv2` if the DALI extension fails or batch decoding encounters an error, ensuring robust pipeline execution.

## 3. Dataloader Wiring & Propagation

To seamlessly inject the new components into the training loop, the initialization pathways were modified:

* **Dataloader Wiring:** The dataloader builder was updated to accept a `use_dali=True` configuration. When enabled, it attaches the DALI provider to the dataset, adjusts cache behavior, and aligns worker handling with DALI's requirements.
* **Build Path Propagation:** The `use_dali` flag was propagated through all dataset builders. This allows the DALI architecture to be consistently enabled across various tasks (`detect`, `segment`, `pose`, `obb`, `multimodal`, `grounding`).

## 4. The Resulting Architecture Flow

The current architecture is intentionally conservative, acting as a direct drop-in replacement for the image acquisition step. The execution flow is as follows:

1.  **Training Config:** User sets `use_dali=True`.
2.  **Dataset Builder:** `build_yolo_dataset` or `build_grounding` is called.
3.  **Dataset Creation:** The dataset is instantiated with the `use_dali` flag enabled.
4.  **Dataloader Builder:** `build_dataloader` intercepts the setup.
5.  **Provider Attachment:** The DALI Image Provider is attached to the dataset.
6.  **Data Fetching:** During iteration, `dataset.load_image()` delegates the call to the DALI provider.
7.  **GPU Decoding:** DALI decodes the image batch on the GPU.
8.  **Pipeline Continuation:** The decoded images are passed back to the standard YOLO pipeline. **Existing YOLO transforms, collate functions, and task semantics continue unchanged.**

> [!IMPORTANT]
> **Architectural Boundary:** DALI currently accelerates **decode/backend only**. It does not replace YOLO's spatial/color augmentation semantics, collate semantics, label parsing, or task formatting logic.

---

## 🙏 Acknowledgments

- Ultralytics Team
- NVIDIA DALI Team

## 📜 License & Compliance

This project is built upon Ultralytics’ open-source code.<br>
Unauthorized sales and any violations of Ultralytics’ terms of use are strictly prohibited.<br>
This library fully complies with Ultralytics’ license.

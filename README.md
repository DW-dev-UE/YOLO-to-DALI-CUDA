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

## 🙏 Acknowledgments

- Ultralytics Team
- NVIDIA DALI Team

## 📜 License & Compliance

This project is built upon Ultralytics’ open-source code.<br>
Unauthorized sales and any violations of Ultralytics’ terms of use are strictly prohibited.<br>
This library fully complies with Ultralytics’ license.

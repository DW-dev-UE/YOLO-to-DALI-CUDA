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

> *Coming soon* — Detailed benchmarks will be added after testing.  <br>
Please add your benchmark table here (this section is very important for this type of project).

---

## 💡 Quick Start

> INSTALL <br>

```bash
git clone https://github.com/DW-dev-UE/YOLO-to-DALI-CUDA.git
```

---

## 📜 License & Compliance

This project is built upon Ultralytics’ open-source code.<br>
Unauthorized sales and any violations of Ultralytics’ terms of use are strictly prohibited.<br>
This library fully complies with Ultralytics’ license.

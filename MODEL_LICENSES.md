# Model Licensing and Commercial Feasibility Audit

**Project**: Waste Detection System  
**Date Checked**: November 2026  
**Auditor**: Technical Lead & Computer Vision Engineer  

---

## 1. Overview and Legal Gate

The project requires full readiness for potential commercial deployment. In open-source machine learning and computer vision, software code licenses frequently differ from pretrained weight licenses, and certain popular architectures impose restrictive copyleft or dual-licensing terms that prohibit closed-source commercial redistribution without costly commercial agreements.

This audit evaluates candidate architectures across code licenses, pretrained weight terms, derivative model export implications, and runtime library licensing.

---

## 2. Comprehensive Model Family Evaluation

| Candidate Architecture | Official Source | Code License | Pretrained Weights License | Exported ONNX License | Commercial Viability | Commercial Gate Decision |
|---|---|---|---|---|---|---|
| **Ultralytics YOLO (v8 / 11)** | [Ultralytics GitHub](https://github.com/ultralytics/ultralytics) | AGPL-3.0 | AGPL-3.0 / Enterprise | AGPL-3.0 (Copyleft viral contamination) | Requires commercial enterprise license for proprietary commercial deployment | **RESTRICTED / CONDITIONAL**: Allowed for internal R&D, requires commercial waiver or Apache alternative for commercial distribution. |
| **YOLOX (Nano / Tiny)** | [Megvii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | Apache 2.0 | Apache 2.0 | Apache 2.0 | Full commercial use permitted without royalties | **APPROVED**: Highly compatible, permissive, no copyleft contamination. |
| **PP-YOLOE / PP-PicoDet** | [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) | Apache 2.0 | Apache 2.0 | Apache 2.0 | Full commercial use permitted | **APPROVED**: Fully permissive for commercial products. |
| **RT-DETR (Nano / Small)** | [Baidu / Lyuwenyu RT-DETR](https://github.com/lyuwenyu/RT-DETR) | Apache 2.0 | Apache 2.0 | Apache 2.0 | Full commercial use permitted | **APPROVED**: Permissive real-time transformer detector. |
| **RF-DETR** | [Roboflow / Community](https://github.com/roboflow/rf-detr) | Apache 2.0 / MIT (varying by submodule) | Community / Checkpoint dependent | Permissive | Subject to specific base checkpoint evaluation | **PENDING WEIGHT AUDIT**: Checkpoint weights must be verified individually. |
| **MobileNetV4-SSD** | [Google / Torchvision](https://github.com/tensorflow/models) | Apache 2.0 | Apache 2.0 | Apache 2.0 | Full commercial use permitted | **APPROVED**: Ultra-lightweight CPU fallback. |
| **ONNX Runtime (Runtime Engine)** | [Microsoft ONNX Runtime](https://github.com/microsoft/onnxruntime) | MIT | N/A (Engine) | MIT | Full commercial use permitted | **APPROVED**: Zero licensing friction for desktop delivery. |

---

## 3. Deep Dive: Ultralytics Licensing Restrictions (AGPL-3.0)

Ultralytics transitioned starting with YOLOv8 to the **GNU Affero General Public License v3.0 (AGPL-3.0)** alongside a proprietary commercial license option.

### Key Implications of AGPL-3.0:
1. **Source Disclosure Mandate**: If an application containing or linking to AGPL-3.0 code is distributed to external users (or interacted with over a network), the entire source code of the containing application must be licensed under AGPL-3.0 and made available to users.
2. **Derivative Weights**: Fine-tuning or converting weights using Ultralytics AGPL-3.0 training pipelines creates derivative works that Ultralytics considers subject to AGPL-3.0 terms unless an enterprise commercial license is purchased.
3. **Desktop Application Protection**:
   - To keep the desktop application cleanly decoupled and protected from copyleft contamination, the desktop application runtime does **NOT** import `ultralytics` or link to AGPL code.
   - The desktop runtime uses **Microsoft ONNX Runtime** (MIT License) and **OpenCV** (Apache 2.0 License).
   - Training scripts are maintained in a separate CLI tool directory (`training/`) so training dependencies remain isolated from the runtime application distribution.

---

## 4. Recommended Active Model Selection Rationale

1. **Inference Engine**:
   - **Microsoft ONNX Runtime CPU** (MIT License): High-speed Intel CPU execution using vectorized AVX2 instructions, small disk footprint (~14 MB), zero licensing fee or copyleft risk.
2. **Recommended Detection Model Family**:
   - **Apache 2.0 Licensed Detectors** (such as **YOLOX-Nano**, **RT-DETR**, or **PP-PicoDet**): Offer clean commercial rights without AGPL copyleft strings.
   - If the user prefers YOLOv8n or YOLO11n for raw accuracy benchmarks, it must be noted that commercial deployment requires an Ultralytics enterprise license or internal open-source compliance under AGPL-3.0.
3. **Current Active Baseline**:
   - **Synthetic Waste Detector (Test Fixture)**: MIT License. Clearly separated from production claims, verified for end-to-end integration testing.

---

## 5. Mandatory Attribution & Notices

When distributing the application, the following notices must be retained:
- **ONNX Runtime**: MIT License (Copyright (c) Microsoft Corporation).
- **OpenCV**: Apache 2.0 License (Copyright (c) OpenCV team).
- **Pillow**: HPND License (Copyright (c) Jeffrey A. Clark).
- **NumPy**: BSD 3-Clause License (Copyright (c) NumPy Developers).


# Waste Detection System

A local, CPU-first computer vision and object tracking system for automated real-time waste detection, classification, and event logging across images, prerecorded video files, live webcams, and IP/RTSP streams.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [End-to-End Workflow](#end-to-end-workflow)
5. [Models and Computer Vision Technologies](#models-and-computer-vision-technologies)
6. [Detection and Tracking Logic](#detection-and-tracking-logic)
7. [Module Reference](#module-reference)
8. [Input Sources and Supported Formats](#input-sources-and-supported-formats)
9. [Configuration Reference](#configuration-reference)
10. [Data Storage and Detection Records](#data-storage-and-detection-records)
11. [Installation and Environment Setup](#installation-and-environment-setup)
12. [Running the Application](#running-the-application)
13. [Project Directory Structure](#project-directory-structure)
14. [Technology Stack](#technology-stack)
15. [Model Licensing and Usage Considerations](#model-licensing-and-usage-considerations)
16. [Known Limitations](#known-limitations)
17. [Troubleshooting](#troubleshooting)
18. [Testing and Development](#testing-and-development)

---

## Project Overview

The **Waste Detection System** is an offline-capable, terminal-driven computer vision application designed to identify and categorize discarded materials in complex, real-world scenes. The system is engineered to run **entirely locally on commodity CPU hardware**, eliminating dependencies on external cloud APIs, container runtimes (such as Docker), or dedicated database servers.

### Primary Purpose
Automate the identification and spatial tracking of waste items to support domestic, commercial, and industrial waste sorting workflows without exposing visual streams to remote cloud services.

### Main Capabilities
- **Local Neural Inference:** Detects and classifies waste items using a pretrained YOLO11 medium (`yolo11m`) checkpoint.
- **Secondary Non-Waste Verification:** Filters out false positives caused by humans, domestic animals, furniture, or vehicles using a secondary COCO detector (`yolov8n`), while retaining genuine waste held in hands or placed on surfaces.
- **Continuous Multi-Frame Tracking:** Maintains continuous object identity across video frames using ByteTrack.
- **Track Confirmation & Class Stabilization:** Prevents transient 1–2 frame false detections through multi-frame confirmation, and stabilizes category predictions over time using confidence-weighted exponential moving average (EMA) voting.
- **Zero-Redundancy Persistence:** Appends structured unique detection events to a single project-root CSV file (`waste_detections.csv`), preventing duplicate records for continuing objects.
- **Visual & Headless Modes:** Features an interactive OpenCV graphical preview window with color-coded bounding boxes and terminal telemetry, as well as a fully headless execution mode for low-power or server environments.

---

## Key Features

| Capability | Status | Implementation Details |
| :--- | :---: | :--- |
| **Waste Detection** | Implemented | Pretrained YOLO11m detector targeting 3 native waste classes. |
| **Waste Classification** | Implemented | Classifies items into `organic`, `recyclable`, or `residual`. |
| **Still Image Processing** | Implemented | Single-frame inference via `--source-image` with clean terminal summary and optional preview. |
| **Prerecorded Video Processing** | Implemented | File decoding via `--source <file.mp4>` with media-time track lifecycle management. |
| **USB / Integrated Webcam** | Implemented | Live video capture via `--source webcam` with threaded capture loop and bounded timeout. |
| **IP / RTSP Network Streams** | Implemented | Network camera ingestion via `--source rtsp --rtsp-url <url>` with auto-reconnect backoff. |
| **ByteTrack Tracking** | Implemented | Track assignment, velocity estimation, and spatial matching via Ultralytics ByteTrack. |
| **Secondary Verifier** | Implemented | COCO-based spatial overlap filter rejecting human/animal/furniture false positives. |
| **Track Confirmation** | Implemented | Requires objects to persist across `>= 3` frames before logging events for video/camera feeds. |
| **EMA Class Stabilization** | Implemented | Suppresses rapid per-frame classification flicker via decayed confidence voting. |
| **Duplicate Event Prevention** | Implemented | Only unique detection events are persisted; ongoing tracks update in memory. |
| **Single-File CSV Persistence** | Implemented | Append-only logging to root `waste_detections.csv` with permission-retry buffers. |
| **Live Stream Recording** | Implemented | Optional continuous MP4 session recording (`--record`) with graceful frame flushing. |
| **OpenCV Visual Preview** | Implemented | Interactive display window (`q` to quit, `space` to pause/resume, `s` to snapshot). |
| **Headless Execution** | Implemented | `--no-preview` flag allows operation in environments without an active display server. |
| **CPU-First Execution** | Implemented | Defaults to CPU; auto-detects CUDA acceleration when explicitly requested (`--device cuda`). |

---

## System Architecture

The application follows a modular, decoupled architecture organized under the `app/` package, separating hardware acquisition, neural inference, tracking state, visual rendering, and file persistence.

```mermaid
graph TD
    subgraph Input_Layer ["Input Acquisition Layer (app/sources.py)"]
        Webcam["WebcamSource (Threaded & Bounded)"]
        Video["VideoFileSource (Frame-Accurate)"]
        RTSP["RTSPCameraSource (Auto-Reconnect)"]
        Image["ImageFileSource (Single-Frame)"]
    end

    subgraph Config_Layer ["Configuration & Models (app/config.py, app/model_loader.py)"]
        AppConfig["AppConfig (JSON & Defaults)"]
        ModelLoader["Model Loader (SHA-256 Checksum)"]
        WeightsPrimary["Primary Checkpoint: YOLO11m"]
        WeightsVerifier["Secondary Verifier: YOLO8n"]
    end

    subgraph Pipeline_Layer ["Inference & Tracking Core (app/detector.py, app/tracking.py)"]
        PrimaryDetector["WasteDetector (YOLO11m)"]
        NonWasteVerifier["NonWasteVerifier (COCO Overlap)"]
        ByteTrack["ByteTrack (Track ID Assignment)"]
        TrackBuffer["_PendingTrack Buffer (Confirmation >= 3)"]
        ClassStabilizer["Class Stabilizer (EMA Score Voting)"]
        Tracker["DetectionTracker (Active Tracks State)"]
    end

    subgraph Output_Layer ["Output & Persistence (app/storage.py, app/visualizer.py, app/recorder.py)"]
        Storage["StorageManager (Thread-Safe Append)"]
        CSV[("waste_detections.csv")]
        Visualizer["VisualPreviewWindow (OpenCV GUI)"]
        Recorder["VideoRecorder (MP4 Writer)"]
    end

    Webcam -->|Raw Frame| PrimaryDetector
    Video -->|Raw Frame| PrimaryDetector
    RTSP -->|Raw Frame| PrimaryDetector
    Image -->|Raw Frame| PrimaryDetector

    AppConfig -.-> PrimaryDetector
    AppConfig -.-> Tracker
    ModelLoader --> WeightsPrimary --> PrimaryDetector
    ModelLoader --> WeightsVerifier --> NonWasteVerifier

    PrimaryDetector -->|Raw Detections| NonWasteVerifier
    NonWasteVerifier -->|Accepted Candidates| ByteTrack
    ByteTrack --> TrackBuffer
    TrackBuffer -->|Confirmed >= 3 Frames| ClassStabilizer
    ClassStabilizer --> Tracker

    Tracker -->|New Unique Event| Storage
    Storage -->|Append Row| CSV
    Tracker -->|Annotated State| Visualizer
    Webcam -.->|Optional Stream| Recorder
```

---

## End-to-End Workflow

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Main as main.py
    participant Source as app.sources
    participant Detector as app.detector (Primary)
    participant Verifier as app.detector (Verifier)
    participant Tracker as app.tracking
    participant Storage as app.storage
    participant Visual as app.visualizer

    User->>Main: Execute CLI command (e.g., --source video.mp4)
    Main->>Main: Load AppConfig & verify weights checksum
    Main->>Detector: Initialize WasteDetector (load YOLO11m)
    Main->>Verifier: Initialize NonWasteVerifier (load YOLO8n)
    Main->>Source: Open input adapter (Video / Webcam / RTSP / Image)
    Main->>Storage: Initialize StorageManager (bind waste_detections.csv)
    Main->>Tracker: Initialize DetectionTracker (set media FPS & confirmation)

    loop Frame Processing Loop
        Source->>Main: Read frame (RGB NumPy array)
        Main->>Detector: detector.track(frame)
        Detector-->>Main: raw_candidates (boxes, classes, confidences, track_ids)
        Main->>Verifier: verify_candidates(frame, raw_candidates)
        Verifier-->>Main: accepted_detections (filter out human/animal/furniture false positives)
        Main->>Tracker: process_frame(accepted_detections, frame, frame_idx)
        alt Track newly confirmed (hits >= min_confirmation_frames)
            Tracker->>Storage: record_event(DetectionEvent)
            Storage->>Storage: Append row to waste_detections.csv
        else Continuing active track
            Tracker->>Tracker: Update EMA class scores & spatial box in memory
        end
        opt Visual Preview Enabled
            Main->>Visual: render(frame, active_tracks, telemetry)
            Visual-->>User: Display OpenCV window (handle key events: q, space, s)
        end
    end

    Main->>Storage: finalize_session(duration, avg_fps)
    Main->>Source: close()
    Main-->>User: Print final terminal summary and latency breakdown
```

### Processing Stages:
1. **Startup & Configuration:** `AppConfig` resolves settings from `config/default_config.json` with fallback defaults.
2. **Model Loading & Checksum Validation:** `app.model_loader.load_waste_model` validates the SHA-256 hash of `models/pretrained/waste_detection_yolo11m.pt` (`9f2d19a7ce386...`) and initializes the neural weights on the requested hardware device.
3. **Input Adaptation:** `app.sources` instantiates the appropriate adapter (`WebcamSource`, `VideoFileSource`, `RTSPCameraSource`, or `ImageFileSource`), normalizing frame rates and resolutions.
4. **Primary Inference:** YOLO11m performs detection (or tracking) at the configured inference resolution (default `640x640`).
5. **Secondary Verification:** `NonWasteVerifier` inspects candidate bounding boxes against secondary COCO detections. Candidates overlapping detected humans, animals, or furniture are evaluated using Intersection over Union (IoU), Intersection over Area (IoA), and relative area ratios to filter out false alarms while preserving held waste.
6. **Track Confirmation & Stabilization:** `DetectionTracker` buffers new tracks in a pending queue. Once an object persists across `>= 3` frames (for multi-frame inputs), a `DetectionEvent` is created. Per-frame classification flicker is resolved through confidence-weighted EMA scores.
7. **Single-File Logging:** When an event is confirmed, `StorageManager` atomically appends a record to `waste_detections.csv`. If the CSV file is temporarily locked (e.g., opened in Excel), events are buffered in memory and retried automatically.
8. **Teardown & Metrics:** Upon exit, camera handles and video writers are cleanly released, session summary metrics are displayed, and internal tracker states are reset.

---

## Models and Computer Vision Technologies

The system employs a dual-stage vision architecture:

### 1. Primary Waste Detector
- **Model Name:** `khanhromvn/waste-detection-yolo11`
- **Architecture:** Ultralytics YOLO11 Medium (`yolo11m.pt`)
- **Checkpoint Path:** `models/pretrained/waste_detection_yolo11m.pt`
- **SHA-256 Digest:** `9f2d19a7ce386c6dbb95a725e6197f7b6fb699cd601bbeca9407034396f2ad6a`
- **Input Dimensions:** `3 x 640 x 640` (configurable via `--imgsz`)
- **Default Confidence Threshold:** `0.25`
- **Native Classes (3):**
  - `0`: `organic` (food waste, fruit peels, compostable matter)
  - `1`: `recyclable` (rigid plastic bottles, clean cardboard, clean paper, metal cans)
  - `2`: `residual` (film plastic shopping bags, dirty packaging wrappers, general non-recyclable discards)
- **Framework:** PyTorch / Ultralytics
- **Default Device:** `cpu` (supports `cuda` if an NVIDIA GPU is available)

### 2. Secondary Non-Waste Verifier
- **Model Name:** COCO Pretrained YOLOv8 Nano (`yolov8n`)
- **Architecture:** Ultralytics YOLOv8 Nano (`yolov8n.pt`)
- **Checkpoint Path:** Auto-resolved from `models/pretrained/yolov8n.pt` or local cache
- **Input Dimensions:** `3 x 416 x 416` (optimized for high-speed secondary filtering)
- **Default Confidence Threshold:** `0.35`
- **Target Filter Categories:**
  - `Biological / Non-Waste`: `person`, `cat`, `dog`, `bird`, `horse`, `sheep`, `cow`, `elephant`, `bear`, `zebra`, `giraffe`
  - `Corroborating Items`: `bottle`, `cup`, `wine glass`, `bowl`, `banana`, `apple`, `sandwich`, `book`, etc.
  - `Furniture & Vehicles`: `chair`, `couch`, `bed`, `dining table`, `toilet`, `tv`, `laptop`, `car`, `bus`, `bicycle`, etc.
- **Role:** Eliminates background false positives (e.g., a human limb or couch corner detected as trash) without suppressing held or surface-placed waste.

---

## Detection and Tracking Logic

### Spatial Overlap Decision Engine
In `app/detector.py`, each candidate waste box ($B_{\text{cand}}$) is compared against detected non-waste boxes ($B_{\text{nw}}$) using three geometric metrics:

1. **Intersection over Union (IoU):**
   $$\text{IoU} = \frac{\text{Area}(B_{\text{cand}} \cap B_{\text{nw}})}{\text{Area}(B_{\text{cand}} \cup B_{\text{nw}})}$$
2. **Intersection over Area (IoA / Inclusion Ratio):**
   $$\text{IoA} = \frac{\text{Area}(B_{\text{cand}} \cap B_{\text{nw}})}{\text{Area}(B_{\text{cand}})}$$
3. **Relative Area Ratio:**
   $$\text{RelArea} = \frac{\text{Area}(B_{\text{cand}})}{\text{Area}(B_{\text{nw}})}$$

#### Verification Decision Rules:
- **Held Waste Preservation:** If a waste candidate has high confidence ($\ge 0.80$) and overlaps a person box whose confidence is lower ($< 0.80$), the waste item is **ACCEPTED** (protects plastic bags or bottles held in hands).
- **Strong Person / Animal Overlap:** If $\text{IoU} \ge 0.40$, $\text{IoA} \ge 0.60$, and $\text{RelArea} \ge 0.25$, the detection is **REJECTED** (prevents entire people from being logged as waste).
- **Body Part / Enclosed Detection:** If $\text{IoA} \ge 0.85$, $\text{RelArea} \ge 0.25$, and candidate confidence is low ($< 0.65$), the candidate is **REJECTED** as an uncorroborated limb or clothing false positive.
- **Furniture / Structural Matches:** If a candidate overlaps a detected table, chair, or bed with $\text{IoU} \ge 0.55$ and $\text{IoA} \ge 0.75$, it is **REJECTED** as a non-waste structure match (unless primary confidence exceeds the furniture confidence).

### Multi-Frame Track Confirmation
To eliminate transient 1-frame noise from video or camera feeds:
- Newly observed track IDs are placed into a `_PendingTrack` buffer.
- An event is emitted and written to CSV **only after the track has been detected in at least 3 distinct frames** (`min_confirmation_frames=3`).
- Still image processing bypasses this buffer (`min_confirmation_frames=1`), ensuring immediate reporting.

### Confidence-Weighted EMA Class Stabilization
To prevent an item from flickering between classes (e.g., flipping between `residual` and `recyclable` due to lighting variations):
- The tracker maintains a running decayed score for each class:
  $$S_{\text{class}}(t) = S_{\text{class}}(t-1) \times 0.97 + \text{Confidence}_{\text{det}}$$
- The stabilized class is assigned to the category with the highest cumulative score.
- Raw per-frame model outputs are preserved in `TrackState.raw_class_counts` for auditability.

### Media-Time Track Expiration
For prerecorded video files, track timeouts are evaluated using **video media timestamp** rather than wall-clock time:
$$\Delta t_{\text{elapsed}} = \frac{\text{frame\_index} - \text{last\_seen\_frame}}{\text{native\_fps}}$$
This prevents tracks from prematurely expiring during compute-heavy CPU processing where inference FPS may be lower than video capture FPS.

---

## Module Reference

| Module / File | Responsibility | Key Classes & Functions |
| :--- | :--- | :--- |
| `main.py` | CLI entry point, execution orchestration, and telemetry display. | `main()`, `run_detection()`, `parse_args()` |
| `app/config.py` | Configuration schema loading, path resolution, and default fallbacks. | `AppConfig`, `sanitize_source_reference()` |
| `app/model_loader.py` | Checkpoint resolution, SHA-256 verification, and cached YOLO model instantiation. | `load_waste_model()`, `_verify_sha256()`, `_download_model_weights()` |
| `app/detector.py` | Primary YOLO11m detector, secondary COCO verifier, and spatial overlap logic. | `WasteDetector`, `NonWasteVerifier`, `compute_box_overlap()`, `Detection` |
| `app/tracking.py` | ByteTrack integration, confirmation buffering, class stabilization, and track state. | `DetectionTracker`, `TrackState`, `_PendingTrack`, `compute_iou()` |
| `app/storage.py` | Atomic CSV persistence, permission-retry buffer, and session event deduplication. | `StorageManager`, `DetectionEvent`, `CSV_HEADERS` |
| `app/sources.py` | Hardware abstraction for webcams, prerecorded videos, RTSP streams, and images. | `BaseSource`, `WebcamSource`, `VideoFileSource`, `RTSPCameraSource`, `ImageFileSource` |
| `app/visualizer.py` | OpenCV preview window, color-coded bounding boxes, and diagnostic overlay. | `VisualPreviewWindow`, `format_display_label()`, `CLASS_COLORS` |
| `app/recorder.py` | Local MP4 video recording for live camera feeds with background worker. | `VideoRecorder` |

---

## Input Sources and Supported Formats

### 1. Still Images
- **Supported Extensions:** `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tiff`
- **CLI Flag:** `--source-image <path/to/image.jpg>`
- **Behavior:** Processes a single frame, reports detected waste categories in the terminal, writes confirmed detections to CSV, and opens a preview window (unless `--no-preview` is passed).

### 2. Prerecorded Video Files
- **Supported Containers:** `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`, `.webm`, `.m4v`
- **CLI Flag:** `--source <path/to/video.mp4>`
- **Behavior:** Decodes frames sequentially at source resolution, evaluates media-time tracking, applies 3-frame confirmation, and logs summary metrics upon reaching EOF.

### 3. USB / Integrated Webcams
- **Supported Interfaces:** Standard DirectShow (`DSHOW`) and Media Foundation (`MSMF`) camera devices on Windows.
- **CLI Flag:** `--source webcam [--camera-id 0]`
- **Behavior:** Runs background capture in a dedicated thread to decouple camera hardware I/O from model inference. Implements a bounded 5.0-second read timeout to prevent terminal hangs.

### 4. IP / RTSP Camera Streams
- **Supported Protocols:** `rtsp://`, `http://`, `https://`
- **CLI Flag:** `--source rtsp --rtsp-url "rtsp://user:pass@192.168.1.50:554/stream"` (or via `WASTE_RTSP_URL` environment variable).
- **Behavior:** Sanitizes credentials in logs/CSV, and uses exponential backoff reconnection if the network link drops.

---

## Configuration Reference

Settings are defined in `config/default_config.json` and accessed via `app/config.py`:

```json
{
  "detection": {
    "confidence_threshold": 0.25,
    "image_size": 640
  },
  "verifier": {
    "enabled": true,
    "confidence_threshold": 0.35,
    "weights_path": null
  },
  "tracking": {
    "iou_threshold": 0.35,
    "lost_track_timeout_seconds": 3.0,
    "min_confirmation_frames": 3
  },
  "storage": {
    "csv_filename": "waste_detections.csv"
  },
  "camera": {
    "default_camera_index": 0,
    "read_timeout_seconds": 5.0,
    "max_reconnect_attempts": 5,
    "reconnect_initial_delay_sec": 2.0
  }
}
```

### Configuration Options:
- `confidence_threshold` (float, default `0.25`): Minimum confidence required for primary waste candidates.
- `image_size` (int, default `640`): Square dimension to which input frames are scaled for YOLO11 inference.
- `verifier.enabled` (bool, default `true`): Enables or disables secondary non-waste filtering.
- `verifier.confidence_threshold` (float, default `0.35`): Confidence floor for secondary COCO detections.
- `tracking.iou_threshold` (float, default `0.35`): Spatial overlap required to associate fallback detections.
- `tracking.lost_track_timeout_seconds` (float, default `3.0`): Elapsed time without detection before a track is retired.
- `tracking.min_confirmation_frames` (int, default `3`): Consecutive/recent frame appearances required to confirm an event.
- `camera.read_timeout_seconds` (float, default `5.0`): Maximum time allowed for webcam hardware frame capture before raising a timeout.

---

## Data Storage and Detection Records

Detection records are persisted to a single project-root CSV file (`waste_detections.csv`).

### CSV Schema

| Column Index | Field Name | Type | Description | Example |
| :---: | :--- | :--- | :--- | :--- |
| 1 | `record_id` | String | Unique record identifier (`rec_<timestamp>_<hex>`). | `rec_20261113_104533_8d2bfc` |
| 2 | `session_id` | String | Identifier for the running CLI session. | `session_20261113_104506` |
| 3 | `event_id` | String | Identifier for the confirmed detection event. | `evt_20261113_104533_8d2bfc` |
| 4 | `track_id` | Integer | ByteTrack assigned tracking ID. | `1` |
| 5 | `detected_at` | String | ISO 8601 formatted timestamp with timezone offset. | `2026-11-13T10:45:33.688210+05:30` |
| 6 | `source_type` | String | Input type (`webcam`, `video_file`, `rtsp`, `image`). | `video_file` |
| 7 | `source_name` | String | Sanitized filename or camera index (credentials stripped). | `sample_video.mp4` |
| 8 | `frame_number`| Integer | Video or session frame index where event was confirmed. | `6` |
| 9 | `class_id` | Integer | Zero-indexed category ID (`0`, `1`, `2`). | `2` |
| 10 | `class_name` | String | Category label (`organic`, `recyclable`, `residual`). | `residual` |
| 11 | `confidence` | Float | Model detection confidence score (4 decimal places). | `0.8570` |
| 12 | `bbox_x1` | Integer | Top-left bounding box X coordinate (pixels). | `60` |
| 13 | `bbox_y1` | Integer | Top-left bounding box Y coordinate (pixels). | `8` |
| 14 | `bbox_x2` | Integer | Bottom-right bounding box X coordinate (pixels). | `412` |
| 15 | `bbox_y2` | Integer | Bottom-right bounding box Y coordinate (pixels). | `624` |
| 16 | `model_name` | String | Model repository or identifier string. | `khanhromvn/waste-detection-yolo11` |
| 17 | `checkpoint_filename` | String | Checkpoint file used during inference. | `waste_detection_yolo11m.pt` |

### Persistence & Deduplication Guarantees:
- **Zero Duplicate Rows for Continuing Tracks:** An object tracked across hundreds of consecutive frames generates **only 1 CSV record** upon confirmation.
- **In-Memory State Synchronization:** While an object remains in the scene, its peak confidence and latest bounding box update in memory without writing duplicate lines to disk.
- **Lock Contention Protection:** If `waste_detections.csv` is locked by external software (such as Microsoft Excel), events are queued in an internal thread-safe buffer and flushed automatically once file access is restored.

---

## Installation and Environment Setup

The application is tested on Windows 10/11 using Python 3.10 through 3.12.

### Step 1: Open PowerShell and Navigate to the Project
```powershell
cd C:\Users\A\OneDrive\Desktop\Waste_detection
```

### Step 2: Create and Activate a Python Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

### Step 3: Install Required Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Primary Model Checkpoint
Ensure the primary YOLO11m weights file exists in `models/pretrained/`:
```powershell
Test-Path "models\pretrained\waste_detection_yolo11m.pt"
```
*(If missing, the system will attempt to download the verified checkpoint from Hugging Face Hub on first launch).*

---

## Running the Application

### Process a Still Image
```powershell
python main.py --source-image "path\to\photo.jpg"
```

### Process a Prerecorded Video File
```powershell
python main.py --source "path\to\video.mp4"
```

### Run Live Webcam Detection
```powershell
python main.py --source webcam --camera-id 0
```

### Connect to an IP / RTSP Camera Stream
```powershell
python main.py --source rtsp --rtsp-url "rtsp://192.168.1.100:554/live"
```

### Run in Headless Mode (No GUI Window)
```powershell
python main.py --source "path\to\video.mp4" --no-preview
```

### Adjust Confidence and Image Resolution
```powershell
python main.py --source webcam --conf 0.35 --imgsz 640
```

### Interactive Preview Controls
When the OpenCV preview window is active:
- **`Ctrl + C`**: Gracefully quit the application and finalize detection logging.
- **`SPACE`**: Pause / unpause playback.
- **`s`**: Save an annotated visual snapshot of the current frame to disk.

---

## Project Directory Structure

```
Waste_detection/
├── app/                              # Core application package
│   ├── __init__.py                   # Package initialization and logging setup
│   ├── config.py                     # Configuration loader and path resolver
│   ├── detector.py                   # YOLO11m waste detector & secondary verifier
│   ├── model_loader.py               # Model loading & SHA-256 verification
│   ├── recorder.py                   # Local video recording adapter
│   ├── sources.py                    # Input source adapters (Webcam, Video, RTSP, Image)
│   ├── storage.py                    # CSV storage engine and event deduplication
│   ├── tracking.py                   # ByteTrack tracking, confirmation & stabilization
│   └── visualizer.py                 # OpenCV GUI rendering and HUD annotations
├── config/
│   └── default_config.json           # Default configuration settings
├── models/
│   ├── registry.json                 # Model metadata registry
│   └── pretrained/
│       └── waste_detection_yolo11m.pt# Pretrained primary YOLO11m checkpoint
├── tests/
│   └── test_waste_system.py          # Comprehensive unit and integration test suite
├── training/                         # Offline utilities for dataset audit and training
│   ├── benchmark.py
│   ├── evaluate.py
│   ├── export_onnx.py
│   ├── train.py
│   └── validate_dataset.py
├── main.py                           # CLI entry point
├── MODEL_LICENSES.md                 # Commercial feasibility and license audit
├── requirements.txt                  # Python dependency specifications
└── waste_detections.csv              # Persistent detection records (single project CSV)
```

---

## Technology Stack

- **Runtime Environment:** Python 3.10+
- **Deep Learning Framework:** PyTorch (`torch`, `torchvision`)
- **Computer Vision Engine:** OpenCV (`opencv-python`)
- **Model Architecture & Tracking:** Ultralytics YOLO11 (`ultralytics`)
- **Object Association & Solvers:** `lap` (Linear Assignment Problem solver for ByteTrack)
- **Mathematical Processing:** NumPy (`numpy`)
- **Model Distribution:** Hugging Face Hub (`huggingface-hub`)
- **Data Persistence:** Standard Library `csv` engine with thread-safe file handling

---

## Model Licensing and Usage Considerations

Review `MODEL_LICENSES.md` for complete commercial compliance analysis.

- **Application Source Code:** Released under the project's internal terms.
- **Ultralytics Framework & Weights:** Distributed under the **GNU Affero General Public License v3 (AGPL-3.0)**. Commercial proprietary deployment without source disclosure requires an enterprise commercial license from Ultralytics.
- **Pretrained Weights (`waste_detection_yolo11m.pt`):** Originates from community fine-tuning of YOLO11m on waste sorting datasets (`khanhromvn/waste-detection-yolo11`).
- **Secondary Verifier Weights (`yolov8n.pt`):** COCO-pretrained weights under AGPL-3.0.
- **Commercial Migration Path:** For deployments requiring permissive commercial licensing (e.g., Apache 2.0 or MIT), refer to the ONNX export tooling (`training/export_onnx.py`) and architecture migration guide in `MODEL_LICENSES.md`.

---

## Known Limitations

1. **CPU Inference Latency:** Running YOLO11 medium (`yolo11m`) alongside secondary verification on CPU averages ~600–750 ms per frame on quad-core CPUs (~1.3–1.6 FPS). For real-time 30 FPS camera tracking, an acceleration device (CUDA GPU) or export to INT8/ONNX is recommended.
2. **Context Ambiguity:** Thin, translucent plastic shopping bags held by a person are detected as `residual` material based on physical appearance. The visual model cannot infer whether the person intends to keep or discard the item.
3. **Background False Positives:** Highly textured background patterns (such as intricate carpets, wood grain, or dense foliage) may occasionally trigger transient low-confidence waste predictions. The 3-frame confirmation filter significantly reduces, but does not entirely eliminate, static background false positives.
4. **Class Flickering on Edge Categories:** Boundary items (such as coated paper cups or plastic-wrapped cardboard) can generate fluctuating predictions between `organic`, `recyclable`, and `residual`. While EMA smoothing stabilizes tracked video instances, single still images reflect raw model predictions.

---

## Troubleshooting

### Camera Fails to Open
- **Symptom:** `ERROR: Could not open camera device 0`.
- **Solution:** Verify that no other application (e.g., Zoom, Teams, Windows Camera) is using the webcam. Try specifying `--camera-id 1` if multiple video capture devices are connected.

### Camera Frame Read Timeout
- **Symptom:** `Timeout reading from webcam 0 after 5.0s`.
- **Solution:** Indicates camera hardware driver hang. The system terminates cleanly without corrupting CSV state. Reconnect the USB camera cable or restart the Windows Camera Frame Server service.

### CSV Permission / Locking Errors
- **Symptom:** Terminal reports `[STORAGE ERROR] CSV write failed: Permission denied`.
- **Solution:** Occurs when `waste_detections.csv` is opened exclusively in Microsoft Excel. Close the file in Excel. The application retains events in an in-memory queue and flushes them as soon as the file is unlocked.

### Low FPS / High Processing Latency
- **Symptom:** Playback appears slow during video processing.
- **Solution:** Reduce inference resolution using `--imgsz 416` or run headless with `--no-preview`. If an NVIDIA GPU is available, pass `--device cuda`.

---

## Testing and Development

The repository includes a comprehensive, non-destructive automated test suite in `tests/test_waste_system.py`. All tests run using temporary directories and mock fixtures, ensuring that production records in `waste_detections.csv` are never altered.

### Run All Unit and Integration Tests
```powershell
python -m unittest tests/test_waste_system.py -v
```

### Verified Test Coverage:
- **Model Loading & Checksum Integrity:** Validates SHA-256 weight verification, caching, and class mapping.
- **Secondary Verification Logic:** Tests human/animal false positive rejection, furniture exclusion, and held-waste retention.
- **Object Tracking & Deduplication:** Confirms ByteTrack assignment, spatial fallback, and media-time expiration.
- **Track Confirmation & EMA Class Smoothing:** Tests that transient 1–2 frame noise is ignored and that multi-frame class flicker is suppressed.
- **CSV Storage Reliability:** Validates schema compliance, locked-file handling, and memory retry queues.
- **Input Adapter Stability:** Tests webcam timeout boundaries and video file resolution handling.
- **Visual Presentation Rules:** Validates display label formatting (`WASTE`) and frame immutability.
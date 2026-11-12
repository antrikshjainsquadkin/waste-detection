"""Waste Detection System application package.

Core modules:
- app.config: Application settings, thresholds, and taxonomies
- app.model_loader: Model downloading, SHA-256 verification, and YOLO loading
- app.detector: YOLO11 waste detector and tracker wrapper
- app.sources: Input sources (Webcam, RTSP, VideoFile, Image)
- app.recorder: Local session video recording
- app.tracking: Object tracking and duplicate prevention
- app.storage: CSV and JSON persistence, evidence preservation
"""

from __future__ import annotations

from typing import Any

from app.config import AppConfig, NATIVE_CLASSES, CLASS_EXAMPLES, sanitize_source_reference
from app.recorder import VideoRecorder
from app.sources import (
    BaseSource,
    WebcamSource,
    RtspSource,
    VideoFileSource,
    ImageSource,
    resolve_video_path,
    resolve_image_path,
    resolve_stream_url,
    SUPPORTED_VIDEO_EXTS,
    SUPPORTED_IMAGE_EXTS,
)
from app.storage import StorageManager, DetectionEvent
from app.tracking import DetectionTracker
from app.visualizer import VisualPreviewWindow, render_annotated_frame, get_class_color

_LAZY_DETECTOR_EXPORTS = {"WasteDetector", "Detection"}
_LAZY_MODEL_LOADER_EXPORTS = {
    "load_waste_model",
    "compute_sha256",
    "download_model",
    "EXPECTED_SHA256",
    "EXPECTED_CLASSES",
}


def __getattr__(name: str) -> Any:
    """Lazily load heavy PyTorch/YOLO symbols only when accessed."""
    if name in _LAZY_DETECTOR_EXPORTS:
        from app import detector as _detector
        return getattr(_detector, name)
    if name in _LAZY_MODEL_LOADER_EXPORTS:
        from app import model_loader as _model_loader
        return getattr(_model_loader, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AppConfig",
    "NATIVE_CLASSES",
    "CLASS_EXAMPLES",
    "sanitize_source_reference",
    "WasteDetector",
    "Detection",
    "load_waste_model",
    "compute_sha256",
    "download_model",
    "EXPECTED_SHA256",
    "EXPECTED_CLASSES",
    "VideoRecorder",
    "BaseSource",
    "WebcamSource",
    "RtspSource",
    "VideoFileSource",
    "ImageSource",
    "resolve_stream_url",
    "StorageManager",
    "DetectionEvent",
    "DetectionTracker",
    "VisualPreviewWindow",
    "render_annotated_frame",
    "get_class_color",
]


"""Application Configuration and Taxonomy Settings.

Defines native classes, detection thresholds, storage directories,
and input source configuration.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Native classes supported by khanhromvn/waste-detection-yolo11
NATIVE_CLASSES: Dict[int, str] = {
    0: "organic",
    1: "recyclable",
    2: "residual",
}

CLASS_EXAMPLES: Dict[str, str] = {
    "organic": "Food scraps, fruit, vegetables, bread, pizza, eggshells",
    "recyclable": "Plastic bottles, cans, glass bottles, cardboard, paper, metal lids",
    "residual": "Plastic bags, wrappers, packets, styrofoam, disposable cups, tissues, batteries",
}


def sanitize_source_reference(source: str) -> str:
    """Sanitize RTSP URLs or file paths by removing passwords/credentials."""
    if not source:
        return ""
    # Matches rtsp://user:pass@host:port/path
    return re.sub(r"://([^:@]+):([^@]+)@", r"://\1:***@", str(source))


class AppConfig:
    """Central configuration for the Waste Detection System."""

    def __init__(self, config_path: Optional[Path] = None):
        self.workspace_root = Path(__file__).resolve().parent.parent
        self.config_path = config_path or (self.workspace_root / "config" / "default_config.json")
        self._data: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        """Load configuration from JSON or initialize defaults."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception as e:
                logger.error("Failed to load config %s: %s. Using defaults.", self.config_path, e)
                self._data = {}
        else:
            self._data = {}

    @property
    def confidence_threshold(self) -> float:
        """Default starting confidence threshold (recommended 0.25)."""
        det = self._data.get("detection", {})
        return float(det.get("confidence_threshold", 0.25))

    @property
    def image_size(self) -> int:
        """Inference image size (recommended 640)."""
        det = self._data.get("detection", {})
        return int(det.get("image_size", 640))

    @property
    def iou_threshold(self) -> float:
        """IoU threshold for object tracking deduplication."""
        trk = self._data.get("tracking", {})
        return float(trk.get("iou_threshold", 0.35))

    @property
    def lost_track_timeout_seconds(self) -> float:
        """Seconds before an inactive track is finalized."""
        trk = self._data.get("tracking", {})
        return float(trk.get("lost_track_timeout_seconds", 3.0))

    @property
    def csv_path(self) -> Path:
        """Single persistent detections CSV file in project root."""
        st = self._data.get("storage", {})
        filename = st.get("csv_filename", "waste_detections.csv")
        return self.workspace_root / filename

    @property
    def recordings_dir(self) -> Path:
        """Directory where session recordings and data are saved (not auto-created)."""
        st = self._data.get("storage", {})
        rel = st.get("recordings_dir", "recordings")
        return self.workspace_root / rel

    @property
    def save_sessions_to_disk(self) -> bool:
        """Whether to persist CSV, JSON, and evidence images to disk."""
        st = self._data.get("storage", {})
        return bool(st.get("save_sessions_to_disk", False))

    @property
    def models_dir(self) -> Path:
        """Directory where model checkpoints are stored."""
        return self.workspace_root / "models" / "pretrained"

    @property
    def verifier_enabled(self) -> bool:
        """Whether the secondary non-waste verifier is enabled."""
        ver = self._data.get("verifier", {})
        return bool(ver.get("enabled", True))

    @property
    def verifier_weights_path(self) -> Optional[Path]:
        """Path to secondary non-waste verifier model checkpoint."""
        ver = self._data.get("verifier", {})
        custom_path = ver.get("weights_path")
        if custom_path:
            return Path(custom_path)
        return None

    @property
    def verifier_confidence(self) -> float:
        """Confidence threshold for secondary verifier detections."""
        ver = self._data.get("verifier", {})
        return float(ver.get("confidence_threshold", 0.35))

    @property
    def record_live_sessions(self) -> bool:
        """Whether to record video for live camera streams."""
        rec = self._data.get("recording", {})
        return bool(rec.get("record_live_sessions", False))

    @property
    def camera_index(self) -> int:
        cam = self._data.get("camera", {})
        return int(cam.get("default_camera_index", 0))

    @property
    def camera_read_timeout_seconds(self) -> float:
        """Bounded timeout in seconds for webcam frame reads."""
        cam = self._data.get("camera", {})
        return float(cam.get("read_timeout_seconds", 5.0))

    @property
    def rtsp_url(self) -> str:
        """RTSP URL from environment variable or config."""
        env_url = os.getenv("WASTE_RTSP_URL", "")
        if env_url:
            return env_url
        cam = self._data.get("camera", {})
        return str(cam.get("rtsp_url", ""))

    @property
    def max_reconnect_attempts(self) -> int:
        cam = self._data.get("camera", {})
        return int(cam.get("max_reconnect_attempts", 5))

    @property
    def reconnect_initial_delay_sec(self) -> float:
        cam = self._data.get("camera", {})
        return float(cam.get("reconnect_initial_delay_sec", 2.0))

    @property
    def reconnect_max_delay_sec(self) -> float:
        cam = self._data.get("camera", {})
        return float(cam.get("reconnect_max_delay_sec", 15.0))


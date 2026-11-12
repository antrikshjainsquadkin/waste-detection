"""Configuration manager for the Waste Detection System.

Handles loading, validating, persisting and querying application settings,
class taxonomies, category-specific confidence thresholds, and hardware profiles.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Mandatory 8 Waste Categories as approved
DEFAULT_CATEGORIES: List[Dict[str, Any]] = [
    {"id": 0, "internal_name": "paper", "display_name": "Paper", "default_confidence": 0.45},
    {"id": 1, "internal_name": "cardboard", "display_name": "Cardboard", "default_confidence": 0.45},
    {"id": 2, "internal_name": "plastic_bottle", "display_name": "Plastic Bottle", "default_confidence": 0.45},
    {"id": 3, "internal_name": "polythene_bag", "display_name": "Polythene Bag", "default_confidence": 0.45},
    {"id": 4, "internal_name": "metal", "display_name": "Metal", "default_confidence": 0.45},
    {"id": 5, "internal_name": "glass", "display_name": "Glass", "default_confidence": 0.45},
    {"id": 6, "internal_name": "organic_waste", "display_name": "Organic Waste", "default_confidence": 0.45},
    {"id": 7, "internal_name": "general_trash", "display_name": "General Trash", "default_confidence": 0.45},
]


def sanitize_source_reference(source: str) -> str:
    """Remove embedded passwords/credentials from RTSP or camera URLs for safe logging/CSV."""
    if not source:
        return ""
    # Matches rtsp://user:pass@host:port/path
    return re.sub(r"://([^:@]+):([^@]+)@", r"://\1:***@", str(source))


class AppConfig:
    """Central application configuration provider."""

    def __init__(self, config_path: Optional[Path] = None):
        self.workspace_root = Path(__file__).resolve().parent.parent.parent
        self.config_path = config_path or (self.workspace_root / "config" / "default_config.json")
        self._raw_data: Dict[str, Any] = {}
        self._confidence_thresholds: Dict[str, float] = {}
        self.load()

    def load(self) -> None:
        """Load configuration from JSON file or fall back to safe defaults."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self._raw_data = json.load(f)
            except Exception as e:
                logger.error("Failed to parse config file %s: %s. Using default values.", self.config_path, e)
                self._raw_data = {}
        else:
            self._raw_data = {}

        # Initialize confidence thresholds
        categories = self._raw_data.get("categories", DEFAULT_CATEGORIES)
        for cat in categories:
            name = cat["internal_name"]
            self._confidence_thresholds[name] = float(cat.get("default_confidence", 0.45))

    def save(self) -> bool:
        """Persist current configuration to disk safely."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            # Update thresholds in raw data
            categories = self._raw_data.get("categories", DEFAULT_CATEGORIES)
            for cat in categories:
                name = cat["internal_name"]
                if name in self._confidence_thresholds:
                    cat["default_confidence"] = self._confidence_thresholds[name]
            self._raw_data["categories"] = categories

            temp_path = self.config_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self._raw_data, f, indent=2)
            temp_path.replace(self.config_path)
            return True
        except Exception as e:
            logger.error("Failed to save config to %s: %s", self.config_path, e)
            return False

    @property
    def categories(self) -> List[Dict[str, Any]]:
        return self._raw_data.get("categories", DEFAULT_CATEGORIES)

    @property
    def category_internal_names(self) -> List[str]:
        return [c["internal_name"] for c in self.categories]

    def get_category_display_name(self, internal_name: str) -> str:
        for c in self.categories:
            if c["internal_name"] == internal_name:
                return c["display_name"]
        if internal_name == "unknown":
            return "Unknown"
        return internal_name.replace("_", " ").title()

    def get_confidence_threshold(self, category_name: str) -> float:
        """Get minimum confidence threshold for a category."""
        return self._confidence_thresholds.get(category_name, 0.45)

    def set_confidence_threshold(self, category_name: str, threshold: float) -> None:
        """Set minimum confidence threshold for a category (bounded 0.05 to 0.95)."""
        clamped = max(0.05, min(0.95, float(threshold)))
        self._confidence_thresholds[category_name] = clamped

    def get_all_confidence_thresholds(self) -> Dict[str, float]:
        return dict(self._confidence_thresholds)

    @property
    def unknown_candidate_min_confidence(self) -> float:
        """Detections with confidence above this but below category threshold become Unknown."""
        unknown_cfg = self._raw_data.get("unknown_settings", {})
        return float(unknown_cfg.get("candidate_confidence_min", 0.25))

    @property
    def processing_resolution(self) -> Tuple[int, int]:
        perf = self._raw_data.get("performance", {})
        res = perf.get("processing_resolution", [416, 416])
        return int(res[0]), int(res[1])

    def set_processing_resolution(self, width: int, height: int) -> None:
        perf = self._raw_data.setdefault("performance", {})
        perf["processing_resolution"] = [int(width), int(height)]

    @property
    def frame_skip(self) -> int:
        perf = self._raw_data.get("performance", {})
        return int(perf.get("frame_skip", 2))

    def set_frame_skip(self, skip: int) -> None:
        perf = self._raw_data.setdefault("performance", {})
        perf["frame_skip"] = max(1, min(10, int(skip)))

    @property
    def adaptive_performance(self) -> bool:
        perf = self._raw_data.get("performance", {})
        return bool(perf.get("adaptive_performance", True))

    def set_adaptive_performance(self, enabled: bool) -> None:
        perf = self._raw_data.setdefault("performance", {})
        perf["adaptive_performance"] = bool(enabled)

    @property
    def lost_track_timeout_seconds(self) -> float:
        tracking = self._raw_data.get("tracking", {})
        return float(tracking.get("lost_track_timeout_seconds", 3.0))

    def set_lost_track_timeout_seconds(self, timeout: float) -> None:
        tracking = self._raw_data.setdefault("tracking", {})
        tracking["lost_track_timeout_seconds"] = max(0.5, min(30.0, float(timeout)))

    @property
    def iou_threshold(self) -> float:
        tracking = self._raw_data.get("tracking", {})
        return float(tracking.get("iou_threshold", 0.35))

    @property
    def recordings_dir(self) -> Path:
        st = self._raw_data.get("storage", {})
        rel = st.get("recordings_dir", "recordings")
        return self.workspace_root / rel

    @property
    def save_evidence_images(self) -> bool:
        st = self._raw_data.get("storage", {})
        return bool(st.get("save_evidence_images", True))

    @property
    def evidence_format(self) -> str:
        st = self._raw_data.get("storage", {})
        return str(st.get("evidence_format", "jpg"))

    @property
    def evidence_quality(self) -> int:
        st = self._raw_data.get("storage", {})
        return int(st.get("evidence_quality", 92))

    @property
    def auto_retention_enabled(self) -> bool:
        st = self._raw_data.get("storage", {})
        return bool(st.get("auto_retention_enabled", False))

    def set_auto_retention_enabled(self, enabled: bool) -> None:
        st = self._raw_data.setdefault("storage", {})
        st["auto_retention_enabled"] = bool(enabled)

    @property
    def retention_max_days(self) -> int:
        st = self._raw_data.get("storage", {})
        return int(st.get("retention_max_days", 30))

    @property
    def retention_max_storage_mb(self) -> int:
        st = self._raw_data.get("storage", {})
        return int(st.get("retention_max_storage_mb", 5000))

    @property
    def active_model_id(self) -> str:
        models_cfg = self._raw_data.get("models", {})
        return str(models_cfg.get("active_model_id", "mock_waste_detector_v1"))

    def set_active_model_id(self, model_id: str) -> None:
        models_cfg = self._raw_data.setdefault("models", {})
        models_cfg["active_model_id"] = str(model_id)

    @property
    def models_dir(self) -> Path:
        models_cfg = self._raw_data.get("models", {})
        rel = models_cfg.get("models_dir", "models")
        return self.workspace_root / rel

    @property
    def max_reconnect_attempts(self) -> int:
        cam = self._raw_data.get("camera", {})
        return int(cam.get("max_reconnect_attempts", 5))

    @property
    def reconnect_initial_delay_sec(self) -> float:
        cam = self._raw_data.get("camera", {})
        return float(cam.get("reconnect_initial_delay_sec", 2.0))

    @property
    def reconnect_max_delay_sec(self) -> float:
        cam = self._raw_data.get("camera", {})
        return float(cam.get("reconnect_max_delay_sec", 15.0))


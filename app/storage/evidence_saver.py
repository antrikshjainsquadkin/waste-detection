"""Evidence image storage subsystem.

Preserves full-resolution, unannotated source frames for audit trails
and forensic verification of detection events.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class EvidenceSaver:
    """Safely writes pristine source frames to disk as evidence files."""

    def __init__(self, evidence_dir: Path, image_format: str = "jpg", quality: int = 92):
        self.evidence_dir = evidence_dir
        self.image_format = image_format.lower().lstrip(".")
        self.quality = max(50, min(100, int(quality)))
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        try:
            self.evidence_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error("Failed creating evidence directory %s: %s", self.evidence_dir, e)

    def save_evidence(self, raw_frame: np.ndarray, event_id: str, category: str) -> str:
        """Save a pristine source frame without bounding boxes or annotations.

        Args:
            raw_frame: Original unmodified BGR numpy image.
            event_id: Unique event identifier.
            category: Category name of the detected object.

        Returns:
            Absolute or relative file path to the saved image, or empty string on failure.
        """
        if raw_frame is None or raw_frame.size == 0:
            logger.warning("Empty raw frame provided for event %s. Cannot save evidence.", event_id)
            return ""

        safe_cat = "".join(c for c in category if c.isalnum() or c in ("_", "-"))
        filename = f"{event_id}_{safe_cat}.{self.image_format}"
        target_path = self.evidence_dir / filename

        try:
            # OpenCV encode parameters
            encode_params = []
            if self.image_format in ("jpg", "jpeg"):
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
            elif self.image_format == "png":
                encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 4]

            success, buffer = cv2.imencode(f".{self.image_format}", raw_frame, encode_params)
            if not success:
                logger.error("cv2.imencode failed for evidence image %s", target_path)
                return ""

            with open(target_path, "wb") as f:
                f.write(buffer)

            logger.debug("Evidence saved: %s", target_path)
            return str(target_path)
        except Exception as e:
            logger.error("Failed writing evidence image to %s: %s", target_path, e)
            return ""


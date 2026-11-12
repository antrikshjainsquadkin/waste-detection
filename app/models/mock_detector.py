"""Synthetic Waste Detector - TEST FIXTURE ONLY.

IMPORTANT NOTICE:
This module is explicitly a test fixture designed solely for offline
automated testing, UI validation, and pipeline verification when hardware
weights are not present. It is NOT a real trained waste detector.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from app.core.types import BoundingBox, Detection
from app.models.base import BaseDetector

logger = logging.getLogger(__name__)


class MockWasteDetector(BaseDetector):
    """Deterministic synthetic detector fixture for automated and UI testing."""

    def __init__(
        self,
        model_name: str = "Synthetic Waste Detector (Test Fixture)",
        version: str = "1.0.0-test",
        class_mapping: Dict[int, str] = None,
    ):
        self.model_name = model_name
        self.version = version
        self.is_test_fixture = True
        self.loaded = False
        self._frame_count = 0

        self.class_mapping = class_mapping or {
            0: "paper",
            1: "cardboard",
            2: "plastic_bottle",
            3: "polythene_bag",
            4: "metal",
            5: "glass",
            6: "organic_waste",
            7: "general_trash",
        }

    def load(self) -> bool:
        self.loaded = True
        logger.info("Loaded %s (TEST FIXTURE)", self.model_name)
        return True

    def predict(
        self,
        frame: np.ndarray,
        confidence_thresholds: Dict[str, float],
        unknown_min_conf: float = 0.25,
    ) -> List[Detection]:
        """Produce deterministic mock detections for testing based on frame properties."""
        if not self.loaded or frame is None or frame.size == 0:
            return []

        self._frame_count += 1
        h, w = frame.shape[:2]
        detections: List[Detection] = []

        # Generate a predictable test detection every few frames for pipeline verification
        # e.g., a "plastic_bottle" moving slowly across the frame
        if self._frame_count % 3 == 0:
            # Deterministic bounding box in middle
            bx = int(w * 0.25)
            by = int(h * 0.25)
            bw = int(w * 0.20)
            bh = int(h * 0.30)

            cat_id = 2  # plastic_bottle
            cat_name = self.class_mapping.get(cat_id, "plastic_bottle")
            conf = 0.88
            thresh = confidence_thresholds.get(cat_name, 0.45)

            is_unknown = conf < thresh and conf >= unknown_min_conf
            display_name = "Unknown" if is_unknown else cat_name.replace("_", " ").title()

            if conf >= unknown_min_conf:
                detections.append(
                    Detection(
                        class_id=cat_id,
                        class_name=cat_name if not is_unknown else "unknown",
                        display_name=display_name,
                        confidence=conf,
                        bbox=BoundingBox(x=bx, y=by, width=bw, height=bh),
                        is_unknown=is_unknown,
                    )
                )

        return detections

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "architecture": "synthetic_fixture",
            "framework": "internal_fixture",
            "is_test_fixture": True,
            "license": "MIT",
            "description": "Deterministic test fixture. Not an actual trained model.",
        }

    def unload(self) -> None:
        self.loaded = False
        logger.info("Unloaded %s", self.model_name)


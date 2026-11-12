"""Abstract base class for object detection models.

Establishes a uniform interface decoupling the application pipeline from
specific model frameworks and runtime engines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from app.core.types import Detection


class BaseDetector(ABC):
    """Abstract interface for all computer vision detectors."""

    @abstractmethod
    def load(self) -> bool:
        """Initialize and load the model into memory. Returns True if successful."""
        pass

    @abstractmethod
    def predict(
        self,
        frame: np.ndarray,
        confidence_thresholds: Dict[str, float],
        unknown_min_conf: float = 0.25,
    ) -> List[Detection]:
        """Perform inference on a single BGR frame.

        Args:
            frame: Raw BGR image as a numpy array.
            confidence_thresholds: Mapping of category internal name to confidence threshold.
            unknown_min_conf: Lower bound below which candidates are discarded.

        Returns:
            List of filtered Detection instances.
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata dictionary."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Release runtime resources."""
        pass


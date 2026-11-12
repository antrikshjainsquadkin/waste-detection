"""Model Lifecycle and Version Registry Manager.

Validates model files, computes checksums, runs smoke tests on candidate models,
and provides auditable activation and rollback controls.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.config import AppConfig
from app.models.base import BaseDetector
from app.models.mock_detector import MockWasteDetector
from app.models.onnx_detector import OnnxWasteDetector

logger = logging.getLogger(__name__)


def compute_file_sha256(file_path: Path) -> str:
    """Calculate SHA256 checksum of a model file."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


class ModelManager:
    """Handles discovery, validation, activation, and rollback of object detection models."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.models_dir = config.models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.models_dir / "registry.json"

        self._active_detector: Optional[BaseDetector] = None
        self._previous_active_id: Optional[str] = None
        self._registry_data: Dict[str, Any] = {"active_model_id": "", "models": []}
        self.load_registry()

    def load_registry(self) -> None:
        """Load model registry from JSON."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    self._registry_data = json.load(f)
            except Exception as e:
                logger.error("Failed to parse model registry %s: %s", self.registry_path, e)

        # Ensure test fixture model is present if empty
        if not self._registry_data.get("models"):
            fixture_entry = {
                "model_id": "mock_waste_detector_v1",
                "model_name": "Synthetic Waste Detector (Test Fixture)",
                "version": "1.0.0-test",
                "architecture": "synthetic_fixture",
                "framework": "internal_fixture",
                "weights_file": "",
                "class_mapping": {str(i): c["internal_name"] for i, c in enumerate(self.config.categories)},
                "license": "MIT",
                "commercial_usable": True,
                "description": "Deterministic test fixture for offline pipeline verification and UI testing.",
                "created_at": "2026-11-03T16:00:00Z",
                "validated": True,
            }
            self._registry_data = {
                "active_model_id": "mock_waste_detector_v1",
                "models": [fixture_entry],
            }
            self._save_registry()

    def _save_registry(self) -> bool:
        """Write registry JSON to disk safely."""
        try:
            temp_path = self.registry_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self._registry_data, f, indent=2)
            temp_path.replace(self.registry_path)
            return True
        except Exception as e:
            logger.error("Failed saving model registry: %s", e)
            return False

    @property
    def active_model_id(self) -> str:
        return self._registry_data.get("active_model_id", "mock_waste_detector_v1")

    @property
    def registered_models(self) -> List[Dict[str, Any]]:
        return self._registry_data.get("models", [])

    def get_model_entry(self, model_id: str) -> Optional[Dict[str, Any]]:
        for m in self.registered_models:
            if m.get("model_id") == model_id:
                return m
        return None

    def validate_model_file(self, weights_path: Path) -> Tuple[bool, str]:
        """Verify model file integrity, readability and compatibility."""
        if not weights_path.exists():
            return False, f"Model file not found: {weights_path}"

        suffix = weights_path.suffix.lower()
        if suffix not in (".onnx",):
            return False, f"Unsupported model file extension: {suffix}. Only .onnx is supported for CPU inference."

        try:
            import onnxruntime as ort
            # Attempt to create an inference session as smoke test
            session = ort.InferenceSession(str(weights_path), providers=["CPUExecutionProvider"])
            inputs = session.get_inputs()
            if not inputs:
                return False, "Model has no inputs."
            return True, "Model file and ONNX structure successfully verified."
        except Exception as e:
            return False, f"Model validation failed: {e}"

    def register_model(
        self,
        model_id: str,
        model_name: str,
        version: str,
        weights_file: Path,
        architecture: str = "onnx_yolo_like",
        class_mapping: Optional[Dict[int, str]] = None,
        license_str: str = "Proprietary",
        commercial_usable: bool = True,
        description: str = "",
    ) -> Tuple[bool, str]:
        """Register a new model into the registry after validation."""
        valid, msg = self.validate_model_file(weights_file)
        if not valid:
            return False, f"Cannot register model: {msg}"

        sha256 = compute_file_sha256(weights_file)

        # Default class mapping to the 8 approved categories
        if class_mapping is None:
            c_map = {str(i): c["internal_name"] for i, c in enumerate(self.config.categories)}
        else:
            c_map = {str(k): str(v) for k, v in class_mapping.items()}

        new_entry = {
            "model_id": model_id,
            "model_name": model_name,
            "version": version,
            "architecture": architecture,
            "framework": "onnxruntime_cpu",
            "weights_file": str(weights_file.resolve()),
            "sha256": sha256,
            "class_mapping": c_map,
            "license": license_str,
            "commercial_usable": commercial_usable,
            "description": description,
            "validated": True,
        }

        # Check if already registered
        existing = [m for m in self.registered_models if m.get("model_id") == model_id]
        if existing:
            self._registry_data["models"] = [
                m if m.get("model_id") != model_id else new_entry for m in self.registered_models
            ]
        else:
            self._registry_data["models"].append(new_entry)

        self._save_registry()
        logger.info("Registered model %s (%s)", model_name, model_id)
        return True, "Model registered successfully."

    def activate_model(self, model_id: str) -> Tuple[bool, str]:
        """Activate a registered model after performing a safe smoke test."""
        entry = self.get_model_entry(model_id)
        if not entry:
            return False, f"Model ID '{model_id}' is not registered."

        # If it's the test fixture
        if entry.get("architecture") == "synthetic_fixture":
            detector = MockWasteDetector(
                model_name=entry.get("model_name", "Synthetic Waste Detector (Test Fixture)"),
                version=entry.get("version", "1.0.0-test"),
            )
            detector.load()
            if self._active_detector:
                self._active_detector.unload()
            if self.active_model_id != model_id:
                self._previous_active_id = self.active_model_id
            self._active_detector = detector
            self._registry_data["active_model_id"] = model_id
            self.config.set_active_model_id(model_id)
            self._save_registry()
            return True, "Synthetic Test Fixture activated."

        # Real ONNX Model
        weights_file = Path(entry.get("weights_file", ""))
        valid, msg = self.validate_model_file(weights_file)
        if not valid:
            return False, f"Activation aborted: {msg}"

        class_map = {int(k): v for k, v in entry.get("class_mapping", {}).items()}
        res = self.config.processing_resolution

        detector = OnnxWasteDetector(
            model_path=weights_file,
            model_name=entry.get("model_name", model_id),
            version=entry.get("version", "1.0.0"),
            class_mapping=class_map,
            input_size=res,
        )

        if not detector.load():
            return False, "Failed to load model into ONNX Runtime."

        # Safe smoke test with blank frame
        dummy_frame = np.zeros((res[1], res[0], 3), dtype=np.uint8)
        try:
            detector.predict(dummy_frame, self.config.get_all_confidence_thresholds())
        except Exception as e:
            detector.unload()
            return False, f"Model smoke test failed on dummy input: {e}"

        # Unload current and switch
        if self._active_detector:
            self._active_detector.unload()

        if self.active_model_id != model_id:
            self._previous_active_id = self.active_model_id
        self._active_detector = detector
        self._registry_data["active_model_id"] = model_id
        self.config.set_active_model_id(model_id)
        self._save_registry()

        logger.info("Successfully activated model: %s", model_id)
        return True, f"Model '{entry.get('model_name')}' activated successfully."

    def rollback(self) -> Tuple[bool, str]:
        """Roll back to the previous active model."""
        if not self._previous_active_id:
            return False, "No previous model in history to roll back to."

        target = self._previous_active_id
        logger.info("Rolling back model to: %s", target)
        return self.activate_model(target)

    def get_active_detector(self) -> BaseDetector:
        """Return currently loaded active detector instance (initializes default if none loaded)."""
        if self._active_detector is None:
            active_id = self.active_model_id
            success, msg = self.activate_model(active_id)
            if not success:
                logger.warning("Could not activate '%s': %s. Falling back to test fixture.", active_id, msg)
                self.activate_model("mock_waste_detector_v1")
        return self._active_detector

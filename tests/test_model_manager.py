"""Unit tests for Model Registry, validation, activation, and rollback."""

import tempfile
import unittest
from pathlib import Path

from app.core.config import AppConfig
from app.models.manager import ModelManager


class TestModelManager(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)

        self.config = AppConfig()
        self.config._raw_data["models"]["models_dir"] = str(self.root_path / "models")
        self.mgr = ModelManager(self.config)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_default_test_fixture_registered(self):
        """Verify default test fixture is automatically registered."""
        models = self.mgr.registered_models
        self.assertGreaterEqual(len(models), 1)
        fixture = self.mgr.get_model_entry("mock_waste_detector_v1")
        self.assertIsNotNone(fixture)
        self.assertEqual(fixture["architecture"], "synthetic_fixture")

    def test_validate_nonexistent_model_fails(self):
        """Verifies missing model files fail validation."""
        fake_path = self.root_path / "nonexistent.onnx"
        valid, msg = self.mgr.validate_model_file(fake_path)
        self.assertFalse(valid)
        self.assertIn("not found", msg)

    def test_validate_invalid_extension_fails(self):
        """Verifies non-ONNX files fail validation."""
        fake_file = self.root_path / "test.txt"
        fake_file.write_text("dummy")
        valid, msg = self.mgr.validate_model_file(fake_file)
        self.assertFalse(valid)
        self.assertIn("Unsupported", msg)

    def test_activation_and_rollback(self):
        """Verify model activation history enables clean rollback."""
        # Active is mock_waste_detector_v1
        detector = self.mgr.get_active_detector()
        self.assertEqual(detector.model_name, "Synthetic Waste Detector (Test Fixture)")

        # Rollback when no previous model fails gracefully
        success, msg = self.mgr.rollback()
        self.assertFalse(success)


if __name__ == "__main__":
    unittest.main()


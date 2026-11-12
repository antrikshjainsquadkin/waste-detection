"""Unit tests for configuration loading, taxonomy validation, and URL sanitization."""

import unittest
from pathlib import Path
import tempfile
import json

from app.core.config import AppConfig, sanitize_source_reference, DEFAULT_CATEGORIES


class TestAppConfig(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = Path(self.temp_dir.name) / "test_config.json"
        self.config = AppConfig(self.config_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_default_categories_count_and_keys(self):
        """Verify exactly 8 approved categories exist with proper keys."""
        categories = self.config.categories
        self.assertEqual(len(categories), 8)
        names = [c["internal_name"] for c in categories]
        expected = [
            "paper", "cardboard", "plastic_bottle", "polythene_bag",
            "metal", "glass", "organic_waste", "general_trash"
        ]
        self.assertEqual(names, expected)

    def test_confidence_threshold_clamp(self):
        """Verify confidence thresholds are bounded between 0.05 and 0.95."""
        self.config.set_confidence_threshold("plastic_bottle", 0.99)
        self.assertEqual(self.config.get_confidence_threshold("plastic_bottle"), 0.95)

        self.config.set_confidence_threshold("plastic_bottle", 0.01)
        self.assertEqual(self.config.get_confidence_threshold("plastic_bottle"), 0.05)

        self.config.set_confidence_threshold("plastic_bottle", 0.65)
        self.assertAlmostEqual(self.config.get_confidence_threshold("plastic_bottle"), 0.65)

    def test_sanitize_source_reference(self):
        """Verify embedded credentials in RTSP URLs are stripped."""
        url_with_creds = "rtsp://admin:SecretPass123@192.168.1.100:554/stream1"
        sanitized = sanitize_source_reference(url_with_creds)
        self.assertNotIn("SecretPass123", sanitized)
        self.assertIn("rtsp://admin:***@192.168.1.100:554/stream1", sanitized)

        webcam_ref = "Webcam #0"
        self.assertEqual(sanitize_source_reference(webcam_ref), "Webcam #0")

    def test_save_and_reload_config(self):
        """Verify modifications persist across reloads."""
        self.config.set_frame_skip(4)
        self.config.set_lost_track_timeout_seconds(5.5)
        self.config.save()

        # Reload in new instance
        reloaded = AppConfig(self.config_path)
        self.assertEqual(reloaded.frame_skip, 4)
        self.assertAlmostEqual(reloaded.lost_track_timeout_seconds, 5.5)


if __name__ == "__main__":
    unittest.main()


"""End-to-End Pipeline Integration Test.

Runs complete monitoring session with synthetic video source, verifying
concurrent capture, tracking, deduplication, unannotated evidence saving,
live recording, and CSV finalization.
"""

import tempfile
import time
import unittest
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from app.core.config import AppConfig
from app.core.types import SourceType
from app.engine.pipeline import VisionPipeline
from app.engine.video_source import BaseVideoSource
from app.models.mock_detector import MockWasteDetector
from app.storage.session_manager import SessionManager


class SyntheticTestVideoSource(BaseVideoSource):
    """Generates synthetic video frames for deterministic integration testing."""

    def __init__(self, num_frames: int = 15, width: int = 320, height: int = 240):
        self._num_frames = num_frames
        self._width = width
        self._height = height
        self._current_frame = 0
        self._opened = False

    def open(self) -> bool:
        self._opened = True
        self._current_frame = 0
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._opened or self._current_frame >= self._num_frames:
            return False, None

        self._current_frame += 1
        # Create a frame with a moving synthetic white square (simulating waste object)
        frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        x = 50 + (self._current_frame * 5)
        y = 50
        cv2.rectangle(frame, (x, y), (x + 40, y + 60), (255, 255, 255), -1)
        time.sleep(0.02)
        return True, frame

    def close(self) -> None:
        self._opened = False

    @property
    def source_type(self) -> SourceType:
        return SourceType.WEBCAM  # Simulate webcam to test live recording trigger

    @property
    def reference_string(self) -> str:
        return "Synthetic Camera Feed"

    @property
    def native_fps(self) -> float:
        return 30.0

    @property
    def dimensions(self) -> Tuple[int, int]:
        return self._width, self._height


class TestPipelineIntegration(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)

        self.config = AppConfig()
        self.config._raw_data["storage"]["recordings_dir"] = str(self.root_path / "recordings")
        self.session_manager = SessionManager(self.config)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_full_pipeline_run(self):
        """Execute complete pipeline lifecycle and verify all persistent artifacts."""
        source = SyntheticTestVideoSource(num_frames=12, width=320, height=240)
        detector = MockWasteDetector()

        pipeline = VisionPipeline(
            config=self.config,
            session_manager=self.session_manager,
            video_source=source,
            detector=detector,
        )

        started = pipeline.start()
        self.assertTrue(started)
        self.assertTrue(pipeline.is_running)

        # Allow worker threads to process synthetic frames
        time.sleep(1.0)

        # Stop pipeline
        pipeline.stop()
        self.assertFalse(pipeline.is_running)

        # Verify session artifacts
        sessions = self.session_manager.list_sessions()
        self.assertEqual(len(sessions), 1)

        s_dir = Path(sessions[0]["dir_path"])
        self.assertTrue(s_dir.exists())

        # 1. detections.csv
        det_csv = s_dir / "detections.csv"
        self.assertTrue(det_csv.exists())
        self.assertGreater(det_csv.stat().st_size, 0)

        # 2. summary.csv
        sum_csv = s_dir / "summary.csv"
        self.assertTrue(sum_csv.exists())
        self.assertGreater(sum_csv.stat().st_size, 0)

        # 3. session_metadata.json
        meta_json = s_dir / "session_metadata.json"
        self.assertTrue(meta_json.exists())

        # 4. Evidence directory
        evidence_dir = s_dir / "evidence"
        self.assertTrue(evidence_dir.exists())
        evidence_files = list(evidence_dir.glob("*.jpg"))
        self.assertGreaterEqual(len(evidence_files), 1)

        # 5. Live Recording file
        rec_file = s_dir / "recording.mp4"
        self.assertTrue(rec_file.exists())
        self.assertGreater(rec_file.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()


"""Unit tests for CSV persistence, evidence image saving, and session lifecycle."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from app.core.config import AppConfig
from app.core.types import BoundingBox, DetectionEvent, SessionMetadata, SessionStatus
from app.storage.csv_writer import DetectionCsvWriter, SummaryCsvWriter, DETECTION_CSV_HEADERS
from app.storage.evidence_saver import EvidenceSaver
from app.storage.session_manager import SessionManager
from app.storage.retention import StorageManager


class TestStorageSubsystem(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)

        # Mock app config pointing to temp storage
        self.config = AppConfig()
        self.config._raw_data["storage"]["recordings_dir"] = str(self.root_path / "recordings")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_detection_csv_writer_schema(self):
        """Verify DetectionCsvWriter generates exact documented schema."""
        csv_file = self.root_path / "test_detections.csv"
        writer = DetectionCsvWriter(csv_file)

        event = DetectionEvent(
            event_id="evt_001",
            session_id="session_001",
            track_id=1,
            category="metal",
            display_name="Metal",
            confidence=0.88,
            first_seen="2026-11-03T16:00:00",
            last_seen="2026-11-03T16:00:05",
            duration_seconds=5.0,
            bbox=BoundingBox(10, 20, 30, 40),
            frame_width=640,
            frame_height=480,
            source_type="webcam",
            source_reference="Webcam #0",
            model_name="TestModel",
            model_version="1.0",
            evidence_image_path="evidence/evt_001_metal.jpg",
        )

        writer.write_event(event)

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            row = next(reader)

        self.assertEqual(header, DETECTION_CSV_HEADERS)
        self.assertEqual(row[0], "session_001")
        self.assertEqual(row[1], "evt_001")
        self.assertEqual(row[3], "metal")

    def test_evidence_saver_writes_pristine_frame(self):
        """Verify EvidenceSaver writes unannotated image to disk."""
        evidence_dir = self.root_path / "evidence"
        saver = EvidenceSaver(evidence_dir, image_format="jpg", quality=90)

        dummy_frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        saved_path = saver.save_evidence(dummy_frame, "evt_test_123", "cardboard")

        self.assertTrue(Path(saved_path).exists())
        self.assertTrue(saved_path.endswith(".jpg"))
        self.assertGreater(Path(saved_path).stat().st_size, 0)

    def test_session_lifecycle_and_summary(self):
        """Verify complete session lifecycle from start to finalization."""
        mgr = SessionManager(self.config)

        # 1. Start Session
        meta = mgr.start_session("webcam", "Webcam #0", "TestModel", "1.0.0")
        self.assertEqual(meta.status, SessionStatus.ACTIVE.value)
        self.assertTrue((mgr.current_session_dir / "session_metadata.json").exists())

        # 2. Record an interruption
        mgr.record_interruption("Frame dropped temporarily")

        # 3. Finalize Session
        counts = {"paper": 2, "plastic_bottle": 5}
        final_meta = mgr.finalize_session(
            total_events=7,
            category_counts=counts,
            unknown_count=1,
            duration_seconds=12.5,
            average_fps=15.0,
            recording_path=str(mgr.recordings_root / meta.session_id / "recording.mp4"),
            status=SessionStatus.STOPPED,
        )

        self.assertIsNotNone(final_meta)
        self.assertEqual(final_meta.status, SessionStatus.STOPPED.value)
        self.assertEqual(final_meta.total_events, 7)

        # Check summary CSV written
        summary_csv = mgr.recordings_root / meta.session_id / "summary.csv"
        self.assertTrue(summary_csv.exists())

    def test_storage_manager_deletion_confirmation(self):
        """Verify deletion is strictly rejected without confirm=True."""
        storage = StorageManager(self.config)
        s_dir = self.config.recordings_dir / "session_dummy"
        s_dir.mkdir(parents=True, exist_ok=True)

        # Attempt deletion without confirmation
        success, msg = storage.delete_session("session_dummy", confirm=False)
        self.assertFalse(success)
        self.assertTrue(s_dir.exists())

        # Attempt with explicit confirmation
        success, msg = storage.delete_session("session_dummy", confirm=True)
        self.assertTrue(success)
        self.assertFalse(s_dir.exists())


if __name__ == "__main__":
    unittest.main()


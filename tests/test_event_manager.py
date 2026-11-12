"""Unit tests for Detection Event lifecycle, evidence coordination, and statistics."""

import unittest
import numpy as np

from app.core.config import AppConfig
from app.core.event_manager import EventManager
from app.core.types import BoundingBox, TrackedObject


class TestEventManager(unittest.TestCase):

    def setUp(self):
        self.config = AppConfig()
        self.saved_evidence_calls = []

        def mock_evidence_saver(frame, event_id, cat):
            self.saved_evidence_calls.append((event_id, cat))
            return f"evidence/{event_id}_{cat}.jpg"

        self.created_events = []
        self.updated_events = []
        self.finalized_events = []

        self.mgr = EventManager(
            config=self.config,
            session_id="session_test_001",
            source_type="webcam",
            source_reference="Webcam #0",
            model_name="TestDetector",
            model_version="1.0",
            evidence_save_fn=mock_evidence_saver,
            on_event_created=lambda e: self.created_events.append(e),
            on_event_updated=lambda e: self.updated_events.append(e),
            on_event_finalized=lambda e: self.finalized_events.append(e),
        )

    def test_handle_track_created_generates_event_and_evidence(self):
        """Track creation triggers single full-frame evidence capture and creates event."""
        track = TrackedObject(
            track_id=1,
            class_id=2,
            class_name="plastic_bottle",
            display_name="Plastic Bottle",
            confidence=0.91,
            bbox=BoundingBox(50, 50, 40, 80),
            first_seen=10.0,
            last_seen=10.0,
        )

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        event = self.mgr.handle_track_created(track, raw_frame=dummy_frame)

        self.assertIsNotNone(event)
        self.assertEqual(len(self.created_events), 1)
        self.assertEqual(len(self.saved_evidence_calls), 1)
        self.assertEqual(self.mgr.total_events, 1)
        self.assertEqual(event.category, "plastic_bottle")
        self.assertEqual(event.is_unknown, False)
        self.assertIn("plastic_bottle.jpg", event.evidence_image_path)

    def test_unknown_event_tracking(self):
        """Unknown detections are preserved and categorized as unknown."""
        track = TrackedObject(
            track_id=2,
            class_id=3,
            class_name="unknown",
            display_name="Unknown",
            confidence=0.35,
            bbox=BoundingBox(10, 10, 30, 30),
            first_seen=5.0,
            last_seen=5.0,
            is_unknown=True,
        )

        event = self.mgr.handle_track_created(track, raw_frame=None)
        self.assertTrue(event.is_unknown)
        self.assertEqual(event.category, "unknown")
        self.assertEqual(self.mgr.get_unknown_count(), 1)


if __name__ == "__main__":
    unittest.main()


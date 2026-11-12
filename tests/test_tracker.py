"""Unit tests for spatial-temporal tracking, event deduplication, and track lifecycle."""

import unittest
from app.core.tracker import ObjectTracker
from app.core.types import BoundingBox, Detection, TrackedObject


class TestObjectTracker(unittest.TestCase):

    def setUp(self):
        self.tracker = ObjectTracker(iou_threshold=0.35, lost_track_timeout_seconds=2.0)
        self.created_tracks = []
        self.updated_tracks = []
        self.finalized_tracks = []

        self.tracker.on_track_created = lambda t: self.created_tracks.append(t)
        self.tracker.on_track_updated = lambda t: self.updated_tracks.append(t)
        self.tracker.on_track_finalized = lambda t: self.finalized_tracks.append(t)

    def test_single_detection_creates_single_track(self):
        """Single detection in frame 1 initiates track 1."""
        det = Detection(
            class_id=2,
            class_name="plastic_bottle",
            display_name="Plastic Bottle",
            confidence=0.85,
            bbox=BoundingBox(100, 100, 50, 100),
            is_unknown=False,
        )

        tracks = self.tracker.update([det], current_time=10.0)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].track_id, 1)
        self.assertEqual(len(self.created_tracks), 1)
        self.assertEqual(len(self.updated_tracks), 0)

    def test_consecutive_frames_deduplication(self):
        """Tracking across consecutive frames updates the existing track without creating duplicates."""
        det1 = Detection(2, "plastic_bottle", "Plastic Bottle", 0.85, BoundingBox(100, 100, 50, 100))
        self.tracker.update([det1], current_time=10.0)

        # Frame 2: slightly shifted box
        det2 = Detection(2, "plastic_bottle", "Plastic Bottle", 0.88, BoundingBox(102, 101, 50, 100))
        tracks = self.tracker.update([det2], current_time=10.5)

        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].track_id, 1)
        self.assertEqual(tracks[0].hit_count, 2)
        # Verify no second track was created
        self.assertEqual(len(self.created_tracks), 1)
        self.assertEqual(len(self.updated_tracks), 1)
        self.assertEqual(tracks[0].confidence, 0.88)

    def test_lost_track_timeout_and_finalization(self):
        """When an object disappears beyond the timeout, it is finalized."""
        det = Detection(2, "plastic_bottle", "Plastic Bottle", 0.85, BoundingBox(100, 100, 50, 100))
        self.tracker.update([det], current_time=10.0)

        # Frames with no detections until timeout (timeout is 2.0s)
        self.tracker.update([], current_time=11.0)
        self.assertEqual(len(self.finalized_tracks), 0)

        # Surpass 2.0s timeout (10.0 + 2.1 = 12.1s)
        tracks = self.tracker.update([], current_time=12.2)
        self.assertEqual(len(tracks), 0)
        self.assertEqual(len(self.finalized_tracks), 1)
        self.assertEqual(self.finalized_tracks[0].track_id, 1)

    def test_reappearance_after_timeout_creates_new_track(self):
        """An object reappearing after timeout receives a new tracking identity."""
        det1 = Detection(2, "plastic_bottle", "Plastic Bottle", 0.85, BoundingBox(100, 100, 50, 100))
        self.tracker.update([det1], current_time=10.0)

        # Disappear and finalize
        self.tracker.update([], current_time=13.0)
        self.assertEqual(len(self.finalized_tracks), 1)

        # Reappears at frame 15.0s
        det2 = Detection(2, "plastic_bottle", "Plastic Bottle", 0.82, BoundingBox(105, 102, 50, 100))
        tracks = self.tracker.update([det2], current_time=15.0)

        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].track_id, 2)  # New track ID
        self.assertEqual(len(self.created_tracks), 2)


if __name__ == "__main__":
    unittest.main()


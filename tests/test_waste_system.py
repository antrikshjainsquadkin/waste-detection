"""Comprehensive Unit & Integration Test Suite for Waste Detection System.

Verifies:
- AC-01 to AC-22 compliance
- Model integrity, task and class mapping verification
- Inference and object tracking deduplication
- Video recording and playback validation
- CSV schema and JSON metadata persistence
- Input sources and error resilience
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.config import AppConfig, NATIVE_CLASSES, sanitize_source_reference
from app.detector import (
    Detection,
    NonWasteVerifier,
    VerificationResult,
    VerificationState,
    WasteDetector,
    compute_box_overlap,
)
from app.model_loader import (
    EXPECTED_CLASSES,
    EXPECTED_SHA256,
    compute_sha256,
    load_waste_model,
)
from app.recorder import VideoRecorder
from app.sources import ImageSource, RtspSource, VideoFileSource, WebcamSource, resolve_stream_url
from app.storage import DetectionEvent, StorageManager
from main import PipelineProfiler
from app.tracking import DetectionTracker, compute_iou
from app.visualizer import (
    CLASS_COLORS,
    DISPLAY_LABEL,
    VisualPreviewWindow,
    format_display_label,
    get_class_color,
    render_annotated_frame,
    render_paused_overlay,
)


class TestModelIntegrityAndLoading(unittest.TestCase):
    """Test model integrity, file existence, and class mappings."""

    def test_checkpoint_sha256(self):
        project_root = Path(__file__).resolve().parent.parent
        weights_path = project_root / "models" / "pretrained" / "waste_detection_yolo11m.pt"
        self.assertTrue(weights_path.exists(), f"Model file missing: {weights_path}")
        self.assertGreater(weights_path.stat().st_size, 30 * 1024 * 1024, "Model file is truncated")

        actual_sha = compute_sha256(weights_path)
        self.assertEqual(
            actual_sha.lower(),
            EXPECTED_SHA256.lower(),
            f"SHA256 mismatch! Expected {EXPECTED_SHA256}, got {actual_sha}",
        )

    def test_model_loading_and_classes(self):
        model, path, sha, classes = load_waste_model()
        self.assertIsNotNone(model)
        self.assertEqual(model.task, "detect")
        self.assertEqual(classes, EXPECTED_CLASSES)
        self.assertEqual(classes[0], "organic")
        self.assertEqual(classes[1], "recyclable")
        self.assertEqual(classes[2], "residual")
        self.assertEqual(len(classes), 3, "Model must expose exactly 3 native classes")


class TestDetectorInference(unittest.TestCase):
    """Test detector inference on synthetic and real frames."""

    @classmethod
    def setUpClass(cls):
        cls.detector = WasteDetector(confidence_threshold=0.25, image_size=640, device="cpu")

    def test_inference_on_blank_frame(self):
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = self.detector.predict(blank)
        self.assertIsInstance(detections, list)
        self.assertEqual(len(detections), 0, "Blank frame should yield 0 detections")

    def test_inference_on_real_sample(self):
        project_root = Path(__file__).resolve().parent.parent
        sample_img = project_root / "recordings" / "session_20261103_175001" / "evidence" / "evt_20261103_175002_9620d2_plastic_bottle.jpg"
        if not sample_img.exists():
            self.skipTest("Sample evidence image not found")

        img = cv2.imread(str(sample_img))
        self.assertIsNotNone(img)
        detections = self.detector.predict(img)
        self.assertGreater(len(detections), 0, "Expected at least 1 detection on sample image")
        det = detections[0]
        self.assertIn(det.class_name, ("recyclable", "organic", "residual"))
        self.assertGreaterEqual(det.confidence, 0.25)


class TestTrackingAndDeduplication(unittest.TestCase):
    """Test object tracking and duplicate event prevention."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.temp_dir.name) / "test_detections.csv"
        self.storage = StorageManager(
            csv_path=self.csv_path,
            session_id="test_session",
            source_type="test_video",
            source_reference="test.mp4",
            model_name="khanhromvn/waste-detection-yolo11",
            checkpoint_filename="waste_detection_yolo11m.pt",
            model_sha256="test_sha",
            model_classes={0: "organic", 1: "recyclable", 2: "residual"},
            confidence_threshold=0.25,
            image_size=640,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_compute_iou(self):
        box1 = (10, 10, 50, 50)
        box2 = (10, 10, 50, 50)
        self.assertAlmostEqual(compute_iou(box1, box2), 1.0)

        box3 = (100, 100, 50, 50)
        self.assertAlmostEqual(compute_iou(box1, box3), 0.0)

    def test_deduplication_across_frames(self):
        """Verify that an object seen across consecutive frames creates only 1 event."""
        tracker = DetectionTracker(
            storage_manager=self.storage,
            iou_threshold=0.35,
            lost_track_timeout=3.0,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Frame 1: Object appears
        det1 = Detection(class_id=1, class_name="recyclable", confidence=0.75, bbox_x=100, bbox_y=100, bbox_w=50, bbox_h=50, track_id=1)
        tracks1 = tracker.process_frame([det1], frame, frame_index=1)
        self.assertEqual(len(tracks1), 1)
        self.assertEqual(self.storage.total_events, 1)

        # Frame 2: Same object continues (same track_id)
        det2 = Detection(class_id=1, class_name="recyclable", confidence=0.82, bbox_x=102, bbox_y=101, bbox_w=50, bbox_h=50, track_id=1)
        tracks2 = tracker.process_frame([det2], frame, frame_index=2)
        self.assertEqual(len(tracks2), 1)
        # MUST STILL BE 1 EVENT (no duplicate record!)
        self.assertEqual(self.storage.total_events, 1, "Duplicate event created for continuing track!")

        # Frame 3: Second object appears
        det3 = Detection(class_id=0, class_name="organic", confidence=0.65, bbox_x=300, bbox_y=300, bbox_w=40, bbox_h=40, track_id=2)
        tracks3 = tracker.process_frame([det2, det3], frame, frame_index=3)
        self.assertEqual(len(tracks3), 2)
        self.assertEqual(self.storage.total_events, 2)

        # Verify CSV has exactly 2 rows
        with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2)


class TestStorageAndRecording(unittest.TestCase):
    """Test CSV persistence, append behavior, schema integrity, and duplicate prevention."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)
        self.csv_path = self.base_dir / "waste_detections.csv"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_csv_schema_and_append(self):
        storage1 = StorageManager(
            csv_path=self.csv_path,
            session_id="session_01",
            source_type="image",
            source_reference="test_img.jpg",
            model_name="khanhromvn/waste-detection-yolo11",
            checkpoint_filename="waste_detection_yolo11m.pt",
            confidence_threshold=0.25,
            image_size=640,
        )

        ev1 = DetectionEvent(
            record_id="rec_01",
            session_id="session_01",
            event_id="evt_01",
            track_id=None,
            detected_at="2026-11-04T10:00:00",
            source_type="image",
            source_name="test_img.jpg",
            frame_number=1,
            class_id=1,
            class_name="recyclable",
            confidence=0.8845,
            bbox_x1=10,
            bbox_y1=20,
            bbox_x2=60,
            bbox_y2=70,
            model_name="khanhromvn/waste-detection-yolo11",
            checkpoint_filename="waste_detection_yolo11m.pt",
        )
        storage1.record_event(ev1)
        # Attempt duplicate write
        storage1.record_event(ev1)

        # Verify 1 row only
        with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        self.assertEqual(len(reader), 1)
        self.assertEqual(reader[0]["record_id"], "rec_01")
        self.assertEqual(reader[0]["bbox_x1"], "10")
        self.assertEqual(reader[0]["bbox_x2"], "60")

        # Session 2: append new record to same CSV
        storage2 = StorageManager(
            csv_path=self.csv_path,
            session_id="session_02",
            source_type="video",
            source_reference="clip.mp4",
            model_name="khanhromvn/waste-detection-yolo11",
            checkpoint_filename="waste_detection_yolo11m.pt",
            confidence_threshold=0.25,
            image_size=640,
        )
        ev2 = DetectionEvent(
            record_id="rec_02",
            session_id="session_02",
            event_id="evt_02",
            track_id=42,
            detected_at="2026-11-04T10:05:00",
            source_type="video",
            source_name="clip.mp4",
            frame_number=15,
            class_id=0,
            class_name="organic",
            confidence=0.9123,
            bbox_x1=100,
            bbox_y1=150,
            bbox_x2=200,
            bbox_y2=250,
            model_name="khanhromvn/waste-detection-yolo11",
            checkpoint_filename="waste_detection_yolo11m.pt",
        )
        storage2.record_event(ev2)

        # Verify historical record preserved and new record appended (total 2 rows)
        with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["event_id"], "evt_01")
        self.assertEqual(rows[1]["event_id"], "evt_02")
        self.assertEqual(rows[1]["session_id"], "session_02")
        self.assertEqual(rows[1]["track_id"], "42")

    def test_video_recorder(self):
        rec_path = self.base_dir / "test_recording.mp4"
        recorder = VideoRecorder(output_path=rec_path, frame_size=(320, 240), fps=25.0)
        started = recorder.start()
        self.assertTrue(started, "Failed to start VideoRecorder")

        # Write 15 dummy frames
        dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        for i in range(15):
            cv2.putText(dummy_frame, f"F{i}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            recorder.write_frame(dummy_frame)

        recorder.stop()
        final_path = Path(recorder.final_file_path)
        self.assertTrue(final_path.exists())
        self.assertGreater(final_path.stat().st_size, 0)

        # Validate that the recorded video is readable
        cap = cv2.VideoCapture(str(final_path))
        self.assertTrue(cap.isOpened(), "Recorded video cannot be opened by OpenCV")
        ret, frame = cap.read()
        self.assertTrue(ret, "Failed reading first frame of recorded video")
        self.assertEqual(frame.shape[:2], (240, 320))
        cap.release()


class TestSourcesAndSecurity(unittest.TestCase):
    """Test input sources, credential protection, and path handling."""

    def test_credential_sanitization(self):
        url = "rtsp://admin:super_secret_password@192.168.1.50:554/live"
        sanitized = sanitize_source_reference(url)
        self.assertNotIn("super_secret_password", sanitized)
        self.assertEqual(sanitized, "rtsp://admin:***@192.168.1.50:554/live")

    def test_webcam_graceful_handling(self):
        # Open unlikely camera index
        source = WebcamSource(camera_index=999)
        # Should return False gracefully without raising unhandled exception
        success = source.open()
        self.assertFalse(success)
        source.close()

    def test_video_source_nonexistent(self):
        source = VideoFileSource(Path("non_existent_file_path.mp4"))
        self.assertFalse(source.open())

    def test_resolve_stream_url(self):
        url1, msg1 = resolve_stream_url("https://192.168.1.11:8080/")
        self.assertEqual(url1, "https://192.168.1.11:8080/video")
        self.assertIn("/video", msg1)

        url2, msg2 = resolve_stream_url("http://192.168.1.11:8080")
        self.assertEqual(url2, "http://192.168.1.11:8080/video")

        url3, msg3 = resolve_stream_url("rtsp://192.168.1.50:554/live")
        self.assertEqual(url3, "rtsp://192.168.1.50:554/live")


class TestVisualPreview(unittest.TestCase):
    """Test visual preview rendering, colors, and controls."""

    def test_display_label_formatting(self):
        """Verify centralized DISPLAY_LABEL and format_display_label presentation rule."""
        self.assertEqual(DISPLAY_LABEL, "WASTE")
        self.assertEqual(format_display_label(0.61, 1), "WASTE #1 (0.61)")
        self.assertEqual(format_display_label(0.85, 2), "WASTE #2 (0.85)")
        self.assertEqual(format_display_label(0.72, 3), "WASTE #3 (0.72)")
        self.assertEqual(format_display_label(0.61, None), "WASTE (0.61)")

    def test_class_colors(self):
        self.assertEqual(get_class_color("organic"), (50, 205, 50))
        self.assertEqual(get_class_color("recyclable"), (255, 140, 0))
        self.assertEqual(get_class_color("residual"), (30, 70, 240))
        self.assertEqual(get_class_color("unknown_class"), (200, 200, 200))

    def test_render_annotated_frame_immutability(self):
        """Ensure original frame is never modified in-place."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        frame_copy = frame.copy()
        det = Detection(class_id=1, class_name="recyclable", confidence=0.88, bbox_x=20, bbox_y=20, bbox_w=50, bbox_h=50, track_id=5)

        annotated = render_annotated_frame(frame, [det], show_hud=True)
        # Original frame must be untouched
        self.assertTrue(np.array_equal(frame, frame_copy), "Original frame was mutated by render_annotated_frame")
        # Annotated frame must have visual elements drawn (non-zero)
        self.assertFalse(np.array_equal(annotated, frame), "Annotated frame did not draw any visual elements")

    def test_render_empty_detections(self):
        """Ensure empty detections still render clean frame without error."""
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 100
        annotated = render_annotated_frame(frame, [], show_hud=False)
        self.assertTrue(np.array_equal(frame, annotated))

    def test_paused_overlay(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        frame_copy = frame.copy()
        paused = render_paused_overlay(frame)
        self.assertTrue(np.array_equal(frame, frame_copy), "Original frame was mutated by render_paused_overlay")
        self.assertFalse(np.array_equal(paused, frame), "Paused overlay was not drawn")

    def test_preview_window_lifecycle(self):
        win = VisualPreviewWindow(window_title="Test Unit Window")
        self.assertFalse(win._window_created)
        win.close()  # Must close gracefully even if never shown


class TestNonWasteVerification(unittest.TestCase):
    """Test secondary person and animal false-positive filtering."""

    def test_compute_box_overlap(self):
        # Case 1: Identical boxes
        iou, ioa, rel_area = compute_box_overlap((10, 10, 50, 50), (10, 10, 50, 50))
        self.assertAlmostEqual(iou, 1.0)
        self.assertAlmostEqual(ioa, 1.0)
        self.assertAlmostEqual(rel_area, 1.0)

        # Case 2: Zero overlap
        iou, ioa, rel_area = compute_box_overlap((0, 0, 10, 10), (100, 100, 20, 20))
        self.assertEqual(iou, 0.0)
        self.assertEqual(ioa, 0.0)

        # Case 3: Small item inside larger box (e.g. bottle inside person)
        iou, ioa, rel_area = compute_box_overlap((50, 50, 10, 10), (0, 0, 100, 100))
        self.assertEqual(ioa, 1.0)
        self.assertEqual(rel_area, 100 / 10000)  # 0.01 (1%)
        self.assertLess(iou, 0.02)

    def test_human_false_positive_rejection(self):
        verifier = NonWasteVerifier(enabled=False)  # Test logic directly with mock boxes
        verifier.is_available = True
        verifier.class_mapping = {0: "person", 15: "cat"}

        # Candidate covering person torso (human false positive)
        human_cand = Detection(
            class_id=1, class_name="recyclable", confidence=0.61,
            bbox_x=100, bbox_y=100, bbox_w=200, bbox_h=300
        )

        class MockBoxes:
            def __init__(self, boxes):
                import torch
                self.boxes = boxes
                self.xywh = torch.tensor([[200.0, 250.0, 220.0, 320.0]])  # cx, cy, w, h
                self.conf = torch.tensor([0.90])
                self.cls = torch.tensor([0.0])  # person
            def __len__(self):
                return 1

        class MockResult:
            def __init__(self):
                self.boxes = MockBoxes(self)

        class MockModel:
            def predict(self, *args, **kwargs):
                return [MockResult()]

        verifier.model = MockModel()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        accepted, results, counts = verifier.verify_candidates(frame, [human_cand])
        self.assertEqual(len(accepted), 0, "Human false positive should be rejected")
        self.assertEqual(counts["rejected"], 1)
        self.assertEqual(results[0].state, VerificationState.REJECTED)

    def test_genuine_waste_near_person_retention(self):
        verifier = NonWasteVerifier(enabled=False)
        verifier.is_available = True
        verifier.class_mapping = {0: "person"}

        # Small bottle held by person (small relative area)
        bottle_cand = Detection(
            class_id=1, class_name="recyclable", confidence=0.85,
            bbox_x=180, bbox_y=200, bbox_w=30, bbox_h=60  # area: 1800
        )

        class MockBoxes:
            def __init__(self, boxes):
                import torch
                self.boxes = boxes
                self.xywh = torch.tensor([[200.0, 250.0, 200.0, 400.0]])  # person area: 80,000 (bottle is ~2.2%)
                self.conf = torch.tensor([0.92])
                self.cls = torch.tensor([0.0])
            def __len__(self):
                return 1

        class MockResult:
            def __init__(self):
                self.boxes = MockBoxes(self)

        class MockModel:
            def predict(self, *args, **kwargs):
                return [MockResult()]

        verifier.model = MockModel()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        accepted, results, counts = verifier.verify_candidates(frame, [bottle_cand])
        self.assertEqual(len(accepted), 1, "Genuine waste near/held by person must be preserved")
        self.assertEqual(counts["accepted"], 1)
        self.assertEqual(results[0].state, VerificationState.ACCEPTED)

    def test_person_holding_cardboard_or_waste_bag_preserved(self):
        """T6/T7/T8: Medium cardboard/bag inside or beside person box without COCO corroboration must be accepted."""
        verifier = NonWasteVerifier(enabled=False)
        verifier.is_available = True
        verifier.class_mapping = {0: "person", 39: "bottle"}

        # Cardboard box held in front of torso: area 16,800 vs person area 80,000 -> rel_area=0.21, IoA=1.0, IoU=0.21
        cardboard_cand = Detection(
            class_id=1, class_name="recyclable", confidence=0.76,
            bbox_x=130, bbox_y=180, bbox_w=140, bbox_h=120
        )
        # Waste bag beside person partially overlapping edge: IoA=0.45, rel_area=0.20, IoU=0.14
        bag_cand = Detection(
            class_id=2, class_name="residual", confidence=0.68,
            bbox_x=240, bbox_y=300, bbox_w=100, bbox_h=160
        )

        class MockBoxes:
            def __init__(self, boxes):
                import torch
                self.boxes = boxes
                self.xywh = torch.tensor([[200.0, 250.0, 200.0, 400.0]])  # person (100,50,200,400)
                self.conf = torch.tensor([0.90])
                self.cls = torch.tensor([0.0])
            def __len__(self):
                return 1

        class MockResult:
            def __init__(self):
                self.boxes = MockBoxes(self)

        class MockModel:
            def predict(self, *args, **kwargs):
                return [MockResult()]

        verifier.model = MockModel()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        accepted, results, counts = verifier.verify_candidates(frame, [cardboard_cand, bag_cand])
        self.assertEqual(len(accepted), 2, "Cardboard held by person and bag beside person must be accepted")
        self.assertEqual(counts["accepted"], 2)
        self.assertEqual(counts["rejected"], 0)

    def test_corroborated_bottle_held_by_person_preserved(self):
        """T5: Bottle held by person corroborated by secondary COCO 'bottle' detection must be accepted."""
        verifier = NonWasteVerifier(enabled=False)
        verifier.is_available = True
        verifier.class_mapping = {0: "person", 39: "bottle"}

        # Bottle overlapping person box closely (rel_area=0.30, IoA=1.0)
        bottle_cand = Detection(
            class_id=1, class_name="recyclable", confidence=0.72,
            bbox_x=140, bbox_y=150, bbox_w=120, bbox_h=200
        )

        class MockBoxes:
            def __init__(self, boxes):
                import torch
                self.boxes = boxes
                self.xywh = torch.tensor([
                    [200.0, 250.0, 200.0, 400.0],  # person
                    [200.0, 250.0, 115.0, 195.0],  # bottle
                ])
                self.conf = torch.tensor([0.92, 0.68])
                self.cls = torch.tensor([0.0, 39.0])
            def __len__(self):
                return 2

        class MockResult:
            def __init__(self):
                self.boxes = MockBoxes(self)

        class MockModel:
            def predict(self, *args, **kwargs):
                return [MockResult()]

        verifier.model = MockModel()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        accepted, results, counts = verifier.verify_candidates(frame, [bottle_cand])
        self.assertEqual(len(accepted), 1)
        self.assertEqual(results[0].state, VerificationState.ACCEPTED)
        self.assertIn("corroborated_by_bottle", results[0].reason)

    def test_animal_rejection_and_nearby_waste_preservation(self):
        """T9/T10: Animal mistaken for waste is rejected, while separate waste near animal is accepted."""
        verifier = NonWasteVerifier(enabled=False)
        verifier.is_available = True
        verifier.class_mapping = {16: "dog"}

        dog_fp = Detection(
            class_id=2, class_name="residual", confidence=0.55,
            bbox_x=100, bbox_y=200, bbox_w=180, bbox_h=140
        )
        nearby_can = Detection(
            class_id=1, class_name="recyclable", confidence=0.80,
            bbox_x=320, bbox_y=300, bbox_w=40, bbox_h=60
        )

        class MockBoxes:
            def __init__(self, boxes):
                import torch
                self.boxes = boxes
                self.xywh = torch.tensor([[190.0, 270.0, 185.0, 145.0]])  # dog
                self.conf = torch.tensor([0.88])
                self.cls = torch.tensor([16.0])
            def __len__(self):
                return 1

        class MockResult:
            def __init__(self):
                self.boxes = MockBoxes(self)

        class MockModel:
            def predict(self, *args, **kwargs):
                return [MockResult()]

        verifier.model = MockModel()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        accepted, results, counts = verifier.verify_candidates(frame, [dog_fp, nearby_can])
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0].class_name, "recyclable")
        self.assertEqual(counts["rejected"], 1)
        self.assertEqual(counts["accepted"], 1)

    def test_furniture_false_positive_rejection_preserves_item_on_table(self):
        """Non-biological furniture 1:1 match is rejected, while small waste on a table is preserved."""
        verifier = NonWasteVerifier(enabled=False)
        verifier.is_available = True
        verifier.class_mapping = {60: "dining table"}

        table_fp = Detection(
            class_id=2, class_name="residual", confidence=0.50,
            bbox_x=100, bbox_y=200, bbox_w=300, bbox_h=200
        )
        bottle_on_table = Detection(
            class_id=1, class_name="recyclable", confidence=0.80,
            bbox_x=220, bbox_y=230, bbox_w=30, bbox_h=60
        )

        class MockBoxes:
            def __init__(self, boxes):
                import torch
                self.boxes = boxes
                self.xywh = torch.tensor([[250.0, 300.0, 310.0, 210.0]])
                self.conf = torch.tensor([0.85])
                self.cls = torch.tensor([60.0])
            def __len__(self):
                return 1

        class MockResult:
            def __init__(self):
                self.boxes = MockBoxes(self)

        class MockModel:
            def predict(self, *args, **kwargs):
                return [MockResult()]

        verifier.model = MockModel()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        accepted, results, counts = verifier.verify_candidates(frame, [table_fp, bottle_on_table])
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0].bbox, bottle_on_table.bbox)
        self.assertEqual(counts["rejected"], 1)
        self.assertEqual(counts["accepted"], 1)

    def test_selective_cache_behavior_and_invalidation(self):
        """Cache skips redundant secondary inference for stable tracks when clear, and invalidates on drift/TTL."""
        verifier = NonWasteVerifier(enabled=False, cache_ttl_frames=3)
        verifier.is_available = True
        verifier.class_mapping = {0: "person"}
        call_count = 0

        class EmptyBoxes:
            def __len__(self):
                return 0

        class MockResult:
            def __init__(self):
                self.boxes = EmptyBoxes()

        class MockModel:
            def predict(self, *args, **kwargs):
                nonlocal call_count
                call_count += 1
                return [MockResult()]

        verifier.model = MockModel()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cand = Detection(class_id=1, class_name="recyclable", confidence=0.82, bbox_x=50, bbox_y=50, bbox_w=40, bbox_h=80, track_id=7)

        verifier.verify_candidates(frame, [cand])  # frame 1: runs model
        verifier.verify_candidates(frame, [cand])  # frame 2: cache hit
        verifier.verify_candidates(frame, [cand])  # frame 3: cache hit
        self.assertEqual(call_count, 1)

        verifier.verify_candidates(frame, [cand])  # frame 4: TTL expired -> runs model
        self.assertEqual(call_count, 2)

        # Drift > 15% invalidates cache immediately
        moved_cand = Detection(class_id=1, class_name="recyclable", confidence=0.82, bbox_x=150, bbox_y=150, bbox_w=40, bbox_h=80, track_id=7)
        verifier.verify_candidates(frame, [moved_cand])
        self.assertEqual(call_count, 3)


class TestPipelineProfiler(unittest.TestCase):
    """Test performance latency measurement, warm-up exclusion, and statistics."""

    def test_profiler_warmup_and_stats(self):
        profiler = PipelineProfiler(
            warmup_frames=2,
            device_name="CPU",
            primary_model_name="YOLO11m",
            secondary_model_name="YOLOv8n",
            source_name="test_clip.mp4",
            source_resolution=(640, 480),
            source_fps=30.0,
            preview_enabled=True,
            primary_startup_ms=1500.0,
            secondary_startup_ms=450.0,
        )

        # Record 2 warm-up frames
        for f in [1, 2]:
            profiler.record_frame(
                frame_index=f,
                read_ms=10.0,
                primary_ms=800.0,
                verifier_ms=120.0,
                tracking_ms=5.0,
                filtering_ms=1.0,
                annotation_ms=2.0,
                imshow_ms=3.0,
                waitkey_ms=1.0,
                csv_write_ms=0.0,
                end_to_end_ms=942.0,
            )

        # Warm-up frames must NOT be included in benchmark samples
        self.assertEqual(len(profiler.stage_samples["Primary YOLO11m"]), 0)
        self.assertEqual(profiler.warmup_frames_counted, 2)

        # Record 4 steady-state frames
        latencies = [800.0, 850.0, 900.0, 950.0]
        for idx, lat in enumerate(latencies, start=3):
            profiler.record_frame(
                frame_index=idx,
                read_ms=10.0,
                primary_ms=lat,
                verifier_ms=120.0,
                tracking_ms=5.0,
                filtering_ms=1.0,
                annotation_ms=2.0,
                imshow_ms=3.0,
                waitkey_ms=1.0,
                csv_write_ms=15.0 if idx == 3 else 0.0,
                end_to_end_ms=lat + 140.0,
            )

        self.assertEqual(len(profiler.stage_samples["Primary YOLO11m"]), 4)
        stats = profiler.calculate_stats(profiler.stage_samples["Primary YOLO11m"])
        self.assertEqual(stats["count"], 4)
        self.assertAlmostEqual(stats["min"], 800.0)
        self.assertAlmostEqual(stats["max"], 950.0)
        self.assertAlmostEqual(stats["mean"], 875.0)
        self.assertAlmostEqual(stats["median"], 875.0)
        self.assertGreater(stats["std"], 0.0)

        # CSV writes should be tracked
        self.assertEqual(len(profiler.csv_positive_writes), 1)
        self.assertEqual(profiler.csv_positive_writes[0], 15.0)

        # End-to-end should have 4 samples
        e2e_stats = profiler.calculate_stats(profiler.stage_samples["End-to-End Pipeline"])
        self.assertEqual(e2e_stats["count"], 4)
        self.assertAlmostEqual(e2e_stats["mean"], 875.0 + 140.0)


class TestStagedWebcamStartup(unittest.TestCase):
    """Tests for fast webcam startup, backend fallback cleanup, and lazy ML imports."""

    def test_lazy_app_imports_do_not_load_torch(self):
        import subprocess
        import sys
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import app.sources, app.visualizer; "
                "assert 'torch' not in sys.modules, 'torch eagerly imported'; "
                "assert 'ultralytics' not in sys.modules, 'ultralytics eagerly imported'",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)

    def test_webcam_source_releases_failed_capture_on_fallback(self):
        from unittest.mock import MagicMock, patch
        from app.sources import WebcamSource

        failed_cap = MagicMock()
        failed_cap.isOpened.return_value = False
        ok_cap = MagicMock()
        ok_cap.isOpened.return_value = True
        ok_cap.getBackendName.return_value = "MSMF"
        ok_cap.get.side_effect = lambda prop: 640.0 if prop == 3 else (480.0 if prop == 4 else 30.0)

        with patch("app.sources.cv2.VideoCapture", side_effect=[failed_cap, ok_cap]):
            src = WebcamSource(camera_index=0, backend=700)
            opened = src.open()

        self.assertTrue(opened)
        failed_cap.release.assert_called_once()
        self.assertEqual(src.backend_name, "MSMF")
        src.close()
        ok_cap.release.assert_called_once()

    def test_async_detector_loader_error_propagation(self):
        import main

        def failing_factory():
            raise RuntimeError("boom")

        loader = main.AsyncDetectorLoader(failing_factory)
        loader.start()
        self.assertTrue(loader.ready_event.wait(timeout=2.0))
        self.assertIsInstance(loader.error, RuntimeError)
        self.assertIsNone(loader.detector)

    def test_staged_webcam_startup_previews_before_model_ready(self):
        import threading
        from unittest.mock import MagicMock, patch
        import main
        from app.config import AppConfig
        from app.sources import BaseSource

        class FakeWebcam(BaseSource):
            def __init__(self):
                self.reads = 0
                self.backend_name = "DSHOW"
                self.open_ms = 12.3
            def open(self) -> bool:
                return True
            def read_frame(self):
                self.reads += 1
                if self.reads > 20:
                    return False, None
                return True, np.zeros((240, 320, 3), dtype=np.uint8)
            def close(self) -> None:
                pass
            @property
            def source_type(self) -> str:
                return "webcam"
            @property
            def reference_string(self) -> str:
                return "FakeWebcam #0"
            @property
            def dimensions(self):
                return (320, 240)
            @property
            def native_fps(self) -> float:
                return 30.0

        allow_model = threading.Event()
        preview_calls = []

        mock_det = MagicMock()
        mock_det.device = "cpu"
        mock_det.model_name = "mock-yolo"
        mock_det.weights_path = "mock.pt"
        mock_det.conf = 0.25
        mock_det.imgsz = 640
        mock_det.class_mapping = {0: "organic"}
        mock_det.sha256 = "0" * 64
        mock_det.primary_load_time_ms = 100.0
        mock_det.secondary_load_time_ms = 20.0
        mock_det.verifier.is_available = False
        mock_det.verifier.weights_path = None
        mock_det.verifier.last_inference_ms = 0.0
        mock_det.verifier.last_filtering_ms = 0.0
        mock_det.track.return_value = []
        mock_det.verify_candidates.return_value = ([], [], {"rejected": 0, "ambiguous": 0, "accepted": 0})

        def gated_factory():
            allow_model.wait(timeout=2.0)
            return mock_det

        class FakePreview:
            def __init__(self, window_title: str):
                self.window_title = window_title
            def update_frame_profiled(self, frame, detections, wait_ms=1, hud_text=None):
                preview_calls.append((hud_text, len(detections)))
                if len(preview_calls) == 2:
                    allow_model.set()
                    import time as _t
                    _t.sleep(0.02)
                elif hud_text is not None:
                    import time as _t
                    _t.sleep(0.01)
                return "continue", {"annotation_ms": 0.1, "imshow_ms": 0.1, "waitkey_ms": 0.1}
            def close(self):
                pass

        with tempfile.TemporaryDirectory() as td:
            cfg = AppConfig()
            cfg.workspace_root = Path(td)
            timings = {}
            with patch.object(main, "VisualPreviewWindow", FakePreview):
                summary = main.run_session(
                    FakeWebcam(),
                    detector=None,
                    config=cfg,
                    show_preview=True,
                    detector_factory=gated_factory,
                    startup_timings=timings,
                )

        self.assertIsNotNone(summary)
        # First preview frames showed 'Initializing models...' before model became ready
        self.assertGreaterEqual(len(preview_calls), 2)
        self.assertEqual(preview_calls[0][0], "Initializing models...")
        self.assertEqual(preview_calls[0][1], 0)
        # Detection ran after model became ready
        self.assertGreaterEqual(mock_det.track.call_count, 1)
        for key in (
            "imports_ms",
            "cam_open_ms",
            "first_frame_ms",
            "preview_create_ms",
            "first_preview_ms",
            "time_to_first_preview_ms",
            "primary_load_ms",
            "secondary_load_ms",
            "time_to_detection_active_ms",
            "first_inference_ms",
        ):
            self.assertIn(key, timings)

    def test_model_and_checksum_reuse_across_calls(self):
        from unittest.mock import patch
        from app import model_loader
        from app.config import AppConfig

        cfg = AppConfig()
        # Clear caches to test cold-to-warm reuse within the same process
        with model_loader._CACHE_LOCK:
            model_loader._VERIFIED_SHA256_CACHE.clear()
            model_loader._PRIMARY_MODEL_CACHE.clear()

        sha_calls = 0
        orig_compute = model_loader.compute_sha256

        def counting_sha(p):
            nonlocal sha_calls
            sha_calls += 1
            return orig_compute(p)

        with patch("app.model_loader.compute_sha256", side_effect=counting_sha):
            det1 = WasteDetector(
                confidence_threshold=cfg.confidence_threshold,
                image_size=cfg.image_size,
                device="cpu",
                verifier_weights_path=cfg.verifier_weights_path,
                verifier_confidence=cfg.verifier_confidence,
                verifier_enabled=cfg.verifier_enabled,
            )
            det2 = WasteDetector(
                confidence_threshold=cfg.confidence_threshold,
                image_size=cfg.image_size,
                device="cpu",
                verifier_weights_path=cfg.verifier_weights_path,
                verifier_confidence=cfg.verifier_confidence,
                verifier_enabled=cfg.verifier_enabled,
            )

        # SHA-256 computed at most once for both detectors combined
        self.assertEqual(sha_calls, 1)
        # Underlying primary & secondary models are reused in memory
        self.assertIs(det1.model, det2.model)
        if det1.verifier.is_available and det2.verifier.is_available:
            self.assertIs(det1.verifier.model, det2.verifier.model)


class TestP1CsvPersistenceReliability(unittest.TestCase):
    """P1 regression tests: truthful CSV persistence, lock handling, retry, and deduplication."""

    def _make_event(self, event_id: str = "evt_p1_001", track_id: int = 1):
        from app.storage import DetectionEvent
        return DetectionEvent(
            record_id=f"rec_{event_id}",
            session_id="sess_p1",
            event_id=event_id,
            track_id=track_id,
            detected_at="2026-11-12T12:00:00",
            source_type="webcam",
            source_name="Webcam #0",
            frame_number=1,
            class_id=1,
            class_name="recyclable",
            confidence=0.85,
            bbox_x1=10,
            bbox_y1=10,
            bbox_x2=50,
            bbox_y2=50,
            model_name="YOLO11m",
            checkpoint_filename="waste_detection_yolo11m.pt",
        )

    def test_p1a_normal_csv_write(self):
        import csv
        from app.storage import StorageManager

        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "waste_detections.csv"
            sm = StorageManager(csv_path=csv_path, session_id="s1", source_type="webcam", source_reference="Webcam #0")
            ev = self._make_event("evt_normal_1")
            ok = sm.record_event(ev)

            self.assertTrue(ok)
            self.assertEqual(sm.total_events, 1)
            self.assertEqual(sm.pending_events_count, 0)
            self.assertIn("evt_normal_1", sm._written_event_ids)
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["event_id"], "evt_normal_1")

    def test_p1b_csv_locked_permission_error_and_tracker_output(self):
        import io
        from contextlib import redirect_stdout
        from unittest.mock import patch
        from app.storage import StorageManager
        from app.tracking import DetectionTracker

        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "waste_detections.csv"
            sm = StorageManager(csv_path=csv_path, session_id="s1", source_type="webcam", source_reference="Webcam #0")
            callbacks = []
            tracker = DetectionTracker(storage_manager=sm, on_new_event=lambda e: callbacks.append(e.event_id))

            det = Detection(class_id=1, class_name="recyclable", confidence=0.85, bbox_x=10, bbox_y=10, bbox_w=40, bbox_h=40, track_id=7)
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            buf = io.StringIO()
            with patch("builtins.open", side_effect=PermissionError("Locked by Excel")):
                with redirect_stdout(buf):
                    tracker.process_frame([det], frame, frame_index=1)

            self.assertEqual(sm.total_events, 0)
            self.assertEqual(sm.pending_events_count, 1)
            self.assertEqual(sm.write_count, 0)
            self.assertEqual(len(callbacks), 0)
            self.assertIn("[STORAGE ERROR]", buf.getvalue())
            self.assertIn("Locked by Excel", buf.getvalue())

    def test_p1c_retry_after_unlock_preserves_event_id_and_no_duplicates(self):
        import csv
        from unittest.mock import patch
        from app.storage import StorageManager

        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "waste_detections.csv"
            sm = StorageManager(csv_path=csv_path, session_id="s1", source_type="webcam", source_reference="Webcam #0")
            ev = self._make_event("evt_retry_1")

            with patch("builtins.open", side_effect=PermissionError("Locked")):
                self.assertFalse(sm.record_event(ev))

            self.assertEqual(sm.total_events, 0)
            self.assertEqual(sm.pending_events_count, 1)

            # Unlock and retry
            saved = sm.retry_pending_events(force=True)
            self.assertEqual(saved, 1)
            self.assertEqual(sm.total_events, 1)
            self.assertEqual(sm.pending_events_count, 0)

            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["event_id"], "evt_retry_1")

    def test_p1d_duplicate_event_callback_prevented(self):
        import csv
        from app.storage import StorageManager

        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "waste_detections.csv"
            sm = StorageManager(csv_path=csv_path, session_id="s1", source_type="webcam", source_reference="Webcam #0")
            ev = self._make_event("evt_dup_1")
            self.assertTrue(sm.record_event(ev))
            self.assertTrue(sm.record_event(ev))

            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)

    def test_p1e_existing_csv_preservation(self):
        import csv
        from app.storage import StorageManager

        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "waste_detections.csv"
            sm1 = StorageManager(csv_path=csv_path, session_id="s1", source_type="webcam", source_reference="Webcam #0")
            self.assertTrue(sm1.record_event(self._make_event("evt_hist_1", track_id=1)))

            sm2 = StorageManager(csv_path=csv_path, session_id="s2", source_type="webcam", source_reference="Webcam #0")
            self.assertTrue(sm2.record_event(self._make_event("evt_hist_2", track_id=2)))

            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([r["event_id"] for r in rows], ["evt_hist_1", "evt_hist_2"])


class TestP2WebcamReadTimeout(unittest.TestCase):
    """P2 regression tests: bounded webcam reads, stall handling, and safe owner-only release."""

    def test_p2a_camera_returns_first_frame_normally(self):
        from unittest.mock import MagicMock, patch
        from app.sources import WebcamSource

        ok_cap = MagicMock()
        ok_cap.isOpened.return_value = True
        ok_cap.getBackendName.return_value = "DSHOW"
        ok_cap.get.side_effect = lambda prop: 640.0 if prop == 3 else (480.0 if prop == 4 else 30.0)
        ok_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))

        with patch("app.sources.cv2.VideoCapture", return_value=ok_cap):
            src = WebcamSource(camera_index=41, read_timeout_sec=1.0)
            self.assertTrue(src.open())
            ret, frame = src.read_frame()
            self.assertTrue(ret)
            self.assertIsNotNone(frame)
            self.assertFalse(src.timed_out)
            src.close()
            self.assertFalse(src.cleanup_pending)
            ok_cap.release.assert_called_once()

    def test_p2b_camera_never_returns_first_frame(self):
        import threading
        import time
        from unittest.mock import MagicMock, patch
        import main
        from app.config import AppConfig
        from app.sources import WebcamSource

        unblock = threading.Event()
        blocked_cap = MagicMock()
        blocked_cap.isOpened.return_value = True
        blocked_cap.getBackendName.return_value = "DSHOW"
        blocked_cap.get.side_effect = lambda prop: 640.0 if prop == 3 else (480.0 if prop == 4 else 30.0)

        def hanging_read():
            unblock.wait(timeout=5.0)
            return False, None

        blocked_cap.read.side_effect = hanging_read

        try:
            with patch("app.sources.cv2.VideoCapture", return_value=blocked_cap):
                with tempfile.TemporaryDirectory() as td:
                    cfg = AppConfig()
                    cfg.workspace_root = Path(td)
                    src = WebcamSource(camera_index=42, read_timeout_sec=0.15)
                    factory_called = False

                    def never_called_factory():
                        nonlocal factory_called
                        factory_called = True
                        raise AssertionError("Detector factory should not start when first frame times out")

                    t0 = time.monotonic()
                    summary = main.run_session(
                        src,
                        detector=None,
                        config=cfg,
                        show_preview=False,
                        detector_factory=never_called_factory,
                    )
                    elapsed = time.monotonic() - t0

            self.assertIsNone(summary)
            self.assertTrue(src.timed_out)
            self.assertFalse(factory_called)
            # Bounded timeout + close() wait (~1.15s) must finish well under 3s
            self.assertLess(elapsed, 3.0)
        finally:
            unblock.set()
            src._reader_exited.wait(timeout=1.0)

    def test_p2c_camera_stalls_after_initialization(self):
        import threading
        from unittest.mock import MagicMock, patch
        from app.sources import WebcamSource

        unblock = threading.Event()
        call_count = 0

        def flaky_read():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return True, np.zeros((240, 320, 3), dtype=np.uint8)
            unblock.wait(timeout=5.0)
            return False, None

        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.getBackendName.return_value = "DSHOW"
        cap.get.side_effect = lambda prop: 320.0 if prop == 3 else (240.0 if prop == 4 else 30.0)
        cap.read.side_effect = flaky_read

        try:
            with patch("app.sources.cv2.VideoCapture", return_value=cap):
                src = WebcamSource(camera_index=43, read_timeout_sec=0.15)
                self.assertTrue(src.open())
                ret1, _ = src.read_frame()
                self.assertTrue(ret1)
                # Wait until reader advances to blocked 3rd call, then read until timeout
                for _ in range(3):
                    ret, _ = src.read_frame()
                    if src.timed_out:
                        break
                self.assertTrue(src.timed_out)
                self.assertFalse(ret)
        finally:
            unblock.set()
            src.close()

    def test_p2d_reader_shutdown_and_no_duplicate_reader(self):
        import threading
        from unittest.mock import MagicMock, patch
        from app.sources import WebcamSource

        unblock = threading.Event()
        in_read = threading.Event()
        blocked_cap = MagicMock()
        blocked_cap.isOpened.return_value = True
        blocked_cap.getBackendName.return_value = "DSHOW"
        blocked_cap.get.side_effect = lambda prop: 640.0 if prop == 3 else (480.0 if prop == 4 else 30.0)

        def hanging_read():
            in_read.set()
            unblock.wait(timeout=5.0)
            return False, None

        blocked_cap.read.side_effect = hanging_read

        try:
            with patch("app.sources.cv2.VideoCapture", return_value=blocked_cap):
                src1 = WebcamSource(camera_index=44, read_timeout_sec=0.1)
                self.assertTrue(src1.open())
                self.assertTrue(in_read.wait(timeout=1.0))
                src1.close()
                # Must not falsely claim released or release concurrently while blocked
                self.assertTrue(src1.cleanup_pending)
                blocked_cap.release.assert_not_called()

                # Duplicate open for same camera_index must fail while cleanup is pending
                src2 = WebcamSource(camera_index=44, read_timeout_sec=0.1)
                self.assertFalse(src2.open())
                self.assertIn("cleanup is still pending", src2.last_error)
        finally:
            unblock.set()
            self.assertTrue(src1._reader_exited.wait(timeout=1.0))
            blocked_cap.release.assert_called_once()

    def test_p2e_ctrl_c_shutdown_preserves_csv(self):
        from unittest.mock import MagicMock
        import main
        from app.config import AppConfig
        from app.sources import BaseSource

        class InterruptCam(BaseSource):
            def __init__(self):
                self.reads = 0
                self.closed = False
            def open(self) -> bool:
                return True
            def read_frame(self):
                self.reads += 1
                if self.reads >= 3:
                    raise KeyboardInterrupt()
                return True, np.zeros((240, 320, 3), dtype=np.uint8)
            def close(self) -> None:
                self.closed = True
            @property
            def source_type(self) -> str:
                return "webcam"
            @property
            def reference_string(self) -> str:
                return "InterruptCam #0"
            @property
            def dimensions(self):
                return (320, 240)
            @property
            def native_fps(self) -> float:
                return 30.0

        mock_det = MagicMock()
        mock_det.device = "cpu"
        mock_det.model_name = "mock"
        mock_det.weights_path = "mock.pt"
        mock_det.conf = 0.25
        mock_det.imgsz = 640
        mock_det.class_mapping = {0: "organic"}
        mock_det.sha256 = "0" * 64
        mock_det.primary_load_time_ms = 1.0
        mock_det.secondary_load_time_ms = 1.0
        mock_det.verifier.is_available = False
        mock_det.verifier.weights_path = None
        mock_det.verifier.last_inference_ms = 0.0
        mock_det.verifier.last_filtering_ms = 0.0
        mock_det.track.return_value = []
        mock_det.verify_candidates.return_value = ([], [], {"rejected": 0, "ambiguous": 0, "accepted": 0})

        with tempfile.TemporaryDirectory() as td:
            cfg = AppConfig()
            cfg.workspace_root = Path(td)
            src = InterruptCam()
            summary = main.run_session(src, mock_det, cfg, show_preview=False)
            self.assertIsNotNone(summary)
            self.assertTrue(src.closed)
            self.assertTrue((Path(td) / "waste_detections.csv").exists())


class TestP3ByteTrackSessionIsolation(unittest.TestCase):
    """P3 regression tests: session-boundary ByteTrack reset & model cache preservation."""

    @classmethod
    def setUpClass(cls):
        from app.config import AppConfig
        cls.cfg = AppConfig()
        cls.detector = WasteDetector(
            confidence_threshold=cls.cfg.confidence_threshold,
            image_size=320,
            device="cpu",
            verifier_weights_path=cls.cfg.verifier_weights_path,
            verifier_confidence=cls.cfg.verifier_confidence,
            verifier_enabled=False,
        )

    def _make_video_source(self, name: str, num_frames: int, raise_on_frame: int = 0):
        from app.sources import BaseSource

        class FakeVideo(BaseSource):
            def __init__(self):
                self.n = 0
                self.closed = False
            def open(self) -> bool:
                return True
            def read_frame(self):
                self.n += 1
                if raise_on_frame and self.n == raise_on_frame:
                    raise RuntimeError("simulated stream error")
                if self.n > num_frames:
                    return False, None
                return True, np.zeros((240, 320, 3), dtype=np.uint8)
            def close(self) -> None:
                self.closed = True
            @property
            def source_type(self) -> str:
                return "video"
            @property
            def reference_string(self) -> str:
                return name
            @property
            def dimensions(self):
                return (320, 240)
            @property
            def native_fps(self) -> float:
                return 30.0

        return FakeVideo()

    def test_p3a_cold_tracker_initialization(self):
        from unittest.mock import MagicMock
        det = WasteDetector.__new__(WasteDetector)
        det.model = MagicMock()
        det.model.predictor = None
        det.verifier = None

        reset_cnt = det.reset_tracking()
        self.assertEqual(reset_cnt, 0)
        self.assertIsNone(det.model.predictor)

    def test_p3b_sequential_sessions_real_ultralytics_bytetrack(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        self.detector.reset_tracking()

        # Session 1: run 2 tracked frames using real Ultralytics BYTETracker
        self.detector.track(frame)
        self.detector.track(frame)
        predictor = getattr(self.detector.model, "predictor", None)
        self.assertIsNotNone(predictor)
        trackers = getattr(predictor, "trackers", [])
        self.assertGreaterEqual(len(trackers), 1)
        self.assertEqual(trackers[0].frame_id, 2)

        # Boundary reset between independent sessions
        reset_cnt = self.detector.reset_tracking()
        self.assertEqual(reset_cnt, len(trackers))
        self.assertEqual(trackers[0].frame_id, 0)
        self.assertEqual(trackers[0].tracked_stracks, [])
        self.assertEqual(trackers[0].lost_stracks, [])
        self.assertEqual(trackers[0].removed_stracks, [])

        # Session 2: first frame starts cleanly at frame_id == 1
        self.detector.track(frame)
        self.assertEqual(trackers[0].frame_id, 1)
        self.detector.reset_tracking()

    def test_p3c_intra_session_continuity_real_ultralytics(self):
        import main
        from app.config import AppConfig

        observed_frame_ids = []
        orig_track = self.detector.track

        def spy_track(frame):
            res = orig_track(frame)
            trk = self.detector.model.predictor.trackers[0]
            observed_frame_ids.append(trk.frame_id)
            return res

        with tempfile.TemporaryDirectory() as td:
            cfg = AppConfig()
            cfg.workspace_root = Path(td)
            src = self._make_video_source("clip_a.mp4", num_frames=3)
            self.detector.track = spy_track
            try:
                summary = main.run_session(src, self.detector, cfg, show_preview=False)
            finally:
                self.detector.track = orig_track

        self.assertIsNotNone(summary)
        # Within the session, ByteTrack persisted across frames 1 -> 2 -> 3
        self.assertEqual(observed_frame_ids, [1, 2, 3])
        # After session exit, tracker state is reset
        self.assertEqual(self.detector.model.predictor.trackers[0].frame_id, 0)

    def test_p3d_model_cache_preservation(self):
        from unittest.mock import patch
        import main
        from app import model_loader
        from app.config import AppConfig

        model_before = self.detector.model
        sha_calls = 0
        orig_sha = model_loader.compute_sha256

        def count_sha(p):
            nonlocal sha_calls
            sha_calls += 1
            return orig_sha(p)

        with tempfile.TemporaryDirectory() as td:
            cfg = AppConfig()
            cfg.workspace_root = Path(td)
            with patch("app.model_loader.compute_sha256", side_effect=count_sha):
                main.run_session(self._make_video_source("s1.mp4", 1), self.detector, cfg, show_preview=False)
                main.run_session(self._make_video_source("s2.mp4", 1), self.detector, cfg, show_preview=False)

        self.assertIs(self.detector.model, model_before)
        self.assertEqual(sha_calls, 0)

    def test_p3e_independent_session_identity_and_verifier_cache(self):
        from app.detector import VerificationState

        # Simulate stale active/lost tracks + verifier cache from Session 1
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        self.detector.track(frame)
        trk = self.detector.model.predictor.trackers[0]
        trk.tracked_stracks = ["stale_active_track"]
        trk.lost_stracks = ["stale_lost_track"]
        self.detector.verifier._track_cache[1] = ((10, 10, 20, 20), 5, VerificationState.ACCEPTED, "stale")
        self.detector.verifier._frame_counter = 5

        self.detector.reset_tracking()
        self.assertEqual(trk.tracked_stracks, [])
        self.assertEqual(trk.lost_stracks, [])
        self.assertEqual(self.detector.verifier._track_cache, {})
        self.assertEqual(self.detector.verifier._frame_counter, 0)

    def test_p3f_error_lifecycle_resets_tracker(self):
        import main
        from app.config import AppConfig

        with tempfile.TemporaryDirectory() as td:
            cfg = AppConfig()
            cfg.workspace_root = Path(td)
            # Frame 1 tracks, Frame 2 raises RuntimeError
            src = self._make_video_source("err.mp4", num_frames=3, raise_on_frame=2)
            main.run_session(src, self.detector, cfg, show_preview=False)

        # finally: block must have reset ByteTrack after exception
        self.assertEqual(self.detector.model.predictor.trackers[0].frame_id, 0)
        self.assertEqual(self.detector.model.predictor.trackers[0].tracked_stracks, [])

    def test_p3g_detection_tracker_reset_preserves_csv(self):
        import csv
        from app.storage import StorageManager
        from app.tracking import DetectionTracker

        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "waste_detections.csv"
            sm = StorageManager(csv_path=csv_path, session_id="s1", source_type="video", source_reference="v1.mp4")
            dt = DetectionTracker(storage_manager=sm)
            det = Detection(class_id=1, class_name="recyclable", confidence=0.8, bbox_x=10, bbox_y=10, bbox_w=30, bbox_h=30, track_id=5)
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            dt.process_frame([det], frame, frame_index=1)
            self.assertEqual(len(dt._active_tracks), 1)
            self.assertEqual(sm.total_events, 1)

            dt.reset_session()
            self.assertEqual(len(dt._active_tracks), 0)
            self.assertEqual(dt._next_fallback_id, 1000)
            self.assertEqual(sm.total_events, 1)
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()


"""Unit tests for video sources and background video recorder."""

import tempfile
import unittest
from pathlib import Path
import numpy as np

from app.core.types import SourceType
from app.engine.recorder import VideoRecorder
from app.engine.video_source import RtspSource, VideoFileSource


class TestVideoEngine(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_video_file_source_invalid_path(self):
        """Non-existent video file path is rejected gracefully."""
        src = VideoFileSource(Path("non_existent_video.mp4"))
        self.assertFalse(src.open())
        self.assertEqual(src.source_type, SourceType.VIDEO_FILE)

    def test_rtsp_source_sanitization_and_bounds(self):
        """RTSP source bounds retries and sanitizes passwords."""
        url = "rtsp://operator:topsecret@10.0.0.50:554/cam"
        src = RtspSource(url, max_reconnect_attempts=3, initial_delay_sec=0.1, max_delay_sec=0.5)
        self.assertNotIn("topsecret", src.reference_string)
        self.assertEqual(src.max_reconnect_attempts, 3)

    def test_video_recorder_start_write_stop(self):
        """Recorder starts background thread, writes pristine frames, and closes cleanly."""
        out_path = self.root_path / "test_rec.mp4"
        recorder = VideoRecorder(output_path=out_path, frame_size=(320, 240), fps=20.0, max_queue_size=20)
        started = recorder.start()
        self.assertTrue(started)

        # Write 5 dummy frames
        frame = np.full((240, 320, 3), 100, dtype=np.uint8)
        for _ in range(5):
            recorder.write_frame(frame)

        recorder.stop()
        self.assertFalse(recorder._running)
        self.assertTrue(out_path.exists())
        self.assertGreater(out_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()


"""Session Video Recording subsystem.

Captures pristine, full-resolution source video without bounding box overlays
in an isolated background worker with bounded queue controls to protect RAM.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoRecorder:
    """Encodes pristine source frames into standard MP4/AVI session recordings."""

    def __init__(
        self,
        output_path: Path,
        frame_size: Tuple[int, int],
        fps: float = 25.0,
        max_queue_size: int = 90,
    ):
        self.output_path = Path(output_path)
        self.frame_size = frame_size  # (width, height)
        self.fps = max(5.0, min(60.0, float(fps)))
        self.queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=max_queue_size)

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._writer: Optional[cv2.VideoWriter] = None
        self._frames_written = 0
        self._dropped_frames = 0

    def start(self) -> bool:
        """Initialize writer and launch recording worker thread."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        w, h = self.frame_size
        codecs_to_try = [
            ("mp4v", ".mp4"),
            ("avc1", ".mp4"),
            ("XVID", ".avi"),
        ]

        writer = None
        for codec_str, ext in codecs_to_try:
            fourcc = cv2.VideoWriter_fourcc(*codec_str)
            test_target = self.output_path
            if test_target.suffix.lower() != ext:
                test_target = test_target.with_suffix(ext)

            writer = cv2.VideoWriter(str(test_target), fourcc, self.fps, (w, h))
            if writer.isOpened():
                self.output_path = test_target
                logger.info("VideoWriter initialized: %s with codec %s", self.output_path, codec_str)
                break
            else:
                logger.warning("Codec %s failed for %s", codec_str, test_target)
                writer = None

        if not writer or not writer.isOpened():
            logger.error("Failed to initialize any VideoWriter codec for %s", self.output_path)
            return False

        self._writer = writer
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, name="VideoRecorderWorker", daemon=True)
        self._thread.start()
        return True

    def write_frame(self, raw_frame: np.ndarray) -> None:
        """Enqueue a pristine source frame. Drops frame if queue is full to prevent OOM."""
        if not self._running:
            return

        try:
            self.queue.put_nowait(raw_frame)
        except queue.Full:
            self._dropped_frames += 1
            if self._dropped_frames % 30 == 1:
                logger.warning(
                    "Recorder queue full: dropped %d frames to prevent memory exhaustion.",
                    self._dropped_frames,
                )

    def _worker_loop(self) -> None:
        """Background thread consuming frames and writing to disk."""
        while self._running or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if frame is None:
                break

            try:
                fh, fw = frame.shape[:2]
                w, h = self.frame_size
                if fw != w or fh != h:
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

                self._writer.write(frame)
                self._frames_written += 1
            except Exception as e:
                logger.error("Error writing frame to recording: %s", e)
            finally:
                self.queue.task_done()

    def stop(self) -> None:
        """Finalize video recording safely and wait for pending frames."""
        if not self._running:
            return

        logger.info("Stopping VideoRecorder... (flushing %d frames)", self.queue.qsize())
        self._running = False
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

        if self._writer:
            try:
                self._writer.release()
            except Exception as e:
                logger.error("Error releasing VideoWriter: %s", e)
            self._writer = None

        logger.info(
            "Video recording finalized: %s (%d frames written, %d dropped)",
            self.output_path,
            self._frames_written,
            self._dropped_frames,
        )

    @property
    def final_file_path(self) -> str:
        return str(self.output_path) if self.output_path.exists() else ""

    @property
    def frames_written(self) -> int:
        return self._frames_written


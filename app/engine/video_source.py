"""Unified Video Source abstraction layer.

Provides robust, crash-resilient capture adapters for Built-in/USB Webcams,
RTSP/IP Network Streams with bounded exponential reconnection, and Prerecorded Video Files.
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from app.core.config import sanitize_source_reference
from app.core.types import SourceType

logger = logging.getLogger(__name__)


class BaseVideoSource(ABC):
    """Abstract interface for video capture sources."""

    @abstractmethod
    def open(self) -> bool:
        """Open and initialize video stream. Returns True if successful."""
        pass

    @abstractmethod
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the next frame. Returns (success, frame_bgr)."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Release video capture resources."""
        pass

    @property
    @abstractmethod
    def source_type(self) -> SourceType:
        pass

    @property
    @abstractmethod
    def reference_string(self) -> str:
        """Display reference string (sanitized)."""
        pass

    @property
    @abstractmethod
    def native_fps(self) -> float:
        pass

    @property
    @abstractmethod
    def dimensions(self) -> Tuple[int, int]:
        """(width, height)"""
        pass


class WebcamSource(BaseVideoSource):
    """DirectShow/V4L2 webcam capture adapter."""

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 30.0
        self._width: int = 640
        self._height: int = 480

    def open(self) -> bool:
        # On Windows, prefer CAP_DSHOW for fast initialization and reliable directshow capture
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(self.camera_index, backend)
        if not self._cap or not self._cap.isOpened():
            # Fall back to default backend
            self._cap = cv2.VideoCapture(self.camera_index)

        if not self._cap.isOpened():
            logger.error("Failed to open webcam index %d", self.camera_index)
            return False

        # Read actual capabilities
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        if w > 0 and h > 0:
            self._width = w
            self._height = h
        if fps > 0:
            self._fps = fps

        logger.info("Webcam %d opened: %dx%d @ %.1f FPS", self.camera_index, self._width, self._height, self._fps)
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._cap or not self._cap.isOpened():
            return False, None
        ret, frame = self._cap.read()
        return ret, frame

    def close(self) -> None:
        if self._cap:
            try:
                self._cap.release()
            except Exception as e:
                logger.error("Error releasing webcam: %s", e)
            self._cap = None
            logger.info("Webcam %d released.", self.camera_index)

    @property
    def source_type(self) -> SourceType:
        return SourceType.WEBCAM

    @property
    def reference_string(self) -> str:
        return f"Webcam #{self.camera_index}"

    @property
    def native_fps(self) -> float:
        return self._fps

    @property
    def dimensions(self) -> Tuple[int, int]:
        return self._width, self._height


class RtspSource(BaseVideoSource):
    """RTSP/IP Network camera adapter with bounded exponential backoff reconnection."""

    def __init__(
        self,
        rtsp_url: str,
        max_reconnect_attempts: int = 5,
        initial_delay_sec: float = 2.0,
        max_delay_sec: float = 15.0,
    ):
        self.rtsp_url = rtsp_url
        self.max_reconnect_attempts = max_reconnect_attempts
        self.initial_delay_sec = initial_delay_sec
        self.max_delay_sec = max_delay_sec

        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 25.0
        self._width: int = 1280
        self._height: int = 720
        self._reconnect_count: int = 0

    def open(self) -> bool:
        return self._connect()

    def _connect(self) -> bool:
        if self._cap:
            self._cap.release()

        # Set FFMPEG network timeouts via environment or parameters if supported
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"
        self._cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

        if not self._cap.isOpened():
            logger.warning("RTSP connection failed: %s", self.reference_string)
            return False

        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        if w > 0 and h > 0:
            self._width = w
            self._height = h
        if fps > 0:
            self._fps = fps

        logger.info("RTSP stream connected: %s (%dx%d)", self.reference_string, self._width, self._height)
        self._reconnect_count = 0
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._cap or not self._cap.isOpened():
            if not self._attempt_reconnection():
                return False, None

        ret, frame = self._cap.read()
        if not ret:
            logger.warning("RTSP stream lost frame. Attempting reconnection...")
            if self._attempt_reconnection():
                return self._cap.read()
            return False, None
        return ret, frame

    def _attempt_reconnection(self) -> bool:
        """Bounded exponential backoff reconnection."""
        while self._reconnect_count < self.max_reconnect_attempts:
            self._reconnect_count += 1
            delay = min(self.max_delay_sec, self.initial_delay_sec * (1.5 ** (self._reconnect_count - 1)))
            logger.info("Reconnection attempt %d/%d in %.1fs...", self._reconnect_count, self.max_reconnect_attempts, delay)
            time.sleep(delay)
            if self._connect():
                return True

        logger.error("RTSP stream exceeded maximum reconnect attempts (%d).", self.max_reconnect_attempts)
        return False

    def close(self) -> None:
        if self._cap:
            try:
                self._cap.release()
            except Exception as e:
                logger.error("Error releasing RTSP source: %s", e)
            self._cap = None

    @property
    def source_type(self) -> SourceType:
        return SourceType.RTSP

    @property
    def reference_string(self) -> str:
        return sanitize_source_reference(self.rtsp_url)

    @property
    def native_fps(self) -> float:
        return self._fps

    @property
    def dimensions(self) -> Tuple[int, int]:
        return self._width, self._height


class VideoFileSource(BaseVideoSource):
    """Prerecorded video file playback adapter.

    Supports original-speed paced playback and accelerated full-speed analysis mode.
    Preserves original video files without in-place modification or redundant disk copying.
    """

    def __init__(self, file_path: Path, accelerated: bool = False):
        self.file_path = Path(file_path)
        self.accelerated = accelerated
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 30.0
        self._width: int = 1280
        self._height: int = 720
        self._total_frames: int = 0
        self._last_frame_time: float = 0.0

    def open(self) -> bool:
        if not self.file_path.exists():
            logger.error("Video file does not exist: %s", self.file_path)
            return False

        self._cap = cv2.VideoCapture(str(self.file_path))
        if not self._cap.isOpened():
            logger.error("Failed opening video file: %s", self.file_path)
            return False

        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if w > 0 and h > 0:
            self._width = w
            self._height = h
        if fps > 0:
            self._fps = fps
        self._total_frames = max(0, total)
        self._last_frame_time = time.monotonic()

        logger.info(
            "Video file opened: %s (%dx%d @ %.1f FPS, %d frames, accelerated=%s)",
            self.file_path.name, self._width, self._height, self._fps, self._total_frames, self.accelerated
        )
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._cap or not self._cap.isOpened():
            return False, None

        if not self.accelerated and self._fps > 0:
            # Sync to real-time playback speed
            target_delay = 1.0 / self._fps
            elapsed = time.monotonic() - self._last_frame_time
            if elapsed < target_delay:
                time.sleep(target_delay - elapsed)
            self._last_frame_time = time.monotonic()

        ret, frame = self._cap.read()
        return ret, frame

    def close(self) -> None:
        if self._cap:
            try:
                self._cap.release()
            except Exception as e:
                logger.error("Error releasing video file: %s", e)
            self._cap = None

    @property
    def source_type(self) -> SourceType:
        return SourceType.VIDEO_FILE

    @property
    def reference_string(self) -> str:
        return self.file_path.name

    @property
    def native_fps(self) -> float:
        return self._fps

    @property
    def dimensions(self) -> Tuple[int, int]:
        return self._width, self._height

    @property
    def total_frames(self) -> int:
        return self._total_frames


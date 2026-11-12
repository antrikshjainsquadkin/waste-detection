"""Video and image input sources.

Provides robust, crash-resilient capture adapters for:
1. Webcams (USB / integrated)
2. RTSP / IP Network Cameras (with credential protection and bounded retry)
3. Prerecorded Video Files
4. Still Images
"""

from __future__ import annotations

import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from app.config import sanitize_source_reference

logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".m4v"}
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def resolve_video_path(path: Path | str) -> Tuple[Optional[Path], str]:
    """Resolve a user-provided video path, gracefully handling folders, quotes, and extensions."""
    clean_str = str(path).strip().strip('"').strip("'")
    p = Path(clean_str)

    if not p.exists():
        # Check if user omitted extension
        for ext in SUPPORTED_VIDEO_EXTS:
            test_p = p.with_suffix(ext)
            if test_p.is_file():
                return test_p, f"Found video by adding extension: '{test_p.name}'"
        return None, f"Video path does not exist: {clean_str}"

    if p.is_dir():
        candidates = sorted([f for f in p.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_EXTS])
        if not candidates:
            return None, f"'{clean_str}' is a folder, but contains no supported video files (.mp4, .avi, etc.)."
        selected = candidates[0]
        return selected, f"Target is a folder. Automatically selected video: '{selected.name}'"

    if p.is_file():
        return p, ""

    return None, f"Invalid path: {clean_str}"


def resolve_image_path(path: Path | str) -> Tuple[Optional[Path], str]:
    """Resolve a user-provided image path, gracefully handling folders, quotes, and extensions."""
    clean_str = str(path).strip().strip('"').strip("'")
    p = Path(clean_str)

    if not p.exists():
        for ext in SUPPORTED_IMAGE_EXTS:
            test_p = p.with_suffix(ext)
            if test_p.is_file():
                return test_p, f"Found image by adding extension: '{test_p.name}'"
        return None, f"Image path does not exist: {clean_str}"

    if p.is_dir():
        candidates = sorted([f for f in p.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTS])
        if not candidates:
            return None, f"'{clean_str}' is a folder, but contains no supported image files (.jpg, .jpeg, .png, etc.)."
        selected = candidates[0]
        return selected, f"Target is a folder. Automatically selected image: '{selected.name}'"

    if p.is_file():
        return p, ""

    return None, f"Invalid path: {clean_str}"


def resolve_stream_url(url: str) -> Tuple[str, str]:
    """Normalize and resolve IP camera / RTSP stream URLs.

    Handles bare IP:port and web-server URLs like 'https://192.168.1.11:8080/'
    or 'http://192.168.1.11:8080', auto-resolving to standard video stream endpoints (/video).
    """
    clean_str = str(url).strip().strip('"').strip("'")
    if not clean_str:
        return "", "Stream URL cannot be empty."

    # If scheme omitted (e.g. '192.168.1.11:8080')
    if not (clean_str.startswith("rtsp://") or clean_str.startswith("http://") or clean_str.startswith("https://")):
        clean_str = f"http://{clean_str}"

    from urllib.parse import urlparse
    parsed = urlparse(clean_str)

    if parsed.scheme in ("http", "https"):
        path = parsed.path.rstrip("/")
        # If no endpoint was provided (e.g. http://192.168.1.11:8080/ or http://192.168.1.11:8080)
        if not path:
            resolved = f"{clean_str.rstrip('/')}/video"
            return resolved, f"Auto-detected IP webcam web URL. Appended stream endpoint '/video' -> {resolved}"

    return clean_str, ""


class BaseSource(ABC):
    """Abstract base class for all input sources."""

    @abstractmethod
    def open(self) -> bool:
        """Open source; return True on success."""
        pass

    @abstractmethod
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the next frame; return (success, frame_bgr)."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Release source resources."""
        pass

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Source type identifier (webcam, rtsp, video_file, image)."""
        pass

    @property
    @abstractmethod
    def reference_string(self) -> str:
        """Sanitized human-readable reference string."""
        pass

    @property
    @abstractmethod
    def dimensions(self) -> Tuple[int, int]:
        """(width, height)"""
        pass

    @property
    def native_fps(self) -> float:
        return 30.0


_WEBCAM_READERS_LOCK = threading.Lock()
_ACTIVE_WEBCAM_EXITED: Dict[int, threading.Event] = {}


class WebcamSource(BaseSource):
    """Webcam capture adapter with bounded frame-read timeout and single-owner cleanup."""

    def __init__(
        self,
        camera_index: int = 0,
        backend: Optional[int] = None,
        read_timeout_sec: float = 5.0,
    ):
        self.camera_index = int(camera_index)
        self.requested_backend = backend
        self.read_timeout_sec = max(0.05, float(read_timeout_sec))
        self._cap: Optional[cv2.VideoCapture] = None
        self._width = 640
        self._height = 480
        self._fps = 30.0
        self.open_ms: float = 0.0
        self.first_frame_ms: Optional[float] = None
        self.backend_name: str = "N/A"
        self.last_error: str = ""
        self.timed_out: bool = False
        self.cleanup_pending: bool = False

        self._reader_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_cond = threading.Condition()
        self._reader_exited = threading.Event()
        self._reader_exited.set()
        self._latest_seq: int = 0
        self._consumed_seq: int = 0
        self._latest_ret: bool = False
        self._latest_frame: Optional[np.ndarray] = None

    @property
    def cap(self) -> Optional[cv2.VideoCapture]:
        return self._cap

    def _reader_loop(self, cap: cv2.VideoCapture, exited_event: threading.Event) -> None:
        """Dedicated single-owner reader thread: sole caller of cap.read() and cap.release()."""
        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if self._stop_event.is_set():
                    break
                with self._frame_cond:
                    self._latest_ret = bool(ret)
                    self._latest_frame = frame
                    self._latest_seq += 1
                    self._frame_cond.notify_all()
                if not ret:
                    break
        except Exception as e:
            logger.error("Webcam %d reader loop error: %s", self.camera_index, e)
            with self._frame_cond:
                self._latest_ret = False
                self._latest_frame = None
                self._latest_seq += 1
                self._frame_cond.notify_all()
        finally:
            try:
                cap.release()
            except Exception as e:
                logger.error("Error releasing webcam %d in owner thread: %s", self.camera_index, e)
            self._cap = None
            with self._frame_cond:
                exited_event.set()
                self._frame_cond.notify_all()

    def open(self) -> bool:
        self.last_error = ""
        self.timed_out = False
        self.cleanup_pending = False

        # Prevent opening a duplicate reader if an earlier reader is still blocked in native read()
        with _WEBCAM_READERS_LOCK:
            prev_exited = _ACTIVE_WEBCAM_EXITED.get(self.camera_index)
            if prev_exited is not None and not prev_exited.is_set():
                self.last_error = (
                    f"Webcam #{self.camera_index} cleanup is still pending from a blocked read."
                )
                logger.error(self.last_error)
                return False

        if not self._reader_exited.is_set():
            self.last_error = (
                f"Webcam #{self.camera_index} reader is still active."
            )
            logger.error(self.last_error)
            return False

        t0 = time.perf_counter_ns()
        if self.requested_backend is not None:
            primary_backend = self.requested_backend
        else:
            primary_backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY

        cap = cv2.VideoCapture(self.camera_index, primary_backend)
        if not cap or not cap.isOpened():
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
            if primary_backend != cv2.CAP_ANY:
                cap = cv2.VideoCapture(self.camera_index, cv2.CAP_ANY)

        if not cap or not cap.isOpened():
            self.open_ms = (time.perf_counter_ns() - t0) / 1_000_000
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            self.last_error = f"Failed to open webcam index {self.camera_index}"
            logger.error(self.last_error)
            return False

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        try:
            self.backend_name = str(cap.getBackendName())
        except Exception:
            self.backend_name = "UNKNOWN"

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if w > 0 and h > 0:
            self._width = w
            self._height = h
        if fps > 0:
            self._fps = fps

        self._cap = cap
        self.first_frame_ms = None
        self._stop_event.clear()
        self._latest_seq = 0
        self._consumed_seq = 0
        self._latest_ret = False
        self._latest_frame = None
        self._reader_exited = threading.Event()
        with _WEBCAM_READERS_LOCK:
            _ACTIVE_WEBCAM_EXITED[self.camera_index] = self._reader_exited

        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            args=(cap, self._reader_exited),
            name=f"webcam-reader-{self.camera_index}",
            daemon=True,
        )
        self._reader_thread.start()

        self.open_ms = (time.perf_counter_ns() - t0) / 1_000_000
        logger.info(
            "Webcam %d opened (%s): %dx%d @ %.1f FPS in %.2f ms",
            self.camera_index,
            self.backend_name,
            self._width,
            self._height,
            self._fps,
            self.open_ms,
        )
        return True

    def read_frame(self, timeout_sec: Optional[float] = None) -> Tuple[bool, Optional[np.ndarray]]:
        if self._stop_event.is_set() or (self._reader_thread is None and self._cap is None):
            return False, None

        timeout = self.read_timeout_sec if timeout_sec is None else max(0.01, float(timeout_sec))
        t0 = time.perf_counter_ns()
        deadline = time.monotonic() + timeout

        with self._frame_cond:
            while self._latest_seq <= self._consumed_seq:
                if self._stop_event.is_set() or self._reader_exited.is_set():
                    return False, None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.timed_out = True
                    self.last_error = (
                        f"Webcam #{self.camera_index} frame read timed out after {timeout:.2f}s."
                    )
                    logger.error(self.last_error)
                    return False, None
                self._frame_cond.wait(timeout=remaining)

            self._consumed_seq = self._latest_seq
            ret = self._latest_ret
            frame = self._latest_frame

        if ret and frame is not None and self.first_frame_ms is None:
            self.first_frame_ms = (time.perf_counter_ns() - t0) / 1_000_000
            h, w = frame.shape[:2]
            if w > 0 and h > 0:
                self._width, self._height = w, h
        return ret, frame

    def close(self) -> None:
        self._stop_event.set()
        with self._frame_cond:
            self._frame_cond.notify_all()

        if self._reader_thread is not None:
            exited = self._reader_exited.wait(timeout=1.0)
            if exited:
                self.cleanup_pending = False
                self._reader_thread = None
                logger.info("Webcam %d released.", self.camera_index)
            else:
                # Never release self._cap concurrently from another thread while owner is in native read()
                self.cleanup_pending = True
                logger.warning(
                    "Webcam %d reader still blocked in native read; capture cleanup pending.",
                    self.camera_index,
                )
        elif self._cap is not None:
            try:
                self._cap.release()
            except Exception as e:
                logger.error("Error releasing webcam: %s", e)
            self._cap = None
            logger.info("Webcam %d released.", self.camera_index)

    @property
    def source_type(self) -> str:
        return "webcam"

    @property
    def reference_string(self) -> str:
        return f"Webcam #{self.camera_index}"

    @property
    def dimensions(self) -> Tuple[int, int]:
        return self._width, self._height

    @property
    def native_fps(self) -> float:
        return self._fps


class RtspSource(BaseSource):
    """RTSP / IP Camera stream adapter with credential protection and bounded retry."""

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
        self._fps = 25.0
        self._width = 1280
        self._height = 720
        self._reconnect_count = 0

    def open(self) -> bool:
        return self._connect()

    def _connect(self) -> bool:
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        is_rtsp = self.rtsp_url.lower().startswith("rtsp://")

        if is_rtsp:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"
            self._cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        else:
            # HTTP/HTTPS IP camera stream (e.g. MJPEG from IP Webcam app)
            # Remove RTSP-specific transport option which causes HTTP timeout (-138)
            if "OPENCV_FFMPEG_CAPTURE_OPTIONS" in os.environ:
                del os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]

            # Try candidate URLs in order
            candidates = [self.rtsp_url]
            # If https is used, also try http since local phone streams often fail SSL verification
            if self.rtsp_url.startswith("https://"):
                candidates.append("http://" + self.rtsp_url[8:])
            # If endpoint missing, also try /video and /videofeed
            if not any(self.rtsp_url.rstrip("/").endswith(x) for x in ("/video", "/videofeed")):
                base = self.rtsp_url.rstrip("/")
                candidates.append(f"{base}/video")
                if base.startswith("https://"):
                    candidates.append(f"http://{base[8:]}/video")

            for target in candidates:
                self._cap = cv2.VideoCapture(target)
                if self._cap and self._cap.isOpened():
                    self.rtsp_url = target
                    break

        if not self._cap or not self._cap.isOpened():
            logger.warning("Stream connection failed: %s", self.reference_string)
            return False

        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        if w > 0 and h > 0:
            self._width = w
            self._height = h
        if fps > 0:
            self._fps = fps

        logger.info("Stream connected: %s (%dx%d)", self.reference_string, self._width, self._height)
        self._reconnect_count = 0
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._cap or not self._cap.isOpened():
            if not self._attempt_reconnection():
                return False, None

        ret, frame = self._cap.read()
        if not ret:
            logger.warning("RTSP lost frame. Attempting reconnection...")
            if self._attempt_reconnection():
                return self._cap.read()
            return False, None
        return ret, frame

    def _attempt_reconnection(self) -> bool:
        while self._reconnect_count < self.max_reconnect_attempts:
            self._reconnect_count += 1
            delay = min(self.max_delay_sec, self.initial_delay_sec * (1.5 ** (self._reconnect_count - 1)))
            logger.info("Reconnection attempt %d/%d in %.1fs...", self._reconnect_count, self.max_reconnect_attempts, delay)
            time.sleep(delay)
            if self._connect():
                return True
        logger.error("RTSP stream exceeded max reconnect attempts (%d).", self.max_reconnect_attempts)
        return False

    def close(self) -> None:
        if self._cap:
            try:
                self._cap.release()
            except Exception as e:
                logger.error("Error releasing RTSP source: %s", e)
            self._cap = None

    @property
    def source_type(self) -> str:
        return "rtsp"

    @property
    def reference_string(self) -> str:
        return sanitize_source_reference(self.rtsp_url)

    @property
    def dimensions(self) -> Tuple[int, int]:
        return self._width, self._height

    @property
    def native_fps(self) -> float:
        return self._fps


class VideoFileSource(BaseSource):
    """Prerecorded video file playback adapter.

    Preserves original video files without in-place modification or redundant copies.
    """

    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps = 30.0
        self._width = 1280
        self._height = 720
        self._total_frames = 0

    def open(self) -> bool:
        resolved, msg = resolve_video_path(self.file_path)
        if not resolved:
            logger.error(msg)
            return False
        if msg:
            logger.info(msg)
        self.file_path = resolved

        self._cap = cv2.VideoCapture(str(self.file_path))
        if not self._cap.isOpened():
            logger.error("Failed to open video file: %s", self.file_path)
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

        logger.info(
            "Video file opened: %s (%dx%d @ %.1f FPS, %d frames)",
            self.file_path.name,
            self._width,
            self._height,
            self._fps,
            self._total_frames,
        )
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._cap or not self._cap.isOpened():
            return False, None
        return self._cap.read()

    def close(self) -> None:
        if self._cap:
            try:
                self._cap.release()
            except Exception as e:
                logger.error("Error releasing video file: %s", e)
            self._cap = None

    @property
    def source_type(self) -> str:
        return "video_file"

    @property
    def reference_string(self) -> str:
        return self.file_path.name

    @property
    def dimensions(self) -> Tuple[int, int]:
        return self._width, self._height

    @property
    def native_fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._total_frames


class ImageSource(BaseSource):
    """Single image source adapter."""

    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)
        self._frame: Optional[np.ndarray] = None
        self._read_done = False

    def open(self) -> bool:
        resolved, msg = resolve_image_path(self.file_path)
        if not resolved:
            logger.error(msg)
            return False
        if msg:
            logger.info(msg)
        self.file_path = resolved

        frame = cv2.imread(str(self.file_path))
        if frame is None or frame.size == 0:
            logger.error("Failed to read image or unsupported format: %s", self.file_path)
            return False

        self._frame = frame
        self._read_done = False
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._frame is not None and not self._read_done:
            self._read_done = True
            return True, self._frame
        return False, None

    def close(self) -> None:
        self._frame = None

    @property
    def source_type(self) -> str:
        return "image"

    @property
    def reference_string(self) -> str:
        return self.file_path.name

    @property
    def dimensions(self) -> Tuple[int, int]:
        if self._frame is not None:
            return self._frame.shape[1], self._frame.shape[0]
        return 0, 0


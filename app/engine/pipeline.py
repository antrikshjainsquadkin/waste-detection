"""Master Vision Processing Pipeline.

Coordinates frame acquisition, concurrent recording of pristine source frames,
adaptive CPU-throttled inference, object tracking, event logging, and UI preview rendering.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.core.config import AppConfig
from app.core.event_manager import EventManager
from app.core.tracker import ObjectTracker
from app.core.types import Detection, SessionStatus, SourceType, TrackedObject
from app.engine.recorder import VideoRecorder
from app.engine.video_source import BaseVideoSource
from app.models.base import BaseDetector
from app.storage.session_manager import SessionManager

logger = logging.getLogger(__name__)

# Category visualization color palette (BGR)
CATEGORY_COLORS: Dict[str, Tuple[int, int, int]] = {
    "paper": (220, 180, 50),          # Cyan-Gold
    "cardboard": (40, 130, 200),       # Brown-Orange
    "plastic_bottle": (255, 140, 0),   # Deep Sky Blue
    "polythene_bag": (200, 200, 0),    # Cyan
    "metal": (160, 160, 160),          # Silver-Gray
    "glass": (255, 255, 0),            # Aqua
    "organic_waste": (50, 205, 50),     # Lime Green
    "general_trash": (120, 120, 120),  # Dark Gray
    "unknown": (180, 0, 180),          # Purple / Magenta
}


class VisionPipeline:
    """Multi-threaded pipeline orchestrating live capture, inference, and persistence."""

    def __init__(
        self,
        config: AppConfig,
        session_manager: SessionManager,
        video_source: BaseVideoSource,
        detector: BaseDetector,
        on_status_change: Optional[Callable[[str], None]] = None,
        on_stats_update: Optional[Callable[[float, int, Dict[str, int]], None]] = None,
    ):
        self.config = config
        self.session_manager = session_manager
        self.video_source = video_source
        self.detector = detector
        self.on_status_change = on_status_change
        self.on_stats_update = on_stats_update

        # Runtime state
        self._running = False
        self._paused = False
        self._stop_event = threading.Event()
        self._threads: List[threading.Thread] = []

        # Tracking & Event systems
        self.tracker = ObjectTracker(
            iou_threshold=config.iou_threshold,
            lost_track_timeout_seconds=config.lost_track_timeout_seconds,
        )
        self.event_manager: Optional[EventManager] = None
        self.recorder: Optional[VideoRecorder] = None

        # Bounded frame queues
        self._raw_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=15)
        self.preview_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=2)

        # Performance metrics
        self._frame_count = 0
        self._processed_frames = 0
        self._skipped_frames = 0
        self._current_fps = 0.0
        self._fps_timer = time.monotonic()
        self._active_tracks: List[TrackedObject] = []
        self._lock = threading.Lock()

        # Dynamic performance tuning
        self._frame_skip = config.frame_skip
        self._last_inference_latency = 0.0

    def start(self) -> bool:
        """Start monitoring session and launch pipeline worker threads."""
        if self._running:
            return True

        if not self.video_source.open():
            logger.error("Failed opening video source %s", self.video_source.reference_string)
            if self.on_status_change:
                self.on_status_change("Source Connection Failed")
            return False

        # Load detector if not already loaded
        model_info = self.detector.get_model_info()
        if not model_info.get("loaded", False):
            self.detector.load()

        model_name = model_info.get("model_name", "Unknown Model")
        model_ver = model_info.get("version", "1.0.0")

        # Start storage session
        metadata = self.session_manager.start_session(
            source_type=self.video_source.source_type.value,
            source_reference=self.video_source.reference_string,
            model_name=model_name,
            model_version=model_ver,
        )

        # Initialize EventManager with evidence saving callback
        self.event_manager = EventManager(
            config=self.config,
            session_id=metadata.session_id,
            source_type=self.video_source.source_type.value,
            source_reference=self.video_source.reference_string,
            model_name=model_name,
            model_version=model_ver,
            evidence_save_fn=self.session_manager.evidence_saver.save_evidence if self.session_manager.evidence_saver else None,
            on_event_created=self.session_manager.detection_writer.write_event if self.session_manager.detection_writer else None,
            on_event_updated=self.session_manager.detection_writer.update_event if self.session_manager.detection_writer else None,
            on_event_finalized=self.session_manager.detection_writer.update_event if self.session_manager.detection_writer else None,
        )

        # Connect Tracker callbacks to EventManager
        # We need a reference to the latest raw frame when an event is first created
        self._latest_raw_frame: Optional[np.ndarray] = None

        def _on_track_created(track: TrackedObject) -> None:
            raw = None
            with self._lock:
                if self._latest_raw_frame is not None:
                    raw = self._latest_raw_frame.copy()
            if self.event_manager:
                self.event_manager.handle_track_created(track, raw_frame=raw)

        def _on_track_updated(track: TrackedObject) -> None:
            if self.event_manager:
                self.event_manager.handle_track_updated(track)

        def _on_track_finalized(track: TrackedObject) -> None:
            if self.event_manager:
                self.event_manager.handle_track_finalized(track)

        self.tracker.on_track_created = _on_track_created
        self.tracker.on_track_updated = _on_track_updated
        self.tracker.on_track_finalized = _on_track_finalized

        # Initialize Recorder if live webcam/RTSP monitoring
        if self.video_source.source_type in (SourceType.WEBCAM, SourceType.RTSP):
            rec_target = self.session_manager.current_session_dir / "recording.mp4"
            w, h = self.video_source.dimensions
            fps = self.video_source.native_fps
            self.recorder = VideoRecorder(output_path=rec_target, frame_size=(w, h), fps=fps)
            self.recorder.start()

        self._running = True
        self._stop_event.clear()

        # Launch capture reader thread
        reader_t = threading.Thread(target=self._reader_worker, name="CaptureReaderWorker", daemon=True)
        # Launch inference worker thread
        infer_t = threading.Thread(target=self._inference_worker, name="InferenceWorker", daemon=True)

        self._threads = [reader_t, infer_t]
        for t in self._threads:
            t.start()

        logger.info("Vision Pipeline started successfully.")
        if self.on_status_change:
            self.on_status_change("Monitoring Active")
        return True

    def _reader_worker(self) -> None:
        """Worker thread continuously capturing frames from video source."""
        while self._running and not self._stop_event.is_set():
            ret, frame = self.video_source.read_frame()
            if not ret or frame is None:
                # For video files, EOF signals normal completion
                if self.video_source.source_type == SourceType.VIDEO_FILE:
                    logger.info("Video file reached EOF.")
                    break
                else:
                    logger.warning("Frame capture failed or source disconnected.")
                    time.sleep(0.1)
                    continue

            self._frame_count += 1
            with self._lock:
                self._latest_raw_frame = frame

            # Push pristine frame to video recorder (no bounding boxes!)
            if self.recorder:
                self.recorder.write_frame(frame)

            # Push to inference queue (drop old frame if backed up to maintain real-time)
            try:
                self._raw_queue.put(frame, timeout=0.05)
            except queue.Full:
                try:
                    _ = self._raw_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._raw_queue.put_nowait(frame)
                except queue.Full:
                    pass

        logger.info("Capture reader loop finished.")
        # Signal stop if reader ended (e.g. video file EOF)
        if self._running and self.video_source.source_type == SourceType.VIDEO_FILE:
            self._running = False
            if self.on_status_change:
                self.on_status_change("Processing Completed")

    def _inference_worker(self) -> None:
        """Worker thread running CPU-optimized inference, tracking, and UI preview rendering."""
        conf_thresholds = self.config.get_all_confidence_thresholds()
        unknown_min = self.config.unknown_candidate_min_confidence
        last_tracks: List[TrackedObject] = []
        fps_frame_count = 0
        fps_start = time.monotonic()

        while self._running and not self._stop_event.is_set():
            try:
                raw_frame = self._raw_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if raw_frame is None:
                break

            now = time.monotonic()
            fps_frame_count += 1
            if now - fps_start >= 1.0:
                self._current_fps = fps_frame_count / (now - fps_start)
                fps_frame_count = 0
                fps_start = now

            # Determine whether to infer on this frame or reuse previous tracks
            should_infer = (self._frame_count % self._frame_skip == 0)

            if should_infer:
                t0 = time.monotonic()
                try:
                    detections = self.detector.predict(
                        raw_frame,
                        confidence_thresholds=conf_thresholds,
                        unknown_min_conf=unknown_min,
                    )
                except Exception as e:
                    logger.error("Inference exception: %s", e)
                    detections = []
                self._last_inference_latency = time.monotonic() - t0

                # Adaptive performance tuning
                if self.config.adaptive_performance:
                    # If inference takes > 250ms (less than 4 FPS), throttle frame skip
                    if self._last_inference_latency > 0.25 and self._frame_skip < 5:
                        self._frame_skip += 1
                    # If latency is under 80ms, can afford to infer more frequently
                    elif self._last_inference_latency < 0.08 and self._frame_skip > 1:
                        self._frame_skip -= 1

                # Update spatial-temporal object tracker
                active_tracks = self.tracker.update(detections, current_time=now)
                last_tracks = active_tracks
                self._processed_frames += 1
            else:
                # Non-inference frame: maintain tracks
                active_tracks = last_tracks
                self._skipped_frames += 1

            with self._lock:
                self._active_tracks = active_tracks

            # Notify UI stats
            if self.on_stats_update and self.event_manager:
                counts = self.event_manager.get_category_counts()
                self.on_stats_update(self._current_fps, self.event_manager.total_events, counts)

            # Render preview frame with annotations for the desktop UI
            annotated_frame = self._render_annotations(raw_frame.copy(), active_tracks)

            # Push to UI preview queue
            try:
                self.preview_queue.put_nowait(annotated_frame)
            except queue.Full:
                try:
                    _ = self.preview_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.preview_queue.put_nowait(annotated_frame)
                except queue.Full:
                    pass

        logger.info("Inference worker loop finished.")

    def _render_annotations(self, frame: np.ndarray, tracks: List[TrackedObject]) -> np.ndarray:
        """Draw bounding boxes, class labels, tracking IDs, and status HUD onto preview frame."""
        h, w = frame.shape[:2]

        for track in tracks:
            bx, by, bw, bh = track.bbox.to_xywh()
            x1 = max(0, min(w - 1, bx))
            y1 = max(0, min(h - 1, by))
            x2 = max(0, min(w - 1, bx + bw))
            y2 = max(0, min(h - 1, by + bh))

            color = CATEGORY_COLORS.get(track.class_name, (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"#{track.track_id} {track.display_name}: {track.confidence:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Draw background box for text
            text_y1 = max(0, y1 - text_h - 6)
            text_y2 = y1
            cv2.rectangle(frame, (x1, text_y1), (x1 + text_w + 4, text_y2), color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Draw HUD bar at top-left
        hud_text = f"FPS: {self._current_fps:.1f} | Latency: {self._last_inference_latency * 1000:.0f}ms | Skip: {self._frame_skip}"
        cv2.putText(frame, hud_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, hud_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

        return frame

    def stop(self) -> None:
        """Stop monitoring pipeline, finalize recording, and write session summary."""
        if not self._running:
            return

        logger.info("Stopping Vision Pipeline...")
        self._running = False
        self._stop_event.set()

        # Finalize tracker remaining tracks
        self.tracker.reset()

        # Close video source
        self.video_source.close()

        # Stop recorder if running
        rec_path = ""
        if self.recorder:
            self.recorder.stop()
            rec_path = self.recorder.final_file_path
            self.recorder = None

        # Join threads
        for t in self._threads:
            if t.is_alive():
                t.join(timeout=2.0)
        self._threads.clear()

        # Finalize storage session
        if self.event_manager and self.session_manager.current_metadata:
            total_events = self.event_manager.total_events
            counts = self.event_manager.get_category_counts()
            unknown_cnt = self.event_manager.get_unknown_count()
            duration = time.monotonic() - self._fps_timer

            self.session_manager.finalize_session(
                total_events=total_events,
                category_counts=counts,
                unknown_count=unknown_cnt,
                duration_seconds=duration,
                average_fps=self._current_fps,
                recording_path=rec_path,
                status=SessionStatus.STOPPED,
            )

        logger.info("Vision Pipeline stopped and session finalized.")
        if self.on_status_change:
            self.on_status_change("Monitoring Stopped")

    @property
    def is_running(self) -> bool:
        return self._running


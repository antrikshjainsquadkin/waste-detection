"""Single-file CSV detection storage engine for Waste Detection System.

Implements safe, thread-safe append-only persistence of unique detection events
to a single project-root CSV file (`waste_detections.csv`) with zero auxiliary
folders, zero JSON files, zero log files, and zero evidence image files.
"""

from __future__ import annotations

import csv
import datetime
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from app.config import sanitize_source_reference

logger = logging.getLogger(__name__)

CSV_HEADERS: List[str] = [
    "record_id",
    "session_id",
    "event_id",
    "track_id",
    "detected_at",
    "source_type",
    "source_name",
    "frame_number",
    "class_id",
    "class_name",
    "confidence",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "model_name",
    "checkpoint_filename",
]


@dataclass(slots=True)
class DetectionEvent:
    """Representation of an object detection event tracked across frames."""
    record_id: str
    session_id: str
    event_id: str
    track_id: Optional[int]
    detected_at: str
    source_type: str
    source_name: str
    frame_number: int
    class_id: int
    class_name: str
    confidence: float
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    model_name: str
    checkpoint_filename: str

    def to_csv_dict(self) -> Dict[str, str]:
        """Convert event into formatted dictionary matching CSV_HEADERS exactly."""
        return {
            "record_id": self.record_id,
            "session_id": self.session_id,
            "event_id": self.event_id,
            "track_id": str(self.track_id) if self.track_id is not None else "",
            "detected_at": self.detected_at,
            "source_type": self.source_type,
            "source_name": self.source_name,
            "frame_number": str(self.frame_number),
            "class_id": str(self.class_id),
            "class_name": self.class_name,
            "confidence": f"{self.confidence:.4f}",
            "bbox_x1": str(self.bbox_x1),
            "bbox_y1": str(self.bbox_y1),
            "bbox_x2": str(self.bbox_x2),
            "bbox_y2": str(self.bbox_y2),
            "model_name": self.model_name,
            "checkpoint_filename": self.checkpoint_filename,
        }


class StorageManager:
    """Manages appending unique detection events to a single project CSV file."""

    def __init__(
        self,
        csv_path: Optional[Path] = None,
        session_id: str = "",
        source_type: str = "",
        source_reference: str = "",
        model_name: str = "khanhromvn/waste-detection-yolo11",
        checkpoint_filename: str = "waste_detection_yolo11m.pt",
        confidence_threshold: float = 0.25,
        image_size: int = 640,
        model_classes: Optional[Dict[int, str]] = None,
        model_sha256: str = "",
        # Deprecated compatibility kwargs:
        base_dir: Optional[Any] = None,
        save_to_disk: bool = True,
    ) -> None:
        workspace_root = Path(__file__).resolve().parent.parent
        self.csv_path = Path(csv_path) if csv_path else (workspace_root / "waste_detections.csv")
        self.session_id = session_id or f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.source_type = source_type
        self.source_reference = sanitize_source_reference(source_reference)
        self.model_name = model_name
        self.checkpoint_filename = checkpoint_filename
        self.confidence_threshold = float(confidence_threshold)
        self.image_size = int(image_size)
        self.model_classes = model_classes or {}
        self.model_sha256 = model_sha256

        self.start_time = datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()
        self.end_time = ""
        self.duration_seconds = 0.0
        self.average_fps = 0.0
        self.warnings: List[str] = []

        self._lock = threading.Lock()
        self._events: Dict[str, DetectionEvent] = {}
        self._written_event_ids: Set[str] = set()
        self._pending_events: Dict[str, DetectionEvent] = {}
        self._uncertain_event_ids: Set[str] = set()
        self.last_write_error: str = ""
        self._last_retry_monotonic: float = 0.0
        self._retry_interval_sec: float = 1.0

        self.last_write_duration_ms: float = 0.0
        self.total_write_duration_ms: float = 0.0
        self.write_count: int = 0

        self._init_csv()

    def _init_csv(self) -> None:
        """Create CSV with headers if missing, or load existing event IDs to preserve history."""
        with self._lock:
            if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
                try:
                    with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                        writer.writeheader()
                    logger.info("Initialized detection CSV at: %s", self.csv_path)
                except Exception as e:
                    logger.error("Failed to initialize CSV at %s: %s", self.csv_path, e)
                    self.warnings.append(f"CSV init error: {e}")
            else:
                # Load existing event IDs to prevent duplicate row writes across sessions
                try:
                    with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            eid = row.get("event_id")
                            if eid:
                                self._written_event_ids.add(eid)
                    logger.info("Loaded %d existing events from: %s", len(self._written_event_ids), self.csv_path)
                except Exception as e:
                    logger.warning("Could not read existing CSV header: %s", e)

    def save_evidence_image(self, frame: Any, event_id: str, class_name: str) -> str:
        """Disabled in CSV-only mode: no evidence images are saved to disk."""
        return ""

    def _csv_contains_event_id_locked(self, event_id: str) -> bool:
        """Check whether an event_id is already present in the CSV file."""
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            return False
        try:
            with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("event_id") == event_id:
                        return True
        except Exception:
            return False
        return False

    def _write_event_to_csv_locked(self, event: DetectionEvent) -> bool:
        """Attempt to persist a single event to CSV while holding self._lock."""
        if event.event_id in self._written_event_ids:
            return True

        # Reconcile uncertain outcome before appending to avoid duplicate rows
        if event.event_id in self._uncertain_event_ids:
            if self._csv_contains_event_id_locked(event.event_id):
                self._events[event.event_id] = event
                self._written_event_ids.add(event.event_id)
                self._pending_events.pop(event.event_id, None)
                self._uncertain_event_ids.discard(event.event_id)
                self.last_write_error = ""
                return True

        t0 = time.perf_counter_ns()
        write_started = False
        try:
            needs_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                if needs_header:
                    writer.writeheader()
                write_started = True
                writer.writerow(event.to_csv_dict())
                f.flush()
            write_ms = (time.perf_counter_ns() - t0) / 1_000_000
            self._events[event.event_id] = event
            self._written_event_ids.add(event.event_id)
            self._pending_events.pop(event.event_id, None)
            self._uncertain_event_ids.discard(event.event_id)
            self.last_write_duration_ms = write_ms
            self.total_write_duration_ms += write_ms
            self.write_count += 1
            self.last_write_error = ""
            return True
        except (PermissionError, OSError, Exception) as e:
            if write_started:
                self._uncertain_event_ids.add(event.event_id)
            self._pending_events[event.event_id] = event
            self.last_write_duration_ms = 0.0
            self.last_write_error = str(e)
            warn_msg = f"CSV write error: {e}"
            if warn_msg not in self.warnings:
                self.warnings.append(warn_msg)
            logger.error("Failed writing event %s to CSV: %s", event.event_id, e)
            return False

    def record_event(self, event: DetectionEvent) -> bool:
        """Append a unique detection event to the single CSV file.

        Returns True only after the CSV write and file close succeed.
        """
        with self._lock:
            return self._write_event_to_csv_locked(event)

    def retry_pending_events(self, force: bool = False) -> int:
        """Retry pending unsaved events in a bounded, duplicate-safe manner."""
        with self._lock:
            if not self._pending_events:
                return 0
            now = time.monotonic()
            if not force and (now - self._last_retry_monotonic) < self._retry_interval_sec:
                return 0
            self._last_retry_monotonic = now

            saved = 0
            for pending_ev in list(self._pending_events.values()):
                if self._write_event_to_csv_locked(pending_ev):
                    saved += 1
                else:
                    # Stop batch immediately while file remains locked/unwritable
                    break
            return saved

    def update_event(self, event: DetectionEvent) -> None:
        """Update existing event in memory without creating duplicate CSV rows."""
        with self._lock:
            if event.event_id in self._written_event_ids:
                self._events[event.event_id] = event
            elif event.event_id in self._pending_events:
                self._pending_events[event.event_id] = event

    def get_category_counts(self) -> Dict[str, int]:
        """Return event counts by class for current session."""
        counts = {"organic": 0, "recyclable": 0, "residual": 0}
        with self._lock:
            for ev in self._events.values():
                cat = ev.class_name.lower()
                counts[cat] = counts.get(cat, 0) + 1
        return counts

    @property
    def total_events(self) -> int:
        with self._lock:
            return len(self._events)

    @property
    def pending_events_count(self) -> int:
        with self._lock:
            return len(self._pending_events)

    @property
    def detected_events_count(self) -> int:
        with self._lock:
            return len(self._events) + len(self._pending_events)

    def finalize_session(
        self,
        duration_seconds: float,
        average_fps: float,
        recording_path: str = "",
    ) -> Dict[str, Any]:
        """Finalize session summary in memory without writing any JSON or report files."""
        self.retry_pending_events(force=True)
        self.end_time = datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()
        self.duration_seconds = max(0.0, float(duration_seconds))
        self.average_fps = max(0.0, float(average_fps))

        category_counts = self.get_category_counts()

        return {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": round(self.duration_seconds, 2),
            "source_type": self.source_type,
            "source_reference": self.source_reference,
            "model_name": self.model_name,
            "checkpoint_filename": self.checkpoint_filename,
            "confidence_threshold": self.confidence_threshold,
            "image_size": self.image_size,
            "total_events": self.total_events,
            "detected_events": self.detected_events_count,
            "pending_events": self.pending_events_count,
            "last_write_error": self.last_write_error,
            "category_counts": category_counts,
            "csv_path": str(self.csv_path),
            "average_fps": round(self.average_fps, 2),
            "runtime_errors_or_warnings": self.warnings,
        }

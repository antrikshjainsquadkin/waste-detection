"""Thread-safe CSV logging subsystem for detection events and session summaries.

Enforces documented schemas, regular atomic flushing, credential redaction,
and crash-resilient persistence.
"""

from __future__ import annotations

import csv
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List

from app.core.config import sanitize_source_reference
from app.core.types import DetectionEvent, SessionMetadata

logger = logging.getLogger(__name__)

DETECTION_CSV_HEADERS: List[str] = [
    "session_id",
    "event_id",
    "track_id",
    "category",
    "confidence",
    "first_seen",
    "last_seen",
    "duration_seconds",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
    "frame_width",
    "frame_height",
    "source_type",
    "source_reference",
    "model_name",
    "model_version",
    "evidence_image_path",
]

SUMMARY_CSV_BASE_HEADERS: List[str] = [
    "session_id",
    "start_time",
    "end_time",
    "duration_seconds",
    "source_type",
    "source_reference",
    "total_events",
    "unknown_count",
    "recording_path",
    "model_name",
    "model_version",
    "average_fps",
    "status",
]


class DetectionCsvWriter:
    """Manages appending and updating detection event records in detections.csv."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._lock = threading.Lock()
        self._written_events: Dict[str, Dict[str, str]] = {}
        self._init_file()

    def _init_file(self) -> None:
        """Create file and write header if not present."""
        with self._lock:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.file_path.exists() or self.file_path.stat().st_size == 0:
                with open(self.file_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=DETECTION_CSV_HEADERS)
                    writer.writeheader()

    def write_event(self, event: DetectionEvent) -> None:
        """Append a newly created detection event."""
        row = event.to_csv_dict()
        row["source_reference"] = sanitize_source_reference(row["source_reference"])

        with self._lock:
            self._written_events[event.event_id] = row
            try:
                with open(self.file_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=DETECTION_CSV_HEADERS)
                    writer.writerow(row)
                    f.flush()
            except Exception as e:
                logger.error("Failed writing event %s to CSV %s: %s", event.event_id, self.file_path, e)

    def update_event(self, event: DetectionEvent) -> None:
        """Update event details (e.g. last_seen, duration) in memory and rewrite file on major updates."""
        row = event.to_csv_dict()
        row["source_reference"] = sanitize_source_reference(row["source_reference"])

        with self._lock:
            self._written_events[event.event_id] = row

    def flush_all(self) -> None:
        """Synchronize all updated events back to disk atomically."""
        with self._lock:
            if not self._written_events:
                return
            temp_path = self.file_path.with_suffix(".tmp")
            try:
                with open(temp_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=DETECTION_CSV_HEADERS)
                    writer.writeheader()
                    for row in self._written_events.values():
                        writer.writerow(row)
                    f.flush()
                temp_path.replace(self.file_path)
            except Exception as e:
                logger.error("Failed flushing detection CSV %s: %s", self.file_path, e)
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError:
                        pass


class SummaryCsvWriter:
    """Writes and updates session-level aggregate statistics in summary.csv."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._lock = threading.Lock()

    def write_summary(self, metadata: SessionMetadata) -> None:
        """Write session summary CSV row."""
        row = metadata.to_summary_csv_dict()
        row["source_reference"] = sanitize_source_reference(row["source_reference"])

        # Determine all fieldnames dynamically to include count_<category>
        fieldnames = list(SUMMARY_CSV_BASE_HEADERS)
        for k in sorted(row.keys()):
            if k not in fieldnames:
                fieldnames.append(k)

        with self._lock:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.file_path.with_suffix(".tmp")
            try:
                with open(temp_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerow(row)
                    f.flush()
                temp_path.replace(self.file_path)
            except Exception as e:
                logger.error("Failed writing summary CSV %s: %s", self.file_path, e)
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError:
                        pass


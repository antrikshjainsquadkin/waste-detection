"""Session persistence and lifecycle management subsystem.

Coordinates per-session storage isolation, metadata persistence in JSON,
CSV reporting, and session recovery.
"""

from __future__ import annotations

import datetime
import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import AppConfig, sanitize_source_reference
from app.core.types import SessionMetadata, SessionStatus
from app.storage.csv_writer import DetectionCsvWriter, SummaryCsvWriter
from app.storage.evidence_saver import EvidenceSaver

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages individual monitoring sessions and directory isolation."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.recordings_root = config.recordings_dir
        self.recordings_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        # Active session references
        self.current_metadata: Optional[SessionMetadata] = None
        self.current_session_dir: Optional[Path] = None
        self.detection_writer: Optional[DetectionCsvWriter] = None
        self.summary_writer: Optional[SummaryCsvWriter] = None
        self.evidence_saver: Optional[EvidenceSaver] = None

    def start_session(
        self,
        source_type: str,
        source_reference: str,
        model_name: str,
        model_version: str,
    ) -> SessionMetadata:
        """Initialize a new isolated monitoring session."""
        with self._lock:
            now = datetime.datetime.now(datetime.timezone.utc).astimezone()
            timestamp_str = now.strftime("%Y%m%d_%H%M%S")
            session_id = f"session_{timestamp_str}"
            session_dir = self.recordings_root / session_id
            session_dir.mkdir(parents=True, exist_ok=True)

            self.current_session_dir = session_dir
            sanitized_ref = sanitize_source_reference(source_reference)

            # Initialize initial metadata
            self.current_metadata = SessionMetadata(
                session_id=session_id,
                start_time=now.isoformat(),
                source_type=source_type,
                source_reference=sanitized_ref,
                model_name=model_name,
                model_version=model_version,
                status=SessionStatus.ACTIVE.value,
                category_counts={c["internal_name"]: 0 for c in self.config.categories},
            )

            # Initialize writers and evidence saver
            detections_csv_path = session_dir / "detections.csv"
            summary_csv_path = session_dir / "summary.csv"
            evidence_dir = session_dir / "evidence"

            self.detection_writer = DetectionCsvWriter(detections_csv_path)
            self.summary_writer = SummaryCsvWriter(summary_csv_path)
            self.evidence_saver = EvidenceSaver(
                evidence_dir=evidence_dir,
                image_format=self.config.evidence_format,
                quality=self.config.evidence_quality,
            )

            # Write initial session metadata JSON
            self._save_metadata_json()

            logger.info("Session started: %s (Source: %s)", session_id, sanitized_ref)
            return self.current_metadata

    def finalize_session(
        self,
        total_events: int,
        category_counts: Dict[str, int],
        unknown_count: int,
        duration_seconds: float,
        average_fps: float,
        recording_path: str = "",
        status: SessionStatus = SessionStatus.STOPPED,
    ) -> Optional[SessionMetadata]:
        """Finalize and seal the current session."""
        with self._lock:
            if not self.current_metadata or not self.current_session_dir:
                logger.warning("No active session to finalize.")
                return None

            now_iso = datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()
            self.current_metadata.end_time = now_iso
            self.current_metadata.duration_seconds = max(0.0, float(duration_seconds))
            self.current_metadata.total_events = total_events
            self.current_metadata.category_counts = category_counts
            self.current_metadata.unknown_count = unknown_count
            self.current_metadata.average_fps = average_fps
            self.current_metadata.recording_path = recording_path
            self.current_metadata.status = status.value

            # Flush CSVs
            if self.detection_writer:
                self.detection_writer.flush_all()
            if self.summary_writer:
                self.summary_writer.write_summary(self.current_metadata)

            # Save final JSON metadata
            self._save_metadata_json()

            meta = self.current_metadata
            logger.info("Session %s finalized. Status: %s, Events: %d", meta.session_id, meta.status, meta.total_events)

            # Reset active pointers
            self.current_metadata = None
            self.current_session_dir = None
            self.detection_writer = None
            self.summary_writer = None
            self.evidence_saver = None

            return meta

    def record_interruption(self, reason: str) -> None:
        """Log a connection interruption to session metadata."""
        with self._lock:
            if self.current_metadata:
                timestamp = datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()
                self.current_metadata.interruptions.append(f"{timestamp}: {reason}")
                self._save_metadata_json()

    def _save_metadata_json(self) -> None:
        """Write session_metadata.json to current session directory."""
        if not self.current_metadata or not self.current_session_dir:
            return

        json_path = self.current_session_dir / "session_metadata.json"
        temp_path = json_path.with_suffix(".tmp")
        try:
            data = {
                "session_id": self.current_metadata.session_id,
                "start_time": self.current_metadata.start_time,
                "end_time": self.current_metadata.end_time,
                "duration_seconds": self.current_metadata.duration_seconds,
                "source_type": self.current_metadata.source_type,
                "source_reference": self.current_metadata.source_reference,
                "total_events": self.current_metadata.total_events,
                "category_counts": self.current_metadata.category_counts,
                "unknown_count": self.current_metadata.unknown_count,
                "recording_path": self.current_metadata.recording_path,
                "model_name": self.current_metadata.model_name,
                "model_version": self.current_metadata.model_version,
                "average_fps": self.current_metadata.average_fps,
                "status": self.current_metadata.status,
                "interruptions": self.current_metadata.interruptions,
            }
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            temp_path.replace(json_path)
        except Exception as e:
            logger.error("Failed saving session_metadata.json: %s", e)
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all discovered session directories and summary metadata."""
        sessions: List[Dict[str, Any]] = []
        if not self.recordings_root.exists():
            return sessions

        for s_dir in sorted(self.recordings_root.glob("session_*"), reverse=True):
            if not s_dir.is_dir():
                continue

            meta_file = s_dir / "session_metadata.json"
            if meta_file.exists():
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        data["dir_path"] = str(s_dir)
                        data["has_recording"] = bool(data.get("recording_path") and Path(data["recording_path"]).exists())
                        sessions.append(data)
                        continue
                except Exception:
                    pass

            # Fallback if metadata json missing/corrupt
            sessions.append({
                "session_id": s_dir.name,
                "start_time": "Unknown",
                "end_time": "Unknown",
                "duration_seconds": 0.0,
                "source_type": "Unknown",
                "source_reference": "",
                "total_events": 0,
                "category_counts": {},
                "unknown_count": 0,
                "recording_path": "",
                "status": "Incomplete",
                "dir_path": str(s_dir),
                "has_recording": (s_dir / "recording.mp4").exists(),
            })

        return sessions


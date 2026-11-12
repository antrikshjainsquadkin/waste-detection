"""Detection Event Lifecycle Manager.

Translates temporal object tracks into discrete, auditable detection events,
coordinates full-frame evidence image capture, and maintains session statistics.
"""

from __future__ import annotations

import datetime
import logging
import uuid
from typing import Callable, Dict, List, Optional

import numpy as np

from app.core.config import AppConfig
from app.core.types import DetectionEvent, TrackedObject

logger = logging.getLogger(__name__)


def generate_event_id() -> str:
    """Generate a unique, sorted event identifier."""
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:6]
    return f"evt_{now_str}_{short_uuid}"


def get_current_iso_timestamp() -> str:
    """Return ISO 8601 formatted local timestamp with offset."""
    return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()


class EventManager:
    """Coordinates detection events, evidence preservation, and statistical metrics."""

    def __init__(
        self,
        config: AppConfig,
        session_id: str,
        source_type: str,
        source_reference: str,
        model_name: str,
        model_version: str,
        evidence_save_fn: Optional[Callable[[np.ndarray, str, str], str]] = None,
        on_event_created: Optional[Callable[[DetectionEvent], None]] = None,
        on_event_updated: Optional[Callable[[DetectionEvent], None]] = None,
        on_event_finalized: Optional[Callable[[DetectionEvent], None]] = None,
    ):
        self.config = config
        self.session_id = session_id
        self.source_type = source_type
        self.source_reference = source_reference
        self.model_name = model_name
        self.model_version = model_version
        self.evidence_save_fn = evidence_save_fn
        self.on_event_created = on_event_created
        self.on_event_updated = on_event_updated
        self.on_event_finalized = on_event_finalized

        self._events: Dict[str, DetectionEvent] = {}  # event_id -> DetectionEvent
        self._track_to_event: Dict[int, str] = {}     # track_id -> event_id

    @property
    def total_events(self) -> int:
        return len(self._events)

    @property
    def events(self) -> List[DetectionEvent]:
        return list(self._events.values())

    def get_category_counts(self) -> Dict[str, int]:
        """Aggregate event counts per category for the session."""
        counts: Dict[str, int] = {c["internal_name"]: 0 for c in self.config.categories}
        for event in self._events.values():
            cat = event.category
            if not event.is_unknown:
                counts[cat] = counts.get(cat, 0) + 1
        return counts

    def get_unknown_count(self) -> int:
        """Count total Unknown detection events."""
        return sum(1 for e in self._events.values() if e.is_unknown)

    def handle_track_created(self, track: TrackedObject, raw_frame: Optional[np.ndarray] = None) -> DetectionEvent:
        """Create a new detection event for a new track."""
        event_id = generate_event_id()
        track.event_id = event_id
        track.event_created = True
        self._track_to_event[track.track_id] = event_id

        iso_now = get_current_iso_timestamp()
        frame_h = raw_frame.shape[0] if raw_frame is not None else 0
        frame_w = raw_frame.shape[1] if raw_frame is not None else 0

        # Save single unannotated full-frame evidence image
        evidence_path = ""
        if raw_frame is not None and self.evidence_save_fn is not None:
            try:
                evidence_path = self.evidence_save_fn(raw_frame, event_id, track.class_name)
            except Exception as e:
                logger.error("Failed to save evidence image for event %s: %s", event_id, e)

        event = DetectionEvent(
            event_id=event_id,
            session_id=self.session_id,
            track_id=track.track_id,
            category=track.class_name,
            display_name=track.display_name,
            confidence=track.confidence,
            first_seen=iso_now,
            last_seen=iso_now,
            duration_seconds=0.0,
            bbox=track.bbox,
            frame_width=frame_w,
            frame_height=frame_h,
            source_type=self.source_type,
            source_reference=self.source_reference,
            model_name=self.model_name,
            model_version=self.model_version,
            evidence_image_path=evidence_path,
            is_unknown=track.is_unknown,
        )

        self._events[event_id] = event
        logger.info(
            "Detection Event Created: %s | Track %d | %s (%.2f)",
            event_id, track.track_id, track.display_name, track.confidence
        )

        if self.on_event_created:
            try:
                self.on_event_created(event)
            except Exception as e:
                logger.error("Error in on_event_created callback: %s", e)

        return event

    def handle_track_updated(self, track: TrackedObject) -> Optional[DetectionEvent]:
        """Update existing detection event timestamps and duration as track continues."""
        event_id = self._track_to_event.get(track.track_id)
        if not event_id or event_id not in self._events:
            return None

        event = self._events[event_id]
        event.last_seen = get_current_iso_timestamp()
        event.duration_seconds = max(0.0, track.last_seen - track.first_seen)
        event.confidence = max(event.confidence, track.confidence)
        event.bbox = track.bbox

        if track.is_unknown != event.is_unknown:
            event.is_unknown = track.is_unknown
            event.category = track.class_name
            event.display_name = track.display_name

        if self.on_event_updated:
            try:
                self.on_event_updated(event)
            except Exception as e:
                logger.error("Error in on_event_updated callback: %s", e)

        return event

    def handle_track_finalized(self, track: TrackedObject) -> Optional[DetectionEvent]:
        """Finalize event when track disappears beyond timeout."""
        event_id = self._track_to_event.pop(track.track_id, None)
        if not event_id or event_id not in self._events:
            return None

        event = self._events[event_id]
        event.last_seen = get_current_iso_timestamp()
        event.duration_seconds = max(0.0, track.last_seen - track.first_seen)

        logger.info(
            "Detection Event Finalized: %s | Track %d | %s | Duration: %.2fs",
            event_id, track.track_id, event.display_name, event.duration_seconds
        )

        if self.on_event_finalized:
            try:
                self.on_event_finalized(event)
            except Exception as e:
                logger.error("Error in on_event_finalized callback: %s", e)

        return event


"""Object tracking and continuous-event deduplication engine.

Preserves track identity across frames, prevents duplicate CSV records
for the same continuous object, and manages evidence image saving.
"""

from __future__ import annotations

import datetime
import logging
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from app.detector import Detection
from app.storage import DetectionEvent, StorageManager

logger = logging.getLogger(__name__)


def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union (IoU) of two (x, y, w, h) boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = max(0, w1) * max(0, h1)
    area2 = max(0, w2) * max(0, h2)
    union = area1 + area2 - intersection
    if union <= 0:
        return 0.0
    return float(intersection) / float(union)


@dataclass
class TrackState:
    """Internal state for an actively tracked object."""
    track_id: int
    event_id: str
    class_id: int
    class_name: str
    confidence: float
    peak_confidence: float
    first_seen_monotonic: float
    last_seen_monotonic: float
    first_seen_iso: str
    last_seen_iso: str
    frame_index: int
    bbox: Tuple[int, int, int, int]
    evidence_image_path: str = ""
    record_id: str = ""


class DetectionTracker:
    """Tracks detected objects across frames and prevents duplicate CSV records."""

    def __init__(
        self,
        storage_manager: StorageManager,
        iou_threshold: float = 0.35,
        lost_track_timeout: float = 3.0,
        on_new_event: Optional[Callable[[DetectionEvent], None]] = None,
    ):
        self.storage_manager = storage_manager
        self.iou_threshold = float(iou_threshold)
        self.lost_track_timeout = float(lost_track_timeout)
        self.on_new_event = on_new_event

        self._active_tracks: Dict[int, TrackState] = {}
        self._next_fallback_id = 1000

    def reset_session(self) -> None:
        """Clear per-session in-memory track state without touching persisted CSV history."""
        self._active_tracks.clear()
        self._next_fallback_id = 1000

    def process_frame(
        self,
        detections: List[Detection],
        raw_frame: np.ndarray,
        frame_index: int,
    ) -> List[TrackState]:
        """Update tracks with new detections from the current frame."""
        now_mono = time.monotonic()
        now_iso = datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()

        # Step 1: Assign IDs to detections that have native track_id from Ultralytics
        matched_track_ids = set()
        unmatched_detections: List[Detection] = []

        for det in detections:
            if det.track_id is not None:
                tid = det.track_id
                if tid in self._active_tracks:
                    # Update existing track
                    track = self._active_tracks[tid]
                    track.last_seen_monotonic = now_mono
                    track.last_seen_iso = now_iso
                    track.peak_confidence = max(track.peak_confidence, det.confidence)
                    track.confidence = det.confidence
                    track.bbox = det.bbox

                    # Update in storage (in-memory only, no duplicate CSV row)
                    event = DetectionEvent(
                        record_id=track.record_id,
                        session_id=self.storage_manager.session_id,
                        event_id=track.event_id,
                        track_id=track.track_id,
                        detected_at=track.first_seen_iso,
                        source_type=self.storage_manager.source_type,
                        source_name=self.storage_manager.source_reference,
                        frame_number=track.frame_index,
                        class_id=track.class_id,
                        class_name=track.class_name,
                        confidence=track.confidence,
                        bbox_x1=track.bbox[0],
                        bbox_y1=track.bbox[1],
                        bbox_x2=track.bbox[0] + track.bbox[2],
                        bbox_y2=track.bbox[1] + track.bbox[3],
                        model_name=self.storage_manager.model_name,
                        checkpoint_filename=self.storage_manager.checkpoint_filename,
                    )
                    self.storage_manager.update_event(event)
                    matched_track_ids.add(tid)
                else:
                    # New track with native track_id
                    self._create_new_track(det, tid, raw_frame, frame_index, now_mono, now_iso)
                    matched_track_ids.add(tid)
            else:
                unmatched_detections.append(det)

        # Step 2: Fallback spatial IoU matching for detections without native track_id
        if unmatched_detections and self._active_tracks:
            unmatched_tracks = [t for tid, t in self._active_tracks.items() if tid not in matched_track_ids]
            for det in unmatched_detections[:]:
                best_iou = 0.0
                best_track: Optional[TrackState] = None
                for trk in unmatched_tracks:
                    # Prefer matching same class
                    if trk.class_id == det.class_id:
                        iou = compute_iou(trk.bbox, det.bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_track = trk

                if best_iou >= self.iou_threshold and best_track is not None:
                    # Matched existing track
                    best_track.last_seen_monotonic = now_mono
                    best_track.last_seen_iso = now_iso
                    best_track.peak_confidence = max(best_track.peak_confidence, det.confidence)
                    best_track.confidence = det.confidence
                    best_track.bbox = det.bbox
                    det.track_id = best_track.track_id

                    # Update in storage (in-memory only, no duplicate CSV row)
                    event = DetectionEvent(
                        record_id=best_track.record_id,
                        session_id=self.storage_manager.session_id,
                        event_id=best_track.event_id,
                        track_id=best_track.track_id,
                        detected_at=best_track.first_seen_iso,
                        source_type=self.storage_manager.source_type,
                        source_name=self.storage_manager.source_reference,
                        frame_number=best_track.frame_index,
                        class_id=best_track.class_id,
                        class_name=best_track.class_name,
                        confidence=best_track.confidence,
                        bbox_x1=best_track.bbox[0],
                        bbox_y1=best_track.bbox[1],
                        bbox_x2=best_track.bbox[0] + best_track.bbox[2],
                        bbox_y2=best_track.bbox[1] + best_track.bbox[3],
                        model_name=self.storage_manager.model_name,
                        checkpoint_filename=self.storage_manager.checkpoint_filename,
                    )
                    self.storage_manager.update_event(event)
                    matched_track_ids.add(best_track.track_id)
                    unmatched_tracks.remove(best_track)
                    unmatched_detections.remove(det)

        # Step 3: Remaining unmatched detections become new tracks
        for det in unmatched_detections:
            tid = self._next_fallback_id
            self._next_fallback_id += 1
            self._create_new_track(det, tid, raw_frame, frame_index, now_mono, now_iso)
            matched_track_ids.add(tid)

        # Step 4: Prune expired tracks
        expired_ids = []
        for tid, trk in self._active_tracks.items():
            if now_mono - trk.last_seen_monotonic > self.lost_track_timeout:
                expired_ids.append(tid)

        for tid in expired_ids:
            self._active_tracks.pop(tid, None)

        return list(self._active_tracks.values())

    def _create_new_track(
        self,
        det: Detection,
        track_id: int,
        raw_frame: np.ndarray,
        frame_index: int,
        now_mono: float,
        now_iso: str,
    ) -> TrackState:
        """Create a new unique detection event and save single evidence image."""
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        record_id = f"rec_{timestamp_str}_{uuid.uuid4().hex[:6]}"
        event_id = f"evt_{timestamp_str}_{uuid.uuid4().hex[:6]}"
        det.track_id = track_id

        track = TrackState(
            track_id=track_id,
            event_id=event_id,
            class_id=det.class_id,
            class_name=det.class_name,
            confidence=det.confidence,
            peak_confidence=det.confidence,
            first_seen_monotonic=now_mono,
            last_seen_monotonic=now_mono,
            first_seen_iso=now_iso,
            last_seen_iso=now_iso,
            frame_index=frame_index,
            bbox=det.bbox,
            record_id=record_id,
        )
        self._active_tracks[track_id] = track

        x1 = det.bbox[0]
        y1 = det.bbox[1]
        x2 = det.bbox[0] + det.bbox[2]
        y2 = det.bbox[1] + det.bbox[3]

        event = DetectionEvent(
            record_id=record_id,
            session_id=self.storage_manager.session_id,
            event_id=event_id,
            track_id=track_id,
            detected_at=now_iso,
            source_type=self.storage_manager.source_type,
            source_name=self.storage_manager.source_reference,
            frame_number=frame_index,
            class_id=det.class_id,
            class_name=det.class_name,
            confidence=det.confidence,
            bbox_x1=x1,
            bbox_y1=y1,
            bbox_x2=x2,
            bbox_y2=y2,
            model_name=self.storage_manager.model_name,
            checkpoint_filename=self.storage_manager.checkpoint_filename,
        )
        persisted = self.storage_manager.record_event(event)

        if persisted:
            logger.info(
                "New Detection Event: %s | Track #%d | %s (%.2f)",
                event_id,
                track_id,
                det.class_name,
                det.confidence,
            )

            if self.on_new_event:
                try:
                    self.on_new_event(event)
                except Exception as e:
                    logger.error("Error in on_new_event callback: %s", e)
        else:
            err_detail = getattr(self.storage_manager, "last_write_error", "") or "CSV write failed"
            print(f"\n[STORAGE ERROR] Detection occurred, but CSV record was not saved: {err_detail}")
            logger.warning(
                "Detection event %s (Track #%d) pending CSV write: %s",
                event_id,
                track_id,
                err_detail,
            )

        return track


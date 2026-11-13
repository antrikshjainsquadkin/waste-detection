"""Object tracking and continuous-event deduplication engine.

Preserves track identity across frames, prevents duplicate CSV records
for the same continuous object, and manages evidence image saving.
"""

from __future__ import annotations

import datetime
import logging
import time
import uuid
from dataclasses import dataclass, field
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
    last_seen_frame: int = 0
    evidence_image_path: str = ""
    record_id: str = ""
    class_scores: Dict[int, float] = field(default_factory=dict)
    class_names: Dict[int, str] = field(default_factory=dict)
    raw_class_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class _PendingTrack:
    """Buffer for a newly observed track awaiting confirmation."""
    track_id: int
    hit_count: int
    first_seen_monotonic: float
    last_seen_monotonic: float
    first_seen_iso: str
    last_seen_iso: str
    first_frame: int
    last_seen_frame: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    peak_confidence: float
    class_scores: Dict[int, float] = field(default_factory=dict)
    class_names: Dict[int, str] = field(default_factory=dict)
    raw_class_counts: Dict[str, int] = field(default_factory=dict)


def _update_track_class(
    class_scores: Dict[int, float],
    class_names: Dict[int, str],
    raw_class_counts: Dict[str, int],
    det: Detection,
    decay: float = 0.97,
) -> Tuple[int, str]:
    """Update confidence-weighted EMA class scores while preserving raw counts."""
    raw_class_counts[det.class_name] = raw_class_counts.get(det.class_name, 0) + 1
    for cid in list(class_scores):
        class_scores[cid] *= decay
    class_scores[det.class_id] = class_scores.get(det.class_id, 0.0) + float(det.confidence)
    class_names[det.class_id] = det.class_name
    best_cid = max(class_scores, key=lambda k: class_scores[k])
    return best_cid, class_names[best_cid]


class DetectionTracker:
    """Tracks detected objects across frames and prevents duplicate CSV records."""

    def __init__(
        self,
        storage_manager: StorageManager,
        iou_threshold: float = 0.35,
        lost_track_timeout: float = 3.0,
        on_new_event: Optional[Callable[[DetectionEvent], None]] = None,
        source_fps: Optional[float] = None,
        min_confirmation_frames: int = 1,
        stabilize_classes: bool = True,
    ):
        self.storage_manager = storage_manager
        self.iou_threshold = float(iou_threshold)
        self.lost_track_timeout = float(lost_track_timeout)
        self.on_new_event = on_new_event
        self.source_fps = float(source_fps) if source_fps is not None and source_fps > 0 else None
        self.min_confirmation_frames = max(1, int(min_confirmation_frames))
        self.stabilize_classes = bool(stabilize_classes)

        self._active_tracks: Dict[int, TrackState] = {}
        self._pending_tracks: Dict[int, _PendingTrack] = {}
        self._next_fallback_id = 1000

    def reset_session(self) -> None:
        """Clear per-session in-memory track state without touching persisted CSV history."""
        self._active_tracks.clear()
        self._pending_tracks.clear()
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
                    track.last_seen_frame = frame_index
                    track.peak_confidence = max(track.peak_confidence, det.confidence)
                    track.confidence = det.confidence
                    track.bbox = det.bbox
                    best_cid, best_cname = _update_track_class(
                        track.class_scores, track.class_names, track.raw_class_counts, det
                    )
                    if self.stabilize_classes:
                        track.class_id = best_cid
                        track.class_name = best_cname
                        det.class_id = best_cid
                        det.class_name = best_cname

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
                elif self.min_confirmation_frames > 1:
                    pending = self._pending_tracks.get(tid)
                    if pending is None:
                        pending = _PendingTrack(
                            track_id=tid,
                            hit_count=0,
                            first_seen_monotonic=now_mono,
                            last_seen_monotonic=now_mono,
                            first_seen_iso=now_iso,
                            last_seen_iso=now_iso,
                            first_frame=frame_index,
                            last_seen_frame=frame_index,
                            bbox=det.bbox,
                            confidence=det.confidence,
                            peak_confidence=det.confidence,
                        )
                        self._pending_tracks[tid] = pending
                    pending.hit_count += 1
                    pending.last_seen_monotonic = now_mono
                    pending.last_seen_iso = now_iso
                    pending.last_seen_frame = frame_index
                    pending.bbox = det.bbox
                    pending.confidence = det.confidence
                    pending.peak_confidence = max(pending.peak_confidence, det.confidence)
                    best_cid, best_cname = _update_track_class(
                        pending.class_scores, pending.class_names, pending.raw_class_counts, det
                    )
                    if self.stabilize_classes:
                        det.class_id = best_cid
                        det.class_name = best_cname

                    if pending.hit_count >= self.min_confirmation_frames:
                        self._pending_tracks.pop(tid, None)
                        trk = self._create_new_track(det, tid, raw_frame, frame_index, now_mono, now_iso)
                        trk.class_scores = pending.class_scores
                        trk.class_names = pending.class_names
                        trk.raw_class_counts = pending.raw_class_counts
                        trk.peak_confidence = pending.peak_confidence
                        matched_track_ids.add(tid)
                else:
                    # New track with native track_id
                    trk = self._create_new_track(det, tid, raw_frame, frame_index, now_mono, now_iso)
                    _update_track_class(trk.class_scores, trk.class_names, trk.raw_class_counts, det)
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
                    best_track.last_seen_frame = frame_index
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

        # Step 4: Prune expired active and pending tracks (use media time when source_fps is known)
        expired_ids = []
        for tid, trk in self._active_tracks.items():
            if self.source_fps is not None:
                elapsed_s = (frame_index - trk.last_seen_frame) / self.source_fps
            else:
                elapsed_s = now_mono - trk.last_seen_monotonic
            if elapsed_s > self.lost_track_timeout:
                expired_ids.append(tid)

        for tid in expired_ids:
            self._active_tracks.pop(tid, None)

        expired_pending = []
        for tid, p_trk in self._pending_tracks.items():
            if self.source_fps is not None:
                elapsed_s = (frame_index - p_trk.last_seen_frame) / self.source_fps
            else:
                elapsed_s = now_mono - p_trk.last_seen_monotonic
            if elapsed_s > self.lost_track_timeout:
                expired_pending.append(tid)

        for tid in expired_pending:
            self._pending_tracks.pop(tid, None)

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
            last_seen_frame=frame_index,
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


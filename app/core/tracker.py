"""Object tracking and continuous-event association engine.

Maintains spatial-temporal identity of waste items across consecutive frames
to prevent duplicate detection events while an item remains in view.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, List, Optional, Tuple

from app.core.types import BoundingBox, Detection, TrackedObject

logger = logging.getLogger(__name__)


class ObjectTracker:
    """IoU-based spatial-temporal object tracker for video streams."""

    def __init__(
        self,
        iou_threshold: float = 0.35,
        lost_track_timeout_seconds: float = 3.0,
        min_hits: int = 1,
    ):
        self.iou_threshold = iou_threshold
        self.lost_track_timeout = lost_track_timeout_seconds
        self.min_hits = min_hits
        self._next_track_id: int = 1
        self._tracks: Dict[int, TrackedObject] = {}

        # Callbacks
        self.on_track_created: Optional[Callable[[TrackedObject], None]] = None
        self.on_track_updated: Optional[Callable[[TrackedObject], None]] = None
        self.on_track_finalized: Optional[Callable[[TrackedObject], None]] = None

    def reset(self) -> None:
        """Clear all active tracks and reset track counter (e.g., on source change or reconnection)."""
        now = time.monotonic()
        for track in list(self._tracks.values()):
            if not track.finalized and self.on_track_finalized:
                track.finalized = True
                self.on_track_finalized(track)
        self._tracks.clear()
        self._next_track_id = 1
        logger.debug("Tracker reset.")

    @property
    def active_tracks(self) -> List[TrackedObject]:
        """Return currently active, un-finalized tracks."""
        return list(self._tracks.values())

    def update(self, detections: List[Detection], current_time: Optional[float] = None) -> List[TrackedObject]:
        """Update tracker with new detections from the current frame.

        Args:
            detections: List of Detection objects detected in the current frame.
            current_time: Monotonic timestamp for time calculations.

        Returns:
            List of active TrackedObject instances.
        """
        now = current_time if current_time is not None else time.monotonic()

        active_track_ids = list(self._tracks.keys())
        active_track_list = [self._tracks[tid] for tid in active_track_ids]

        # Compute IoU matrix between active tracks and current detections
        num_tracks = len(active_track_list)
        num_dets = len(detections)

        matched_tracks = set()
        matched_dets = set()

        if num_tracks > 0 and num_dets > 0:
            # Build cost/IoU matrix
            iou_matrix: List[List[float]] = []
            for t in active_track_list:
                row = []
                for d in detections:
                    # Category matching preference: same category or unknown transition
                    if t.class_name == d.class_name or t.is_unknown or d.is_unknown:
                        score = t.bbox.iou(d.bbox)
                    else:
                        # Heavy penalty if categories differ completely
                        score = t.bbox.iou(d.bbox) * 0.5
                    row.append(score)
                iou_matrix.append(row)

            # Greedy matching on highest IoU
            pairs: List[Tuple[float, int, int]] = []
            for t_idx in range(num_tracks):
                for d_idx in range(num_dets):
                    iou_val = iou_matrix[t_idx][d_idx]
                    if iou_val >= self.iou_threshold:
                        pairs.append((iou_val, t_idx, d_idx))

            pairs.sort(key=lambda x: x[0], reverse=True)

            for score, t_idx, d_idx in pairs:
                if t_idx in matched_tracks or d_idx in matched_dets:
                    continue
                matched_tracks.add(t_idx)
                matched_dets.add(d_idx)

                track = active_track_list[t_idx]
                det = detections[d_idx]

                # Update track state with new observation
                track.bbox = det.bbox
                track.last_seen = now
                track.hit_count += 1
                track.miss_count = 0

                # If category was unknown and now we have a confident category, upgrade it
                if track.is_unknown and not det.is_unknown:
                    track.class_id = det.class_id
                    track.class_name = det.class_name
                    track.display_name = det.display_name
                    track.is_unknown = False
                    track.confidence = det.confidence
                else:
                    # Keep maximum confidence observed
                    track.confidence = max(track.confidence, det.confidence)

                if self.on_track_updated:
                    self.on_track_updated(track)

        # Handle unmatched detections -> create new tracks
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                new_track = TrackedObject(
                    track_id=self._next_track_id,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    display_name=det.display_name,
                    confidence=det.confidence,
                    bbox=det.bbox,
                    first_seen=now,
                    last_seen=now,
                    hit_count=1,
                    miss_count=0,
                    is_unknown=det.is_unknown,
                    event_id=None,
                    event_created=False,
                    finalized=False,
                )
                self._tracks[self._next_track_id] = new_track
                self._next_track_id += 1

                if self.on_track_created:
                    self.on_track_created(new_track)

        # Check unmatched tracks for timeout
        dead_track_ids: List[int] = []
        for t_idx, track in enumerate(active_track_list):
            if t_idx not in matched_tracks:
                track.miss_count += 1
                elapsed = now - track.last_seen
                if elapsed > self.lost_track_timeout:
                    track.finalized = True
                    dead_track_ids.append(track.track_id)
                    if self.on_track_finalized:
                        self.on_track_finalized(track)

        for tid in dead_track_ids:
            self._tracks.pop(tid, None)

        return list(self._tracks.values())


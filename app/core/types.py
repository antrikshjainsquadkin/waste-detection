"""Core data types and structures for the Waste Detection System.

Defines unified dataclasses and enums used across inference, tracking,
recording, persistence, and UI presentation layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class SourceType(str, Enum):
    """Supported video input source types."""
    WEBCAM = "webcam"
    RTSP = "rtsp"
    VIDEO_FILE = "video_file"


class SessionStatus(str, Enum):
    """Lifecycle status for a monitoring session."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    STOPPED = "stopped"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass(slots=True)
class BoundingBox:
    """Normalized or pixel bounding box coordinates (top-left x, y, width, height)."""
    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def to_ltrb(self) -> Tuple[int, int, int, int]:
        """Return (left, top, right, bottom)."""
        return self.x, self.y, self.x2, self.y2

    def to_xywh(self) -> Tuple[int, int, int, int]:
        """Return (x, y, width, height)."""
        return self.x, self.y, self.width, self.height

    def iou(self, other: BoundingBox) -> float:
        """Compute Intersection over Union (IoU) with another bounding box."""
        x_left = max(self.x, other.x)
        y_top = max(self.y, other.y)
        x_right = min(self.x2, other.x2)
        y_bottom = min(self.y2, other.y2)

        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = self.area + other.area - intersection
        if union <= 0:
            return 0.0
        return float(intersection) / float(union)


@dataclass(slots=True)
class Detection:
    """Raw detection output from an object detection model."""
    class_id: int
    class_name: str
    display_name: str
    confidence: float
    bbox: BoundingBox
    is_unknown: bool = False


@dataclass(slots=True)
class TrackedObject:
    """Tracked entity across consecutive frames."""
    track_id: int
    class_id: int
    class_name: str
    display_name: str
    confidence: float
    bbox: BoundingBox
    first_seen: float
    last_seen: float
    hit_count: int = 1
    miss_count: int = 0
    is_unknown: bool = False
    event_id: Optional[str] = None
    event_created: bool = False
    finalized: bool = False


@dataclass(slots=True)
class DetectionEvent:
    """A distinct detection event representing an object seen across frames."""
    event_id: str
    session_id: str
    track_id: int
    category: str
    display_name: str
    confidence: float
    first_seen: str
    last_seen: str
    duration_seconds: float
    bbox: BoundingBox
    frame_width: int
    frame_height: int
    source_type: str
    source_reference: str
    model_name: str
    model_version: str
    evidence_image_path: str = ""
    is_unknown: bool = False

    def to_csv_dict(self) -> Dict[str, str]:
        """Serialize event to documented CSV dictionary schema."""
        return {
            "session_id": self.session_id,
            "event_id": self.event_id,
            "track_id": str(self.track_id),
            "category": self.category,
            "confidence": f"{self.confidence:.4f}",
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "duration_seconds": f"{self.duration_seconds:.2f}",
            "bbox_x": str(self.bbox.x),
            "bbox_y": str(self.bbox.y),
            "bbox_width": str(self.bbox.width),
            "bbox_height": str(self.bbox.height),
            "frame_width": str(self.frame_width),
            "frame_height": str(self.frame_height),
            "source_type": self.source_type,
            "source_reference": self.source_reference,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "evidence_image_path": self.evidence_image_path,
        }


@dataclass
class SessionMetadata:
    """Complete summary and metadata of a monitoring session."""
    session_id: str
    start_time: str
    end_time: str = ""
    duration_seconds: float = 0.0
    source_type: str = ""
    source_reference: str = ""
    total_events: int = 0
    category_counts: Dict[str, int] = field(default_factory=dict)
    unknown_count: int = 0
    recording_path: str = ""
    model_name: str = ""
    model_version: str = ""
    average_fps: float = 0.0
    status: str = SessionStatus.INITIALIZING.value
    interruptions: List[str] = field(default_factory=list)

    def to_summary_csv_dict(self) -> Dict[str, str]:
        """Serialize session summary to documented CSV dictionary schema."""
        row: Dict[str, str] = {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": f"{self.duration_seconds:.2f}",
            "source_type": self.source_type,
            "source_reference": self.source_reference,
            "total_events": str(self.total_events),
            "unknown_count": str(self.unknown_count),
            "recording_path": self.recording_path,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "average_fps": f"{self.average_fps:.2f}",
            "status": self.status,
        }
        # Add per-category counts
        for cat, cnt in sorted(self.category_counts.items()):
            row[f"count_{cat}"] = str(cnt)
        return row


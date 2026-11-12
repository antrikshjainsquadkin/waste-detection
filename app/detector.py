"""YOLO11 Waste Detector implementation.

Wraps the pretrained khanhromvn/waste-detection-yolo11 model,
providing predict and track operations with native category mappings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from app.model_loader import EXPECTED_CLASSES, _file_signature, load_waste_model

logger = logging.getLogger(__name__)

_VERIFIER_CACHE_LOCK = threading.Lock()
_VERIFIER_MODEL_CACHE: Dict[Tuple[str, int, int], Tuple[Any, Dict[int, str]]] = {}


class VerificationState(str, Enum):
    """Detection decision states."""
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(slots=True)
class VerificationResult:
    """Detailed audit record for a verified candidate detection."""
    detection: Detection
    state: VerificationState
    reason: str
    non_waste_class: Optional[str] = None
    non_waste_conf: Optional[float] = None
    overlap_iou: float = 0.0
    overlap_ioa: float = 0.0
    relative_area: float = 0.0


def compute_box_overlap(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int],
) -> Tuple[float, float, float]:
    """Compute (IoU, IoA, relative_area) between box1 (candidate) and box2 (non-waste).

    box format: (x, y, w, h)
    - IoU: intersection / union
    - IoA: intersection / area(box1) (fraction of candidate enclosed in non-waste)
    - relative_area: area(box1) / area(box2)
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    area1 = max(0, w1) * max(0, h1)
    area2 = max(0, w2) * max(0, h2)
    if area1 <= 0 or area2 <= 0:
        return 0.0, 0.0, 0.0

    inter_x = max(x1, x2)
    inter_y = max(y1, y2)
    inter_right = min(x1 + w1, x2 + w2)
    inter_bottom = min(y1 + h1, y2 + h2)

    inter_w = max(0, inter_right - inter_x)
    inter_h = max(0, inter_bottom - inter_y)
    intersection = inter_w * inter_h

    if intersection <= 0:
        return 0.0, 0.0, area1 / area2

    union = area1 + area2 - intersection
    iou = float(intersection) / float(union) if union > 0 else 0.0
    ioa = float(intersection) / float(area1)
    rel_area = float(area1) / float(area2)

    return iou, ioa, rel_area


class NonWasteVerifier:
    """Secondary detector to identify humans, animals, and non-waste objects.

    Rejects false detections when strong human/animal overlap exists,
    while preserving genuine waste items held by or near people.
    """

    DEFAULT_LOCAL_PATHS = [
        Path("models/pretrained/yolov8n.pt"),
        Path(r"C:\Users\A\OneDrive\Desktop\vehicle_detect\yolov8n.pt"),
    ]

    PERSON_CLASSES = {"person"}
    ANIMAL_CLASSES = {
        "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe"
    }
    CORROBORATING_WASTE_CLASSES = {
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
        "hot dog", "pizza", "donut", "cake", "book", "toothbrush",
    }
    FURNITURE_VEHICLE_CLASSES = {
        "chair", "couch", "bed", "dining table", "toilet", "tv", "laptop",
        "refrigerator", "oven", "microwave", "sink", "car", "motorcycle",
        "bus", "truck", "bicycle", "bench",
    }

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        confidence_threshold: float = 0.35,
        device: str = "cpu",
        rejection_iou: float = 0.40,
        rejection_ioa: float = 0.65,
        relative_area_threshold: float = 0.25,
        enabled: bool = True,
        imgsz: int = 416,
        cache_ttl_frames: int = 3,
    ):
        self.conf = float(confidence_threshold)
        self.device = device
        self.rejection_iou = float(rejection_iou)
        self.rejection_ioa = float(rejection_ioa)
        self.relative_area_threshold = float(relative_area_threshold)
        self.enabled = enabled
        self.imgsz = int(imgsz)
        self.cache_ttl_frames = int(cache_ttl_frames)
        self.weights_path: Optional[Path] = None
        self.model: Any = None
        self.is_available = False
        self.class_mapping: Dict[int, str] = {}
        self.last_inference_ms: float = 0.0
        self.last_filtering_ms: float = 0.0
        self._frame_counter: int = 0
        self._last_verify_had_non_waste: bool = True
        self._track_cache: Dict[int, Tuple[Tuple[int, int, int, int], int, VerificationState, str]] = {}

        if not self.enabled:
            logger.info("NonWasteVerifier is disabled by configuration.")
            return

        # Resolve weights path from explicit argument or local caches
        target_path = weights_path
        if not target_path or not Path(target_path).exists():
            for p in self.DEFAULT_LOCAL_PATHS:
                if p.exists():
                    target_path = p
                    break

        if target_path and Path(target_path).exists():
            try:
                self.weights_path = Path(target_path)
                sig = _file_signature(self.weights_path)
                cached = None
                if sig is not None:
                    with _VERIFIER_CACHE_LOCK:
                        cached = _VERIFIER_MODEL_CACHE.get(sig)

                if cached is not None:
                    self.model, cached_mapping = cached
                    self.class_mapping = dict(cached_mapping)
                    self.is_available = True
                    logger.info(
                        "Reusing cached NonWasteVerifier from %s (%d classes) on %s",
                        self.weights_path,
                        len(self.class_mapping),
                        self.device,
                    )
                else:
                    from ultralytics import YOLO
                    self.model = YOLO(str(self.weights_path))
                    self.class_mapping = {int(k): str(v).lower() for k, v in self.model.names.items()}
                    self.is_available = True
                    if sig is not None:
                        with _VERIFIER_CACHE_LOCK:
                            _VERIFIER_MODEL_CACHE[sig] = (self.model, dict(self.class_mapping))
                    logger.info(
                        "NonWasteVerifier initialized from %s with %d classes on %s",
                        self.weights_path,
                        len(self.class_mapping),
                        self.device,
                    )
            except Exception as e:
                logger.warning("Failed to initialize NonWasteVerifier from %s: %s", target_path, e)
                self.is_available = False
        else:
            logger.info("No secondary verifier weights found; verification stage inactive.")

    def reset_cache(self) -> None:
        """Clear per-session selective verification cache."""
        self._frame_counter = 0
        self._last_verify_had_non_waste = True
        self._track_cache.clear()

    def verify_candidates(
        self,
        frame: np.ndarray,
        candidates: List[Detection],
    ) -> Tuple[List[Detection], List[VerificationResult], Dict[str, int]]:
        """Filter candidate waste detections against verified person and animal detections.

        Returns:
            (accepted_detections, verification_results, summary_counts)
        """
        counts = {"accepted": 0, "rejected": 0, "ambiguous": 0}
        results: List[VerificationResult] = []

        if not candidates:
            return [], results, counts

        if not self.is_available or self.model is None or frame is None or frame.size == 0:
            self.last_inference_ms = 0.0
            self.last_filtering_ms = 0.0
            for c in candidates:
                vr = VerificationResult(
                    detection=c,
                    state=VerificationState.ACCEPTED,
                    reason="verifier_unavailable",
                )
                results.append(vr)
                counts["accepted"] += 1
            return candidates, results, counts

        # Safe selective verification cache: skip secondary inference only when
        # no non-waste objects were present in the last verified frame AND every
        # candidate has a recently verified, low-drift track_id.
        self._frame_counter += 1
        if (
            self.cache_ttl_frames > 1
            and not self._last_verify_had_non_waste
            and all(c.track_id is not None and c.track_id in self._track_cache for c in candidates)
        ):
            cache_valid = True
            for c in candidates:
                cached_box, cached_frame, cached_state, _ = self._track_cache[c.track_id]
                if (self._frame_counter - cached_frame) >= self.cache_ttl_frames or cached_state != VerificationState.ACCEPTED:
                    cache_valid = False
                    break
                iou_drift, _, _ = compute_box_overlap(c.bbox, cached_box)
                if iou_drift < 0.85:
                    cache_valid = False
                    break
            if cache_valid:
                self.last_inference_ms = 0.0
                t_filter_0 = time.perf_counter_ns()
                for c in candidates:
                    _, _, cached_state, cached_reason = self._track_cache[c.track_id]
                    results.append(VerificationResult(detection=c, state=cached_state, reason=f"cached_{cached_reason}"))
                    counts["accepted"] += 1
                self.last_filtering_ms = (time.perf_counter_ns() - t_filter_0) / 1_000_000
                return candidates, results, counts

        # Run secondary model inference on the same frame
        t_sec_infer_0 = time.perf_counter_ns()
        with torch.inference_mode():
            sec_results = self.model.predict(
                source=frame,
                conf=self.conf,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
        self.last_inference_ms = (time.perf_counter_ns() - t_sec_infer_0) / 1_000_000

        t_filter_0 = time.perf_counter_ns()
        non_waste_boxes: List[Tuple[str, float, Tuple[int, int, int, int]]] = []
        corroborating_boxes: List[Tuple[str, float, Tuple[int, int, int, int]]] = []
        furniture_vehicle_boxes: List[Tuple[str, float, Tuple[int, int, int, int]]] = []
        if sec_results and len(sec_results) > 0:
            first_res = sec_results[0]
            boxes = getattr(first_res, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                xywh = boxes.xywh.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy()
                for i in range(len(boxes)):
                    cid = int(classes[i])
                    cname = self.class_mapping.get(cid, "").lower()
                    is_bio = cname in self.PERSON_CLASSES or cname in self.ANIMAL_CLASSES
                    is_corr = cname in self.CORROBORATING_WASTE_CLASSES
                    is_fv = cname in self.FURNITURE_VEHICLE_CLASSES
                    if is_bio or is_corr or is_fv:
                        cx, cy, w, h = xywh[i]
                        x = int(cx - w / 2.0)
                        y = int(cy - h / 2.0)
                        bw = int(w)
                        bh = int(h)
                        if is_bio:
                            non_waste_boxes.append((cname, float(confs[i]), (x, y, bw, bh)))
                        elif is_corr:
                            corroborating_boxes.append((cname, float(confs[i]), (x, y, bw, bh)))
                        else:
                            furniture_vehicle_boxes.append((cname, float(confs[i]), (x, y, bw, bh)))

        self._last_verify_had_non_waste = bool(non_waste_boxes or furniture_vehicle_boxes)

        # If no people, animals, or furniture/vehicle boxes detected, accept all candidates
        if not non_waste_boxes and not furniture_vehicle_boxes:
            for c in candidates:
                vr = VerificationResult(
                    detection=c,
                    state=VerificationState.ACCEPTED,
                    reason="no_non_waste_overlap",
                )
                results.append(vr)
                counts["accepted"] += 1
                if c.track_id is not None:
                    self._track_cache[c.track_id] = (c.bbox, self._frame_counter, VerificationState.ACCEPTED, "no_non_waste_overlap")
            self.last_filtering_ms = (time.perf_counter_ns() - t_filter_0) / 1_000_000
            return candidates, results, counts

        # Filter each candidate against detected people/animals and furniture/vehicles
        accepted: List[Detection] = []
        for cand in candidates:
            cand_box = cand.bbox
            worst_state = VerificationState.ACCEPTED
            worst_reason = "clear"
            matched_cls = None
            matched_conf = None
            max_iou = 0.0
            max_ioa = 0.0
            max_rel_area = 0.0

            # Check if secondary COCO model independently corroborates a waste-compatible object
            corr_cls: Optional[str] = None
            for c_name, _, c_box in corroborating_boxes:
                c_iou, c_ioa, _ = compute_box_overlap(cand_box, c_box)
                if c_iou >= 0.25 or c_ioa >= 0.50:
                    corr_cls = c_name
                    break

            for nw_cls, nw_conf, nw_box in non_waste_boxes:
                iou, ioa, rel_area = compute_box_overlap(cand_box, nw_box)
                if max(iou, ioa) > max(max_iou, max_ioa):
                    max_ioa = ioa
                    max_iou = iou
                    max_rel_area = rel_area
                    matched_cls = nw_cls
                    matched_conf = nw_conf

                # If corroborated by a COCO object (e.g. bottle/cup) and not a full-body match, preserve waste
                if corr_cls is not None and iou < 0.50:
                    worst_reason = f"corroborated_by_{corr_cls}"
                    continue

                # Evaluation rules:
                # 1. Strong human/animal match (candidate box matches person/animal extent)
                if iou >= self.rejection_iou and ioa >= 0.60 and rel_area >= self.relative_area_threshold:
                    worst_state = VerificationState.REJECTED
                    worst_reason = f"strong_{nw_cls}_overlap (IoU={iou:.2f}, IoA={ioa:.2f}, rel_area={rel_area:.2f})"
                    max_ioa, max_iou, max_rel_area = ioa, iou, rel_area
                    matched_cls, matched_conf = nw_cls, nw_conf
                    break
                # 2. Large candidate enclosed inside person/animal box (torso/body false positive)
                elif ioa >= 0.80 and rel_area >= 0.38:
                    worst_state = VerificationState.REJECTED
                    worst_reason = f"enclosed_in_{nw_cls} (IoA={ioa:.2f}, rel_area={rel_area:.2f})"
                    max_ioa, max_iou, max_rel_area = ioa, iou, rel_area
                    matched_cls, matched_conf = nw_cls, nw_conf
                    break
                # 3. Conflicting moderate overlap with confident non-waste and non-high primary confidence
                elif (
                    (ioa >= 0.55 and rel_area >= 0.30 and iou >= 0.22)
                    or (0.28 <= iou < self.rejection_iou and rel_area >= 0.25)
                ) and nw_conf >= 0.45 and cand.confidence < 0.70:
                    if worst_state != VerificationState.REJECTED:
                        worst_state = VerificationState.AMBIGUOUS
                        worst_reason = f"ambiguous_{nw_cls}_boundary (IoU={iou:.2f}, IoA={ioa:.2f}, rel_area={rel_area:.2f})"

            # 4. Non-biological false positives: strict 1:1 mutual match with furniture/vehicles (e.g. chair/table/car)
            if worst_state == VerificationState.ACCEPTED and corr_cls is None:
                for fv_cls, fv_conf, fv_box in furniture_vehicle_boxes:
                    iou, ioa, rel_area = compute_box_overlap(cand_box, fv_box)
                    if iou >= 0.55 and ioa >= 0.75 and fv_conf >= 0.50:
                        worst_state = VerificationState.REJECTED
                        worst_reason = f"non_waste_{fv_cls}_match (IoU={iou:.2f}, IoA={ioa:.2f})"
                        max_ioa, max_iou, max_rel_area = ioa, iou, rel_area
                        matched_cls, matched_conf = fv_cls, fv_conf
                        break

            vr = VerificationResult(
                detection=cand,
                state=worst_state,
                reason=worst_reason,
                non_waste_class=matched_cls,
                non_waste_conf=matched_conf,
                overlap_iou=max_iou,
                overlap_ioa=max_ioa,
                relative_area=max_rel_area,
            )
            results.append(vr)
            counts[worst_state.value.lower()] += 1
            if cand.track_id is not None:
                self._track_cache[cand.track_id] = (cand_box, self._frame_counter, worst_state, worst_reason)

            if worst_state == VerificationState.ACCEPTED:
                accepted.append(cand)

        self.last_filtering_ms = (time.perf_counter_ns() - t_filter_0) / 1_000_000
        return accepted, results, counts


@dataclass(slots=True)
class Detection:
    """Standardized detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    track_id: Optional[int] = None

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return self.bbox_x, self.bbox_y, self.bbox_w, self.bbox_h


class WasteDetector:
    """Pretrained YOLO11 waste detector and tracker."""

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        confidence_threshold: float = 0.25,
        image_size: int = 640,
        device: str = "cpu",
        verifier_weights_path: Optional[Path] = None,
        verifier_confidence: float = 0.35,
        verifier_enabled: bool = True,
    ):
        self.conf = float(confidence_threshold)
        self.imgsz = int(image_size)
        self.requested_device = device

        # Load and verify model
        t_prim_0 = time.perf_counter_ns()
        self.model, self.weights_path, self.sha256, self.class_mapping = load_waste_model(
            weights_path=weights_path,
            device=self.requested_device,
        )
        self.primary_load_time_ms = (time.perf_counter_ns() - t_prim_0) / 1_000_000

        self.model_name = "khanhromvn/waste-detection-yolo11 (waste_detection_yolo11m.pt)"
        self.device = self._resolve_device(device)
        self._tracking_supported = True

        # Initialize secondary non-waste verifier (humans & animals)
        t_sec_0 = time.perf_counter_ns()
        self.verifier = NonWasteVerifier(
            weights_path=verifier_weights_path,
            confidence_threshold=verifier_confidence,
            device=self.device,
            enabled=verifier_enabled,
        )
        self.secondary_load_time_ms = (time.perf_counter_ns() - t_sec_0) / 1_000_000
        self.total_startup_time_ms = self.primary_load_time_ms + self.secondary_load_time_ms

        logger.info(
            "WasteDetector ready on device: %s (verifier active: %s)",
            self.device,
            self.verifier.is_available,
        )

    def _resolve_device(self, requested: str) -> str:
        """Verify device availability without forcing GPU."""
        has_cuda = torch.cuda.is_available()
        if requested.lower() == "cuda":
            if has_cuda:
                return "cuda"
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        elif requested.lower() == "cpu":
            return "cpu"
        return "cuda" if has_cuda else "cpu"

    @property
    def has_gpu(self) -> bool:
        return torch.cuda.is_available()

    def reset_tracking(self) -> int:
        """Reset session-specific ByteTrack and verifier cache state without reloading weights.

        Safe and idempotent when called before any tracking has occurred.
        Returns the number of tracker instances reset.
        """
        reset_count = 0
        predictor = getattr(self.model, "predictor", None)
        if predictor is not None:
            trackers = getattr(predictor, "trackers", None)
            if trackers:
                for trk in trackers:
                    reset_fn = getattr(trk, "reset", None)
                    if callable(reset_fn):
                        reset_fn()
                        reset_count += 1
            vid_path = getattr(predictor, "vid_path", None)
            if isinstance(vid_path, list):
                for i in range(len(vid_path)):
                    vid_path[i] = None

        if hasattr(self, "verifier") and self.verifier is not None:
            self.verifier.reset_cache()

        return reset_count

    def predict(self, frame: np.ndarray) -> List[Detection]:
        """Run standard inference on a single image/frame."""
        if frame is None or frame.size == 0:
            return []

        with torch.inference_mode():
            results = self.model.predict(
                source=frame,
                conf=self.conf,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
        return self._extract_detections(results)

    def track(self, frame: np.ndarray) -> List[Detection]:
        """Run inference with object tracking enabled to preserve IDs across frames."""
        if frame is None or frame.size == 0:
            return []

        if not self._tracking_supported:
            return self.predict(frame)

        try:
            with torch.inference_mode():
                results = self.model.track(
                    source=frame,
                    conf=self.conf,
                    imgsz=self.imgsz,
                    device=self.device,
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False,
                )
            return self._extract_detections(results)
        except Exception as e:
            self._tracking_supported = False
            logger.warning("Native tracking failed (%s); permanently falling back to single-frame predict + spatial tracker for session.", e)
            return self.predict(frame)

    def _extract_detections(self, results: List[Any]) -> List[Detection]:
        """Convert Ultralytics results into structured Detection dataclasses."""
        detections: List[Detection] = []
        if not results:
            return detections

        first_res = results[0]
        boxes = getattr(first_res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detections

        xywh = boxes.xywh.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        track_ids = None
        if hasattr(boxes, "id") and boxes.id is not None:
            track_ids = boxes.id.cpu().numpy().astype(int)

        for i in range(len(boxes)):
            cx, cy, w, h = xywh[i]
            x = int(cx - w / 2.0)
            y = int(cy - h / 2.0)
            bw = int(w)
            bh = int(h)
            cid = int(classes[i])
            cname = self.class_mapping.get(cid, EXPECTED_CLASSES.get(cid, "unknown"))
            conf = float(confs[i])
            tid = int(track_ids[i]) if track_ids is not None else None

            detections.append(
                Detection(
                    class_id=cid,
                    class_name=cname,
                    confidence=conf,
                    bbox_x=x,
                    bbox_y=y,
                    bbox_w=bw,
                    bbox_h=bh,
                    track_id=tid,
                )
            )

        return detections

    def verify_candidates(
        self,
        frame: np.ndarray,
        candidates: List[Detection],
    ) -> Tuple[List[Detection], List[VerificationResult], Dict[str, int]]:
        """Filter candidate detections through the secondary verification stage.

        Returns:
            (accepted_detections, verification_results, summary_counts)
        """
        return self.verifier.verify_candidates(frame, candidates)

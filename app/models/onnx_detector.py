"""ONNX Runtime CPU object detector.

Provides efficient, CPU-optimized local inference with aspect-ratio letterboxing,
Non-Maximum Suppression (NMS), per-category confidence filtering, and Unknown classification.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.core.types import BoundingBox, Detection
from app.models.base import BaseDetector

logger = logging.getLogger(__name__)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> List[int]:
    """Pure-numpy Non-Maximum Suppression.

    Args:
        boxes: Array of shape (N, 4) in (x1, y1, x2, y2) format.
        scores: Array of shape (N,) containing confidence scores.
        iou_threshold: IoU overlap threshold.

    Returns:
        List of integer indices to keep.
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter
        iou = np.zeros_like(inter)
        valid = union > 0
        iou[valid] = inter[valid] / union[valid]

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


class OnnxWasteDetector(BaseDetector):
    """Production ONNX Runtime object detector targeting CPU execution."""

    def __init__(
        self,
        model_path: Path,
        model_name: str,
        version: str,
        class_mapping: Dict[int, str],
        input_size: Tuple[int, int] = (416, 416),
        nms_threshold: float = 0.45,
    ):
        self.model_path = Path(model_path)
        self.model_name = model_name
        self.version = version
        self.class_mapping = class_mapping
        self.input_size = input_size
        self.nms_threshold = nms_threshold
        self.session = None
        self.input_name = ""
        self.output_names = []

    def load(self) -> bool:
        """Initialize ONNX Runtime session with CPU optimizations."""
        if not self.model_path.exists():
            logger.error("ONNX model file does not exist: %s", self.model_path)
            return False

        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )

            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            logger.info("Successfully loaded ONNX model: %s (Inputs: %s)", self.model_name, self.input_name)
            return True
        except Exception as e:
            logger.error("Failed to initialize ONNX Runtime session for %s: %s", self.model_path, e)
            self.session = None
            return False

    def _letterbox(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Resize image to input_size with aspect-ratio preserving padding."""
        target_w, target_h = self.input_size
        h, w = image.shape[:2]
        scale = min(target_w / w, target_h / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))

        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

        pad_w = target_w - nw
        pad_h = target_h - nh
        pad_left = pad_w // 2
        pad_top = pad_h // 2

        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        canvas[pad_top : pad_top + nh, pad_left : pad_left + nw] = resized
        return canvas, scale, (pad_left, pad_top)

    def predict(
        self,
        frame: np.ndarray,
        confidence_thresholds: Dict[str, float],
        unknown_min_conf: float = 0.25,
    ) -> List[Detection]:
        if self.session is None or frame is None or frame.size == 0:
            return []

        orig_h, orig_w = frame.shape[:2]
        canvas, scale, (pad_x, pad_y) = self._letterbox(frame)

        # Preprocessing: BGR -> RGB, HWC -> CHW, 0-255 -> 0.0-1.0
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)

        # Inference
        try:
            outputs = self.session.run(self.output_names, {self.input_name: blob})
        except Exception as e:
            logger.error("ONNX inference failed: %s", e)
            return []

        preds = outputs[0]
        # Normalize output shape to (num_candidates, 4 + num_classes)
        if preds.ndim == 3:
            # Shape e.g. (1, 84, 8400) -> transpose to (8400, 84)
            if preds.shape[1] < preds.shape[2]:
                preds = np.transpose(preds[0], (1, 0))
            else:
                preds = preds[0]

        num_classes = len(self.class_mapping)
        boxes_list = []
        scores_list = []
        classes_list = []

        for row in preds:
            cx, cy, bw, bh = row[0:4]
            class_scores = row[4 : 4 + num_classes]
            best_idx = int(np.argmax(class_scores))
            best_score = float(class_scores[best_idx])

            if best_score < unknown_min_conf:
                continue

            # Convert from center-based coordinates to corner-based
            x1 = cx - bw / 2.0
            y1 = cy - bh / 2.0
            x2 = cx + bw / 2.0
            y2 = cy + bh / 2.0

            # Inverse scale from letterbox back to original image
            ox1 = (x1 - pad_x) / scale
            oy1 = (y1 - pad_y) / scale
            ox2 = (x2 - pad_x) / scale
            oy2 = (y2 - pad_y) / scale

            # Clip to image boundaries
            ox1 = max(0, min(orig_w - 1, ox1))
            oy1 = max(0, min(orig_h - 1, oy1))
            ox2 = max(0, min(orig_w - 1, ox2))
            oy2 = max(0, min(orig_h - 1, oy2))

            if ox2 <= ox1 or oy2 <= oy1:
                continue

            boxes_list.append([ox1, oy1, ox2, oy2])
            scores_list.append(best_score)
            classes_list.append(best_idx)

        if not boxes_list:
            return []

        boxes_arr = np.array(boxes_list, dtype=np.float32)
        scores_arr = np.array(scores_list, dtype=np.float32)

        keep_indices = nms(boxes_arr, scores_arr, iou_threshold=self.nms_threshold)

        detections: List[Detection] = []
        for idx in keep_indices:
            cid = classes_list[idx]
            conf = float(scores_arr[idx])
            cat_name = self.class_mapping.get(cid, "general_trash")
            target_thresh = confidence_thresholds.get(cat_name, 0.45)

            x1, y1, x2, y2 = boxes_arr[idx]
            bbox = BoundingBox(
                x=int(round(x1)),
                y=int(round(y1)),
                width=int(round(x2 - x1)),
                height=int(round(y2 - y1)),
            )

            # Check Unknown semantics:
            # Candidate confidence is high enough to be a detected object, but lower than category threshold
            is_unknown = conf < target_thresh
            if is_unknown:
                display_name = "Unknown"
                effective_cat = "unknown"
            else:
                display_name = cat_name.replace("_", " ").title()
                effective_cat = cat_name

            detections.append(
                Detection(
                    class_id=cid,
                    class_name=effective_cat,
                    display_name=display_name,
                    confidence=conf,
                    bbox=bbox,
                    is_unknown=is_unknown,
                )
            )

        return detections

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "architecture": "onnx_yolo_like",
            "framework": "onnxruntime_cpu",
            "model_path": str(self.model_path),
            "input_size": list(self.input_size),
            "loaded": self.session is not None,
        }

    def unload(self) -> None:
        self.session = None
        logger.info("Unloaded ONNX model %s", self.model_name)


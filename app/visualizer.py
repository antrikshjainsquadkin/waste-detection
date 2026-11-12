"""Visual preview and OpenCV HighGUI rendering engine for Waste Detection.

Provides annotated bounding box rendering, interactive window display,
keyboard playback controls (Pause/Resume, Quit), and resolution scaling.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import cv2
import numpy as np

if TYPE_CHECKING:
    from app.detector import Detection

logger = logging.getLogger(__name__)

# Centralized presentation display label
DISPLAY_LABEL: str = "WASTE"


def format_display_label(confidence: float, track_id: Optional[int] = None) -> str:
    """Format visible detection label using the centralized DISPLAY_LABEL rule."""
    conf_str = f"{confidence:.2f}"
    if track_id is not None:
        return f"{DISPLAY_LABEL} #{track_id} ({conf_str})"
    return f"{DISPLAY_LABEL} ({conf_str})"


# Native Waste Category Color Palette (BGR)
# Organic: Green, Recyclable: Azure / Cyan-Blue, Residual: Red-Orange / Coral
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "organic": (50, 205, 50),      # Lime Green
    "recyclable": (255, 140, 0),   # Deep Sky Blue
    "residual": (30, 70, 240),     # Coral Red
}
DEFAULT_BOX_COLOR: Tuple[int, int, int] = (200, 200, 200)  # Light Gray


def get_class_color(class_name: str) -> Tuple[int, int, int]:
    """Return BGR color tuple for a given waste class name."""
    return CLASS_COLORS.get(class_name.lower(), DEFAULT_BOX_COLOR)


def render_annotated_frame(
    frame: np.ndarray,
    detections: List[Detection],
    show_hud: bool = True,
    hud_text: Optional[str] = None,
) -> np.ndarray:
    """Render bounding boxes, class labels, confidences, and optional HUD onto a frame copy.

    Does NOT modify the input frame in-place.
    """
    if frame is None or frame.size == 0:
        return frame

    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Draw each detection
    for det in detections:
        color = get_class_color(det.class_name)

        # Clamp box coordinates inside frame bounds
        x1 = max(0, min(det.bbox_x, w - 1))
        y1 = max(0, min(det.bbox_y, h - 1))
        x2 = max(0, min(det.bbox_x + det.bbox_w, w - 1))
        y2 = max(0, min(det.bbox_y + det.bbox_h, h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        # Draw bounding rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=2, lineType=cv2.LINE_AA)

        # Construct label text using centralized presentation label
        label = format_display_label(det.confidence, det.track_id)

        # Calculate label size
        font_scale = 0.5
        font_thickness = 1
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        (label_w, label_h), baseline = cv2.getTextSize(
            label, font_face, font_scale, font_thickness
        )

        # Place label banner above box, or inside top if outside bounds
        if y1 - label_h - 6 >= 0:
            bg_pt1 = (x1, y1 - label_h - 6)
            bg_pt2 = (x1 + label_w + 6, y1)
            text_org = (x1 + 3, y1 - 4)
        else:
            bg_pt1 = (x1, y1)
            bg_pt2 = (x1 + label_w + 6, y1 + label_h + 6)
            text_org = (x1 + 3, y1 + label_h + 2)

        # Draw filled solid banner and crisp white label text
        cv2.rectangle(annotated, bg_pt1, bg_pt2, color, thickness=-1)
        cv2.putText(
            annotated,
            label,
            text_org,
            font_face,
            font_scale,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    # Draw HUD controls badge if requested
    if show_hud:
        default_hud = "[SPACE] Pause/Resume  |  [Q / ESC] Quit"
        display_hud = hud_text or default_hud
        font_scale = 0.45
        font_thickness = 1
        (hw, hh), hbase = cv2.getTextSize(display_hud, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        pad = 6
        # Draw translucent black backing on top-left
        hud_bg_pt1 = (10, 10)
        hud_bg_pt2 = (10 + hw + pad * 2, 10 + hh + pad * 2)
        sub_img = annotated[hud_bg_pt1[1]:hud_bg_pt2[1], hud_bg_pt1[0]:hud_bg_pt2[0]]
        if sub_img.shape[0] > 0 and sub_img.shape[1] > 0:
            black_rect = np.zeros_like(sub_img)
            annotated[hud_bg_pt1[1]:hud_bg_pt2[1], hud_bg_pt1[0]:hud_bg_pt2[0]] = cv2.addWeighted(
                sub_img, 0.4, black_rect, 0.6, 0.0
            )
        cv2.putText(
            annotated,
            display_hud,
            (10 + pad, 10 + hh + pad - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (230, 230, 230),
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    return annotated


def render_paused_overlay(frame: np.ndarray) -> np.ndarray:
    """Render a prominent PAUSED banner onto a frame copy."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Draw semi-transparent dark banner across middle
    banner_h = 60
    y1 = max(0, (h - banner_h) // 2)
    y2 = min(h, y1 + banner_h)
    sub_img = annotated[y1:y2, 0:w]
    if sub_img.shape[0] > 0 and sub_img.shape[1] > 0:
        dark_bar = np.full_like(sub_img, 20)
        annotated[y1:y2, 0:w] = cv2.addWeighted(sub_img, 0.3, dark_bar, 0.7, 0.0)

    pause_text = "PAUSED  -  Press SPACE to Resume  |  Q to Quit"
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    (tw, th), _ = cv2.getTextSize(pause_text, font_face, font_scale, font_thickness)
    tx = max(10, (w - tw) // 2)
    ty = y1 + (banner_h + th) // 2
    cv2.putText(
        annotated,
        pause_text,
        (tx, ty),
        font_face,
        font_scale,
        (0, 255, 255),  # Yellow
        font_thickness,
        lineType=cv2.LINE_AA,
    )
    return annotated


class VisualPreviewWindow:
    """Manages an OpenCV HighGUI window with aspect-ratio scaling and keyboard controls."""

    def __init__(
        self,
        window_title: str,
        max_width: int = 1280,
        max_height: int = 720,
    ) -> None:
        self.window_title = window_title
        self.max_width = max_width
        self.max_height = max_height
        self._window_created = False

    def _ensure_window(self, frame_w: int, frame_h: int) -> None:
        """Create window once with appropriate dimensions preserving aspect ratio."""
        if not self._window_created:
            cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            # Calculate display size fitting within max_width and max_height
            scale = min(self.max_width / max(1, frame_w), self.max_height / max(1, frame_h), 1.0)
            win_w = int(frame_w * scale)
            win_h = int(frame_h * scale)
            cv2.resizeWindow(self.window_title, win_w, win_h)
            self._window_created = True

    def show_image(self, frame: np.ndarray, detections: List[Detection]) -> None:
        """Display static annotated image and wait until user closes it or presses a key."""
        if frame is None or frame.size == 0:
            return {"annotation_ms": 0.0, "imshow_ms": 0.0, "waitkey_ms": 0.0}

        t_annot_0 = time.perf_counter_ns()
        annotated = render_annotated_frame(
            frame,
            detections,
            show_hud=True,
            hud_text="Press any key or close window to continue...",
        )
        t_annot_ms = (time.perf_counter_ns() - t_annot_0) / 1_000_000

        h, w = annotated.shape[:2]
        self._ensure_window(w, h)

        t_imshow_0 = time.perf_counter_ns()
        cv2.imshow(self.window_title, annotated)
        t_imshow_ms = (time.perf_counter_ns() - t_imshow_0) / 1_000_000

        print("\n[PREVIEW] Visual preview window opened.")
        print("[PREVIEW] Press ANY KEY in the preview window (or close it) to continue...")
        t_wait_0 = time.perf_counter_ns()
        cv2.waitKey(0)
        t_wait_ms = (time.perf_counter_ns() - t_wait_0) / 1_000_000
        self.close()

        return {
            "annotation_ms": t_annot_ms,
            "imshow_ms": t_imshow_ms,
            "waitkey_ms": t_wait_ms,
        }

    def update_frame(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        wait_ms: int = 1,
    ) -> str:
        """Display video frame and process keyboard events."""
        action, _ = self.update_frame_profiled(frame, detections, wait_ms=wait_ms)
        return action

    def update_frame_profiled(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        wait_ms: int = 1,
        hud_text: Optional[str] = None,
    ) -> Tuple[str, Dict[str, float]]:
        """Display video frame and measure exact annotation, imshow, and waitKey latencies."""
        if frame is None or frame.size == 0:
            return "continue", {"annotation_ms": 0.0, "imshow_ms": 0.0, "waitkey_ms": 0.0}

        t_annot_0 = time.perf_counter_ns()
        annotated = render_annotated_frame(frame, detections, show_hud=True, hud_text=hud_text)
        t_annot_ms = (time.perf_counter_ns() - t_annot_0) / 1_000_000

        h, w = annotated.shape[:2]
        self._ensure_window(w, h)

        t_imshow_0 = time.perf_counter_ns()
        cv2.imshow(self.window_title, annotated)
        t_imshow_ms = (time.perf_counter_ns() - t_imshow_0) / 1_000_000

        t_wait_0 = time.perf_counter_ns()
        key = cv2.waitKey(wait_ms) & 0xFF
        t_wait_ms = (time.perf_counter_ns() - t_wait_0) / 1_000_000

        timings = {
            "annotation_ms": t_annot_ms,
            "imshow_ms": t_imshow_ms,
            "waitkey_ms": t_wait_ms,
        }

        # Check for quit keys: Q, q, or ESC (27)
        if key in (ord("q"), ord("Q"), 27):
            return "quit", timings

        # Check for pause key: SPACE (32)
        if key == ord(" "):
            return self._handle_pause(annotated), timings

        # Check if window was closed via the [X] button
        try:
            if cv2.getWindowProperty(self.window_title, cv2.WND_PROP_VISIBLE) < 1:
                return "quit", timings
        except Exception:
            pass

        return "continue", timings

    def _handle_pause(self, current_frame: np.ndarray) -> str:
        """Hold the display in a pause loop until SPACE is pressed again or Q/ESC to quit."""
        paused_frame = render_paused_overlay(current_frame)
        cv2.imshow(self.window_title, paused_frame)
        print("\n[PREVIEW] Playback PAUSED. Press SPACE in preview window to resume, or Q to quit.")

        while True:
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print("[PREVIEW] Quit requested during pause.")
                return "quit"
            if key == ord(" "):
                print("[PREVIEW] Playback RESUMED.")
                return "continue"
            try:
                if cv2.getWindowProperty(self.window_title, cv2.WND_PROP_VISIBLE) < 1:
                    return "quit"
            except Exception:
                pass

    def close(self) -> None:
        """Safely destroy the OpenCV window."""
        if self._window_created:
            try:
                cv2.destroyWindow(self.window_title)
            except Exception:
                pass
            self._window_created = False


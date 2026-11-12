"""Waste Detection System - Application Entry Point.

Terminal-based Computer Vision application for real-time waste classification,
object tracking, event deduplication, and session video recording using YOLO11.
"""

from __future__ import annotations

import time

_APP_START_NS: int = time.perf_counter_ns()

import argparse
import csv
import datetime
import glob
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import cv2
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import AppConfig, NATIVE_CLASSES, CLASS_EXAMPLES, sanitize_source_reference
if TYPE_CHECKING:
    from app.detector import Detection, WasteDetector
from app.recorder import VideoRecorder
from app.sources import (
    BaseSource,
    ImageSource,
    RtspSource,
    VideoFileSource,
    WebcamSource,
    resolve_image_path,
    resolve_video_path,
    resolve_stream_url,
)
from app.storage import DetectionEvent, StorageManager
from app.tracking import DetectionTracker
from app.visualizer import DISPLAY_LABEL, VisualPreviewWindow

_IMPORTS_DONE_NS: int = time.perf_counter_ns()

# Setup clean console-only logger (zero disk log files generated)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
# Lower third-party log noise
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logger = logging.getLogger("main")


class PipelineProfiler:
    """Accurate latency measurement and statistical profiling engine.

    Measures individual stages using time.perf_counter_ns(), separates warm-up,
    computes robust percentiles and standard deviations, calculates throughput FPS
    and latency-implied FPS, and isolates preview and CSV overheads.
    """

    def __init__(
        self,
        warmup_frames: int = 5,
        device_name: str = "CPU",
        primary_model_name: str = "YOLO11m",
        secondary_model_name: str = "N/A",
        source_name: str = "Unknown",
        source_resolution: Tuple[int, int] = (0, 0), # pyright: ignore[reportUndefinedVariable]
        source_fps: Optional[float] = None,
        preview_enabled: bool = True,
        primary_startup_ms: float = 0.0,
        secondary_startup_ms: float = 0.0,
    ):
        self.warmup_frames = max(0, warmup_frames)
        self.device_name = device_name
        self.primary_model_name = primary_model_name
        self.secondary_model_name = secondary_model_name
        self.source_name = source_name
        self.source_resolution = source_resolution
        self.source_fps = source_fps
        self.preview_enabled = preview_enabled
        self.primary_startup_ms = primary_startup_ms
        self.secondary_startup_ms = secondary_startup_ms

        self.total_frames_processed = 0
        self.warmup_frames_counted = 0
        self.benchmark_start_ns: Optional[int] = None
        self.benchmark_end_ns: Optional[int] = None
        self.session_wall_start_ns: int = time.perf_counter_ns()
        self.session_wall_end_ns: Optional[int] = None

        self.stage_samples: Dict[str, List[float]] = {
            "Frame Read": [],
            "Primary YOLO11m": [],
            "Secondary Verifier": [],
            "Tracking": [],
            "Filtering": [],
            "Annotation": [],
            "OpenCV Display Calls": [],
            "waitKey": [],
            "CSV Writing": [],
            "End-to-End Pipeline": [],
        }
        self.csv_positive_writes: List[float] = []

    def record_frame(
        self,
        frame_index: int,
        read_ms: float,
        primary_ms: float,
        verifier_ms: float,
        tracking_ms: float,
        filtering_ms: float,
        annotation_ms: float,
        imshow_ms: float,
        waitkey_ms: float,
        csv_write_ms: float,
        end_to_end_ms: float,
    ) -> None:
        self.total_frames_processed += 1

        if frame_index <= self.warmup_frames:
            self.warmup_frames_counted += 1
            return

        if self.benchmark_start_ns is None:
            self.benchmark_start_ns = time.perf_counter_ns()

        self.benchmark_end_ns = time.perf_counter_ns()

        self.stage_samples["Frame Read"].append(read_ms)
        self.stage_samples["Primary YOLO11m"].append(primary_ms)
        if verifier_ms is not None and verifier_ms > 0:
            self.stage_samples["Secondary Verifier"].append(verifier_ms)
        self.stage_samples["Tracking"].append(tracking_ms)
        if filtering_ms is not None and filtering_ms > 0:
            self.stage_samples["Filtering"].append(filtering_ms)
        if self.preview_enabled:
            self.stage_samples["Annotation"].append(annotation_ms)
            self.stage_samples["OpenCV Display Calls"].append(imshow_ms)
            self.stage_samples["waitKey"].append(waitkey_ms)
        self.stage_samples["CSV Writing"].append(csv_write_ms)
        if csv_write_ms > 0:
            self.csv_positive_writes.append(csv_write_ms)
        self.stage_samples["End-to-End Pipeline"].append(end_to_end_ms)

    def finalize(self) -> None:
        if self.session_wall_end_ns is None:
            self.session_wall_end_ns = time.perf_counter_ns()

    @staticmethod
    def calculate_stats(samples: List[float]) -> Dict[str, float]:
        if not samples:
            return {
                "count": 0,
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "std": 0.0,
            }
        arr = np.array(samples, dtype=np.float64)
        return {
            "count": len(samples),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        }

    def print_summary(self) -> None:
        self.finalize()
        e2e_samples = self.stage_samples["End-to-End Pipeline"]
        measured_count = len(e2e_samples)

        session_duration_sec = (self.session_wall_end_ns - self.session_wall_start_ns) / 1_000_000_000

        # Throughput FPS calculation over benchmark window
        if self.benchmark_start_ns and self.benchmark_end_ns and measured_count > 0:
            bench_duration_sec = (self.benchmark_end_ns - self.benchmark_start_ns) / 1_000_000_000
            throughput_fps = (measured_count / bench_duration_sec) if bench_duration_sec > 0 else 0.0
        elif session_duration_sec > 0:
            throughput_fps = self.total_frames_processed / session_duration_sec
        else:
            throughput_fps = 0.0

        # End-to-end statistics
        e2e_stats = self.calculate_stats(e2e_samples)
        latency_implied_fps = (1000.0 / e2e_stats["mean"]) if e2e_stats["mean"] > 0 else 0.0

        # Per-stage statistics
        stage_names = [
            "Frame Read",
            "Primary YOLO11m",
            "Secondary Verifier",
            "Tracking",
            "Filtering",
            "Annotation",
            "OpenCV Display Calls",
            "CSV Writing",
        ]

        stage_stats_dict = {name: self.calculate_stats(self.stage_samples[name]) for name in stage_names}

        # Bottleneck calculation
        active_computational_stages = {
            name: s["mean"] for name, s in stage_stats_dict.items() if s["count"] > 0
        }
        sum_stage_means = sum(active_computational_stages.values())

        if active_computational_stages:
            sorted_stages = sorted(active_computational_stages.items(), key=lambda x: x[1], reverse=True)
            slowest_stage, slowest_mean = sorted_stages[0]
            slowest_share = (slowest_mean / sum_stage_means * 100.0) if sum_stage_means > 0 else 0.0
        else:
            sorted_stages = []
            slowest_stage, slowest_mean, slowest_share = "N/A", 0.0, 0.0

        # Preview overhead
        annot_mean = stage_stats_dict["Annotation"]["mean"]
        imshow_mean = stage_stats_dict["OpenCV Display Calls"]["mean"]
        waitkey_stats = self.calculate_stats(self.stage_samples["waitKey"])
        waitkey_mean = waitkey_stats["mean"]

        res_str = f"{self.source_resolution[0]}x{self.source_resolution[1]}" if self.source_resolution != (0, 0) else "N/A"
        src_fps_str = f"{self.source_fps:.2f} FPS" if (self.source_fps and self.source_fps > 0) else "N/A"

        print("\n" + "=" * 56)
        print("             WASTE DETECTION PERFORMANCE SUMMARY")
        print("=" * 56)
        print("\nSYSTEM INFORMATION")
        print("-" * 56)
        print(f"Device                  : {self.device_name}")
        print(f"Primary Model           : {self.primary_model_name}")
        print(f"Secondary Model         : {self.secondary_model_name}")
        print(f"Input Source            : {self.source_name}")
        print(f"Input Resolution        : {res_str}")
        print(f"Frames Processed        : {self.total_frames_processed}")
        print(f"Frames Measured         : {measured_count}")
        print(f"Warm-up Frames          : {self.warmup_frames_counted}")
        print(f"Session Duration        : {session_duration_sec:.2f}s")

        print("\nMODEL INITIALIZATION")
        print("-" * 56)
        print(f"Primary Model Load      : {self.primary_startup_ms:.2f} ms")
        print(f"Secondary Model Load    : {self.secondary_startup_ms:.2f} ms")
        total_model_startup = self.primary_startup_ms + self.secondary_startup_ms
        print(f"Total Model Startup     : {total_model_startup:.2f} ms")

        print("\nFRAME LATENCY BREAKDOWN")
        print("-" * 56)
        print(f"{'Stage':<24} {'Mean':<10} {'P95':<10} {'Max':<10}")
        print("-" * 56)
        for name in stage_names:
            st = stage_stats_dict[name]
            if st["count"] > 0:
                print(f"{name:<24} {st['mean']:>7.2f} ms {st['p95']:>7.2f} ms {st['max']:>7.2f} ms")
            else:
                print(f"{name:<24} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
        print("-" * 56)

        print("\nEND-TO-END PERFORMANCE")
        print("-" * 56)
        print(f"Mean Frame Latency      : {e2e_stats['mean']:.2f} ms")
        print(f"Median Frame Latency    : {e2e_stats['median']:.2f} ms")
        print(f"P95 Frame Latency       : {e2e_stats['p95']:.2f} ms")
        print(f"P99 Frame Latency       : {e2e_stats['p99']:.2f} ms")
        print(f"Minimum Frame Latency   : {e2e_stats['min']:.2f} ms")
        print(f"Maximum Frame Latency   : {e2e_stats['max']:.2f} ms")
        print(f"Standard Deviation      : {e2e_stats['std']:.2f} ms")
        print("")
        print(f"Measured Throughput     : {throughput_fps:.2f} FPS")
        print(f"Latency-Implied FPS     : {latency_implied_fps:.2f} FPS")
        print(f"Source FPS              : {src_fps_str}")

        print("\nBOTTLENECK ANALYSIS")
        print("-" * 56)
        print(f"Slowest Processing Stage: {slowest_stage}")
        print(f"Stage Mean Latency      : {slowest_mean:.2f} ms")
        print(f"Stage Share             : {slowest_share:.1f} %")
        if len(sorted_stages) > 1:
            print("Top Bottleneck Contributors:")
            for rank, (stg, m_lat) in enumerate(sorted_stages[:3], 1):
                sh = (m_lat / sum_stage_means * 100.0) if sum_stage_means > 0 else 0.0
                print(f"  {rank}. {stg:<22} : {m_lat:>6.2f} ms ({sh:>4.1f}%)")

        print("\nPREVIEW OVERHEAD")
        print("-" * 56)
        prev_str = "Yes" if self.preview_enabled else "No"
        print(f"Preview Enabled         : {prev_str}")
        print(f"Annotation Mean         : {annot_mean:.2f} ms" if self.preview_enabled else "Annotation Mean         : N/A")
        print(f"Display Call Mean       : {imshow_mean:.2f} ms" if self.preview_enabled else "Display Call Mean       : N/A")
        print(f"waitKey Mean            : {waitkey_mean:.2f} ms" if self.preview_enabled else "waitKey Mean            : N/A")
        if self.preview_enabled and e2e_stats["mean"] > 0:
            prev_sum = annot_mean + imshow_mean + waitkey_mean
            prev_share = (prev_sum / e2e_stats["mean"]) * 100.0
            print(f"Total Preview Overhead  : {prev_sum:.2f} ms ({prev_share:.1f}% of frame latency)")

        if self.csv_positive_writes:
            write_stats = self.calculate_stats(self.csv_positive_writes)
            print("\nCSV STORAGE WRITING")
            print("-" * 56)
            print(f"Frames with Event Writes: {write_stats['count']} / {measured_count}")
            print(f"Write Latency (Mean)    : {write_stats['mean']:.2f} ms (Max: {write_stats['max']:.2f} ms)")

        print("=" * 56)
        print("                    SESSION COMPLETE")
        print("=" * 56 + "\n")


def print_banner(detector: WasteDetector) -> None:
    """Print clean terminal header with verified model details."""
    print("=" * 72)
    print("          WASTE DETECTION SYSTEM           ")
    print("=" * 72)
    print(f" Python Interpreter: {sys.version.split()[0]} ({sys.executable})")
    print(f" Deep Learning Engine: PyTorch on {detector.device.upper()}")
    print(f" Pretrained Model: {detector.model_name}")
    print(f" Checkpoint SHA-256: {detector.sha256[:16]}...{detector.sha256[-8:]}")
    verifier_status = "ACTIVE" if detector.verifier.is_available else "INACTIVE"
    if detector.verifier.is_available and detector.verifier.weights_path:
        verifier_details = f"{detector.verifier.weights_path.name} ({len(detector.verifier.class_mapping)} classes)"
    else:
        verifier_details = "None loaded (verification disabled)"
    print(f" Non-Waste Verifier: {verifier_status} - {verifier_details}")
    print(" Supported Native Classes:")
    for cid, cname in sorted(detector.class_mapping.items()):
        examples = CLASS_EXAMPLES.get(cname, "")
        print(f"   [{cid}] {cname.upper():<12} -> {examples}")
    print("=" * 72)


class AsyncDetectorLoader:
    """Loads primary and secondary models and warms them up in a background thread."""

    def __init__(self, detector_factory: Callable[[], "WasteDetector"]) -> None:
        self._factory = detector_factory
        self.ready_event = threading.Event()
        self.detector: Optional["WasteDetector"] = None
        self.error: Optional[Exception] = None
        self.primary_load_ms: float = 0.0
        self.secondary_load_ms: float = 0.0
        self.warmup_ms: float = 0.0
        self.models_ready_ns: Optional[int] = None
        self._thread = threading.Thread(
            target=self._run,
            name="detector-loader",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        try:
            print("[INFO] Loading waste detector & secondary verifier...")
            det = self._factory()
            self.primary_load_ms = float(getattr(det, "primary_load_time_ms", 0.0))
            self.secondary_load_ms = float(getattr(det, "secondary_load_time_ms", 0.0))
            self.warmup_ms = 0.0

            self.detector = det
            self.models_ready_ns = time.perf_counter_ns()
        except Exception as exc:
            self.error = exc
        finally:
            self.ready_event.set()


def print_startup_timings(timings: Dict[str, Any]) -> None:
    """Print nanosecond-measured startup milestone breakdown."""
    print("\n" + "=" * 56)
    print("               STARTUP TIMING BREAKDOWN")
    print("=" * 56)
    rows = [
        ("Python/App Imports", timings.get("imports_ms")),
        ("Camera Open", timings.get("cam_open_ms")),
        ("First Frame Read", timings.get("first_frame_ms")),
        ("Preview Window Create", timings.get("preview_create_ms")),
        ("First Preview Render", timings.get("first_preview_ms")),
        ("Time to First Live Preview", timings.get("time_to_first_preview_ms")),
        ("Primary Model Load", timings.get("primary_load_ms")),
        ("Secondary Verifier Load", timings.get("secondary_load_ms")),
        ("Model Warm-up", timings.get("warmup_ms")),
        ("Time to Detection Active", timings.get("time_to_detection_active_ms")),
        ("First Inference", timings.get("first_inference_ms")),
        ("First Verified Detection", timings.get("time_to_first_verified_detection_ms")),
    ]
    for label, val in rows:
        if val is not None:
            print(f" {label:<30}: {val:8.2f} ms")
        else:
            print(f" {label:<30}: N/A")
    if timings.get("camera_backend"):
        print(f" {'Camera Backend':<30}: {timings['camera_backend']}")
    print("=" * 56 + "\n")


def run_session(
    source: BaseSource,
    detector: Optional["WasteDetector"],
    config: AppConfig,
    record_live: bool = False,
    show_preview: bool = True,
    save_to_disk: bool = False,
    detector_factory: Optional[Callable[[], "WasteDetector"]] = None,
    startup_timings: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Execute detection session over the specified input source."""
    timings: Dict[str, Any] = startup_timings if startup_timings is not None else {}
    timings.setdefault("imports_ms", (_IMPORTS_DONE_NS - _APP_START_NS) / 1_000_000)

    if source.source_type == "webcam":
        print("[INFO] Opening webcam...")

    t_cam_open_0 = time.perf_counter_ns()
    if not source.open():
        print(f"\n[ERROR] Unable to open source: {source.reference_string}")
        return None
    t_cam_open_end = time.perf_counter_ns()
    timings["cam_open_ms"] = getattr(source, "open_ms", (t_cam_open_end - t_cam_open_0) / 1_000_000)
    if hasattr(source, "backend_name"):
        timings["camera_backend"] = source.backend_name

    if source.source_type == "webcam":
        print(f"[INFO] Webcam connected ({timings.get('camera_backend', 'UNKNOWN')}).")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"session_{timestamp}"

    def on_new_event(event: DetectionEvent) -> None:
        print(
            f" [DETECTION SAVED] {DISPLAY_LABEL} event_id={event.event_id} | "
            f"Conf: {event.confidence:.2f} | Track: #{event.track_id}"
        )

    storage: Optional[StorageManager] = None
    tracker: Optional[DetectionTracker] = None

    def _init_storage_and_tracker(active_det: "WasteDetector") -> None:
        nonlocal storage, tracker
        storage = StorageManager(
            csv_path=config.csv_path,
            session_id=session_id,
            source_type=source.source_type,
            source_reference=source.reference_string,
            model_name=active_det.model_name,
            checkpoint_filename=Path(active_det.weights_path).name,
            confidence_threshold=active_det.conf,
            image_size=active_det.imgsz,
            model_classes=active_det.class_mapping,
            model_sha256=active_det.sha256,
        )
        tracker = DetectionTracker(
            storage_manager=storage,
            iou_threshold=config.iou_threshold,
            lost_track_timeout=config.lost_track_timeout_seconds,
            on_new_event=on_new_event,
        )

    # Initialize visual preview window if enabled
    preview_win: Optional[VisualPreviewWindow] = None
    if show_preview:
        if source.source_type == "image":
            win_title = "Waste Detection - Image Preview"
        elif source.source_type == "webcam":
            win_title = "Waste Detection - Webcam Preview"
        elif source.source_type == "rtsp":
            win_title = "Waste Detection - RTSP Stream Preview"
        else:
            win_title = "Waste Detection - Video Preview"
        t_win_0 = time.perf_counter_ns()
        preview_win = VisualPreviewWindow(window_title=win_title)
        timings["preview_create_ms"] = (time.perf_counter_ns() - t_win_0) / 1_000_000

    # For webcam staged startup, read & display the first live frame immediately
    # before starting background model loading so there is zero contention.
    prefetched_frame: Optional[np.ndarray] = None
    prefetched_read_ms: float = 0.0
    if source.source_type == "webcam":
        t_f1_0 = time.perf_counter_ns()
        ret0, frame0 = source.read_frame()
        t_f1_end = time.perf_counter_ns()
        if ret0 and frame0 is not None:
            timings["first_frame_ms"] = (t_f1_end - t_f1_0) / 1_000_000
            if preview_win is not None:
                t_prev_0 = time.perf_counter_ns()
                hud = "Initializing models..." if detector is None else None
                action, _ = preview_win.update_frame_profiled(frame0, [], wait_ms=1, hud_text=hud)
                t_prev_end = time.perf_counter_ns()
                timings["first_preview_ms"] = (t_prev_end - t_prev_0) / 1_000_000
                timings["time_to_first_preview_ms"] = (t_prev_end - _APP_START_NS) / 1_000_000
                print("[INFO] Live preview ready.")
                if action == "quit":
                    preview_win.close()
                    source.close()
                    print_startup_timings(timings)
                    return None
            if detector is not None:
                # Detector already ready: pass frame0 into main loop for immediate inference
                prefetched_frame = frame0
                prefetched_read_ms = timings["first_frame_ms"]
        else:
            err_msg = getattr(source, "last_error", "") or f"Failed to read first frame from {source.reference_string}."
            print(f"\n[ERROR] {err_msg}")
            if preview_win is not None:
                preview_win.close()
            source.close()
            print_startup_timings(timings)
            return None

    loader: Optional[AsyncDetectorLoader] = None
    if detector is None:
        if detector_factory is None:
            raise ValueError("Either detector or detector_factory must be provided to run_session.")
        loader = AsyncDetectorLoader(detector_factory)
        loader.start()
    else:
        if hasattr(detector, "reset_tracking"):
            detector.reset_tracking()
        _init_storage_and_tracker(detector)
        timings["primary_load_ms"] = getattr(detector, "primary_load_time_ms", 0.0)
        timings["secondary_load_ms"] = getattr(detector, "secondary_load_time_ms", 0.0)
        timings.setdefault("warmup_ms", 0.0)
        timings["time_to_detection_active_ms"] = (time.perf_counter_ns() - _APP_START_NS) / 1_000_000

    print(f"\n[INFO] Starting processing for: {source.reference_string}")
    if show_preview:
        print("[INFO] On-screen preview enabled. [SPACE] Pause | [Q / ESC] Quit preview.")
    print("[INFO] Press Ctrl+C at any time to finish session safely.\n")

    # Extract source resolution & FPS if available
    src_res = (0, 0)
    src_fps = None
    if hasattr(source, "cap") and source.cap is not None:
        try:
            w = int(source.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(source.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            src_res = (w, h)
            fps_val = source.cap.get(cv2.CAP_PROP_FPS)
            if fps_val and fps_val > 0:
                src_fps = float(fps_val)
        except Exception:
            pass

    warmup_n = 0 if source.source_type == "image" else 5
    sec_name = (
        detector.verifier.weights_path.name
        if (detector is not None and detector.verifier.is_available and detector.verifier.weights_path)
        else "N/A"
    )

    profiler = PipelineProfiler(
        warmup_frames=warmup_n,
        device_name=detector.device.upper() if detector is not None else "PENDING",
        primary_model_name="YOLO11m",
        secondary_model_name=sec_name,
        source_name=f"{source.source_type} ({source.reference_string})",
        source_resolution=src_res,
        source_fps=src_fps,
        preview_enabled=show_preview,
        primary_startup_ms=getattr(detector, "primary_load_time_ms", 0.0) if detector is not None else 0.0,
        secondary_startup_ms=getattr(detector, "secondary_load_time_ms", 0.0) if detector is not None else 0.0,
    )

    frame_count = 0
    total_inference_time = 0.0
    total_rejected = 0
    total_ambiguous = 0
    start_monotonic = time.monotonic()
    last_status_time = start_monotonic
    status_interval = 2.0  # seconds
    first_infer_recorded = False
    first_verified_recorded = False
    summary: Optional[Dict[str, Any]] = None

    try:
        while True:
            t_frame_start_ns = time.perf_counter_ns()

            # Stage 1: Frame Read
            if prefetched_frame is not None:
                ret, frame = True, prefetched_frame
                t_read_ms = prefetched_read_ms
                prefetched_frame = None
            else:
                t_read_0 = time.perf_counter_ns()
                ret, frame = source.read_frame()
                t_read_ms = (time.perf_counter_ns() - t_read_0) / 1_000_000
                if ret and frame is not None and "first_frame_ms" not in timings:
                    timings["first_frame_ms"] = t_read_ms

            if not ret or frame is None:
                if getattr(source, "timed_out", False):
                    err_msg = getattr(source, "last_error", "Camera frame read timed out.")
                    print(f"\n[ERROR] {err_msg}")
                    if storage is not None:
                        storage.warnings.append(err_msg)
                # Video file, camera timeout, or image finished
                break

            frame_count += 1
            if profiler.source_resolution == (0, 0) and frame is not None:
                fh, fw = frame.shape[:2]
                profiler.source_resolution = (fw, fh)

            # Check background model loader if still initializing
            if detector is None and loader is not None:
                if loader.ready_event.is_set():
                    if loader.error is not None:
                        raise RuntimeError(f"Failed to initialize detector: {loader.error}") from loader.error
                    detector = loader.detector
                    if hasattr(detector, "reset_tracking"):
                        detector.reset_tracking()
                    _init_storage_and_tracker(detector)
                    timings["primary_load_ms"] = loader.primary_load_ms
                    timings["secondary_load_ms"] = loader.secondary_load_ms
                    timings["warmup_ms"] = loader.warmup_ms
                    timings["time_to_detection_active_ms"] = (time.perf_counter_ns() - _APP_START_NS) / 1_000_000
                    profiler.device_name = detector.device.upper()
                    profiler.secondary_model_name = (
                        detector.verifier.weights_path.name
                        if (detector.verifier.is_available and detector.verifier.weights_path)
                        else "N/A"
                    )
                    profiler.primary_startup_ms = loader.primary_load_ms
                    profiler.secondary_startup_ms = loader.secondary_load_ms
                    print("[INFO] Detection active.")
                else:
                    # Models not ready yet: show live raw preview without detections or CSV writes.
                    # Use wait_ms=30 (~30 FPS cap) so the background loader gets CPU headroom.
                    if preview_win:
                        action, _ = preview_win.update_frame_profiled(
                            frame, [], wait_ms=30, hud_text="Initializing models..."
                        )
                        if action == "quit":
                            print("\n[INFO] Playback quit requested by user.")
                            break
                    else:
                        time.sleep(0.02)
                    continue

            assert detector is not None and storage is not None and tracker is not None

            # Reset per-frame CSV write tracking and run bounded retry for any pending writes
            storage.last_write_duration_ms = 0.0
            storage.retry_pending_events()

            # Stage 2: Primary YOLO11m Inference
            t_prim_0 = time.perf_counter_ns()
            if source.source_type == "image":
                raw_candidates = detector.predict(frame)
            else:
                raw_candidates = detector.track(frame)
            t_prim_ms = (time.perf_counter_ns() - t_prim_0) / 1_000_000

            # Stage 3 & 4: Secondary Verifier Inference & Spatial Overlap Filtering
            accepted_detections, v_results, v_counts = detector.verify_candidates(frame, raw_candidates)
            t_sec_infer_ms = getattr(detector.verifier, "last_inference_ms", 0.0)
            t_filtering_ms = getattr(detector.verifier, "last_filtering_ms", 0.0)

            if not first_infer_recorded:
                first_infer_recorded = True
                timings["first_inference_ms"] = t_prim_ms + t_sec_infer_ms + t_filtering_ms
            if not first_verified_recorded and accepted_detections:
                first_verified_recorded = True
                timings["time_to_first_verified_detection_ms"] = (time.perf_counter_ns() - _APP_START_NS) / 1_000_000

            total_rejected += v_counts["rejected"]
            total_ambiguous += v_counts["ambiguous"]

            # Stage 5: In-Memory Tracking & Deduplication
            t_track_0 = time.perf_counter_ns()
            tracker.process_frame(accepted_detections, frame, frame_count)
            t_track_total_ms = (time.perf_counter_ns() - t_track_0) / 1_000_000

            # Stage 8: CSV Writing (isolated from tracking)
            csv_write_ms = storage.last_write_duration_ms
            t_track_ms = max(0.0, t_track_total_ms - csv_write_ms)

            # Stage 6 & 7: Annotation & Display (Preview)
            t_annot_ms = 0.0
            t_imshow_ms = 0.0
            t_waitkey_ms = 0.0

            if preview_win:
                if source.source_type == "image":
                    t_prev_0 = time.perf_counter_ns()
                    p_timings = preview_win.show_image(frame, accepted_detections)
                    t_prev_end = time.perf_counter_ns()
                    if "first_preview_ms" not in timings:
                        timings["first_preview_ms"] = p_timings.get("annotation_ms", 0.0) + p_timings.get("imshow_ms", 0.0)
                        timings["time_to_first_preview_ms"] = (t_prev_end - _APP_START_NS) / 1_000_000
                    t_annot_ms = p_timings.get("annotation_ms", 0.0)
                    t_imshow_ms = p_timings.get("imshow_ms", 0.0)
                    t_waitkey_ms = p_timings.get("waitkey_ms", 0.0)

                    t_frame_end_ns = time.perf_counter_ns()
                    e2e_ms = (t_frame_end_ns - t_frame_start_ns) / 1_000_000
                    profiler.record_frame(
                        frame_index=frame_count,
                        read_ms=t_read_ms,
                        primary_ms=t_prim_ms,
                        verifier_ms=t_sec_infer_ms,
                        tracking_ms=t_track_ms,
                        filtering_ms=t_filtering_ms,
                        annotation_ms=t_annot_ms,
                        imshow_ms=t_imshow_ms,
                        waitkey_ms=t_waitkey_ms,
                        csv_write_ms=csv_write_ms,
                        end_to_end_ms=e2e_ms,
                    )
                    break
                else:
                    t_prev_0 = time.perf_counter_ns()
                    action, p_timings = preview_win.update_frame_profiled(frame, accepted_detections, wait_ms=1)
                    t_prev_end = time.perf_counter_ns()
                    if "first_preview_ms" not in timings:
                        timings["first_preview_ms"] = (t_prev_end - t_prev_0) / 1_000_000
                        timings["time_to_first_preview_ms"] = (t_prev_end - _APP_START_NS) / 1_000_000
                    t_annot_ms = p_timings.get("annotation_ms", 0.0)
                    t_imshow_ms = p_timings.get("imshow_ms", 0.0)
                    t_waitkey_ms = p_timings.get("waitkey_ms", 0.0)
                    if action == "quit":
                        print("\n[INFO] Playback quit requested by user.")
                        t_frame_end_ns = time.perf_counter_ns()
                        e2e_ms = (t_frame_end_ns - t_frame_start_ns) / 1_000_000
                        profiler.record_frame(
                            frame_index=frame_count,
                            read_ms=t_read_ms,
                            primary_ms=t_prim_ms,
                            verifier_ms=t_sec_infer_ms,
                            tracking_ms=t_track_ms,
                            filtering_ms=t_filtering_ms,
                            annotation_ms=t_annot_ms,
                            imshow_ms=t_imshow_ms,
                            waitkey_ms=t_waitkey_ms,
                            csv_write_ms=csv_write_ms,
                            end_to_end_ms=e2e_ms,
                        )
                        break

            t_frame_end_ns = time.perf_counter_ns()
            e2e_ms = (t_frame_end_ns - t_frame_start_ns) / 1_000_000
            total_inference_time += (t_prim_ms + t_sec_infer_ms) / 1000.0

            profiler.record_frame(
                frame_index=frame_count,
                read_ms=t_read_ms,
                primary_ms=t_prim_ms,
                verifier_ms=t_sec_infer_ms,
                tracking_ms=t_track_ms,
                filtering_ms=t_filtering_ms,
                annotation_ms=t_annot_ms,
                imshow_ms=t_imshow_ms,
                waitkey_ms=t_waitkey_ms,
                csv_write_ms=csv_write_ms,
                end_to_end_ms=e2e_ms,
            )

            # Print terminal heartbeat
            now = time.monotonic()
            if now - last_status_time >= status_interval:
                elapsed = now - start_monotonic
                current_fps = frame_count / elapsed if elapsed > 0 else 0.0
                avg_latency_ms = (total_inference_time / frame_count * 1000) if frame_count > 0 else 0.0
                filter_str = f" | Filtered: R={total_rejected} A={total_ambiguous}" if (total_rejected + total_ambiguous > 0) else ""
                print(
                    f"\r [MONITOR] Frames: {frame_count:<5} | FPS: {current_fps:4.1f} | "
                    f"Latency: {avg_latency_ms:5.1f}ms | Events: {storage.total_events:<3}{filter_str}",
                    end="",
                    flush=True,
                )
                last_status_time = now

    except KeyboardInterrupt:
        print("\n\n[INFO] Session interrupted by user (Ctrl+C). Finalizing performance report...")
    except Exception as e:
        logger.error("Unexpected error in processing loop: %s", e, exc_info=True)
        if storage is not None:
            storage.warnings.append(str(e))
        else:
            print(f"\n[CRITICAL] {e}")
    finally:
        total_duration = time.monotonic() - start_monotonic
        avg_fps = frame_count / total_duration if total_duration > 0 else 0.0

        if preview_win:
            preview_win.close()

        source.close()

        if detector is not None and hasattr(detector, "reset_tracking"):
            detector.reset_tracking()

        if storage is not None:
            summary = storage.finalize_session(
                duration_seconds=total_duration,
                average_fps=avg_fps,
            )
            summary["startup_timings"] = timings
            print_session_summary(summary, frame_count, total_inference_time, total_rejected, total_ambiguous)
            profiler.print_summary()
        print_startup_timings(timings)

    return summary


def print_session_summary(
    summary: Dict[str, Any],
    frame_count: int,
    total_inference_time: float,
    total_rejected: int = 0,
    total_ambiguous: int = 0,
) -> None:
    """Print readable session summary in the terminal."""
    print("\n" + "=" * 72)
    print("                     SESSION SUMMARY REPORT                     ")
    print("=" * 72)
    print(f" Session ID:           {summary['session_id']}")
    print(f" Source:               {summary['source_type']} ({summary['source_reference']})")
    print(f" Duration:             {summary['duration_seconds']:.1f} seconds")
    print(f" Total Frames:         {frame_count}")
    print(f" Average FPS:          {summary['average_fps']:.1f} FPS")
    if frame_count > 0:
        avg_latency = (total_inference_time / frame_count) * 1000
        print(f" Average Latency:      {avg_latency:.1f} ms/frame")
    print(f" Total Unique Events:  {summary['total_events']}")
    if summary.get("pending_events", 0) > 0:
        print(
            f" Pending/Unsaved:      {summary['pending_events']} "
            f"(Reason: {summary.get('last_write_error') or 'CSV write failed'})"
        )
    if total_rejected > 0 or total_ambiguous > 0:
        print(f" Verifier Filtered:    Rejected: {total_rejected} | Ambiguous: {total_ambiguous}")
    print(" Category Counts:")
    for cat, cnt in summary.get("category_counts", {}).items():
        print(f"   - {cat.upper():<12}: {cnt}")
    print(f" Persistent CSV:       {summary.get('csv_path', 'waste_detections.csv')}")
    if summary.get("runtime_errors_or_warnings"):
        print(" Warnings/Errors:")
        for w in summary["runtime_errors_or_warnings"]:
            print(f"   ! {w}")
    print("=" * 72 + "\n")


def show_recent_history(csv_path: Path) -> None:
    """Display recent detection events directly from the persistent CSV file."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        print("\n[INFO] No previous detection records found in waste_detections.csv.")
        return

    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
    except Exception as e:
        print(f"[ERROR] Could not read CSV: {e}")
        return

    if not reader:
        print("\n[INFO] waste_detections.csv has no rows.")
        return

    print("\n" + "-" * 72)
    print("                 RECENT DETECTION RECORDS (FROM CSV)            ")
    print("-" * 72)
    for row in reader[-10:]:
        print(
            f" [{row.get('detected_at', '')[:19]}] {row.get('class_name', '').upper():<10} | "
            f"Conf: {float(row.get('confidence', 0)):.2f} | Track: #{row.get('track_id', '-')} | "
            f"Source: {row.get('source_name', '')} ({row.get('source_type', '')})"
        )
    print(f"\n Total stored records: {len(reader)} in {csv_path.name}")
    print("-" * 72 + "\n")


def interactive_menu(
    detector: Optional["WasteDetector"],
    config: AppConfig,
    detector_factory: Optional[Callable[[], "WasteDetector"]] = None,
) -> None:
    """Provide a simple, responsive terminal interactive menu."""
    def _cached_factory() -> "WasteDetector":
        nonlocal detector
        if detector is None:
            if detector_factory is None:
                raise RuntimeError("No detector or detector_factory available.")
            detector = detector_factory()
        return detector

    def _ensure_detector() -> "WasteDetector":
        had_detector = detector is not None
        det = _cached_factory()
        if not had_detector:
            print_banner(det)
        return det

    while True:
        print("\nChoose an action:")
        print(" 1. Start webcam detection")
        print(" 2. Start RTSP/IP camera detection")
        print(" 3. Process a video file")
        print(" 4. Process an image")
        print(" 5. View recent detection history")
        print(" 6. View the latest session summary")
        print(" 7. Exit")

        choice = input("\nEnter option (1-7): ").strip()

        if choice == "1":
            cam_idx_str = input(f"Enter camera index [{config.camera_index}]: ").strip()
            cam_idx = int(cam_idx_str) if cam_idx_str.isdigit() else config.camera_index
            source = WebcamSource(
                camera_index=cam_idx,
                read_timeout_sec=config.camera_read_timeout_seconds,
            )
            run_session(
                source,
                detector,
                config,
                detector_factory=_cached_factory if detector_factory is not None else None,
            )

        elif choice == "2":
            raw_url = config.rtsp_url
            if not raw_url:
                raw_url = input("Enter RTSP or IP Camera URL (e.g. https://192.168.1.11:8080/ or rtsp://...): ").strip()
            if not raw_url:
                print("[ERROR] No stream URL provided.")
                continue
            resolved_url, msg = resolve_stream_url(raw_url)
            if msg:
                print(f"[INFO] {msg}")
            source = RtspSource(
                rtsp_url=resolved_url,
                max_reconnect_attempts=config.max_reconnect_attempts,
                initial_delay_sec=config.reconnect_initial_delay_sec,
                max_delay_sec=config.reconnect_max_delay_sec,
            )
            run_session(source, _ensure_detector(), config)

        elif choice == "3":
            raw_path = input("Enter path to prerecorded video: ").strip()
            if not raw_path:
                print("[ERROR] Path cannot be empty.")
                continue
            resolved, msg = resolve_video_path(raw_path)
            if not resolved:
                print(f"[ERROR] {msg}")
                continue
            if msg:
                print(f"[INFO] {msg}")
            source = VideoFileSource(file_path=resolved)
            run_session(source, _ensure_detector(), config)

        elif choice == "4":
            raw_path = input("Enter path to image: ").strip()
            if not raw_path:
                print("[ERROR] Path cannot be empty.")
                continue
            resolved, msg = resolve_image_path(raw_path)
            if not resolved:
                print(f"[ERROR] {msg}")
                continue
            if msg:
                print(f"[INFO] {msg}")
            source = ImageSource(file_path=resolved)
            run_session(source, _ensure_detector(), config)

        elif choice == "5":
            show_recent_history(config.csv_path)

        elif choice == "6":
            if not config.csv_path.exists() or config.csv_path.stat().st_size == 0:
                print("\n[INFO] No detection records found in waste_detections.csv.")
                continue
            try:
                with open(config.csv_path, "r", newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                if not rows:
                    print("\n[INFO] waste_detections.csv has no rows.")
                    continue
                latest_session_id = rows[-1].get("session_id")
                session_rows = [r for r in rows if r.get("session_id") == latest_session_id]
                counts = {"organic": 0, "recyclable": 0, "residual": 0}
                for r in session_rows:
                    cname = r.get("class_name", "").lower()
                    counts[cname] = counts.get(cname, 0) + 1
                print("\n" + "=" * 72)
                print("               LATEST SESSION SUMMARY (FROM CSV)                ")
                print("=" * 72)
                print(f" Session ID:    {latest_session_id}")
                print(f" Source:        {session_rows[0].get('source_name')} ({session_rows[0].get('source_type')})")
                print(f" Total Events:  {len(session_rows)}")
                for cat, cnt in sorted(counts.items()):
                    print(f"   - {cat.upper():<12}: {cnt}")
                print(f" CSV Storage:   {config.csv_path}")
                print("=" * 72 + "\n")
            except Exception as e:
                print(f"[ERROR] Failed reading latest session from CSV: {e}")

        elif choice == "7":
            print("\nExit !\n")
            break
        else:
            print("[INVALID] Please select a number between 1 and 7.")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for repeatable execution."""
    parser = argparse.ArgumentParser(
        description="Local CPU-first Terminal Waste Detection System using YOLO11."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Input source: 'webcam', 'rtsp', or path to video file.",
    )
    parser.add_argument(
        "--source-image",
        type=str,
        default=None,
        help="Path to an input image file for inference.",
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="Camera device index for webcam source (default: 0).",
    )
    parser.add_argument(
        "--rtsp-url",
        type=str,
        default="",
        help="RTSP stream URL (can also use WASTE_RTSP_URL env var).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Starting confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image inference size (default: 640).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Computation device ('cpu' prioritized, or 'cuda').",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        default=False,
        help="Disable visual on-screen OpenCV preview window (run headless).",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        default=False,
        help="Enable local video recording for live camera feeds (default: disabled).",
    )
    parser.add_argument(
        "--save-session",
        action="store_true",
        default=False,
        help="Save session results, CSV, and evidence to disk (default: disabled, 0 files created).",
    )
    parser.add_argument(
        "--no-verifier",
        action="store_true",
        default=False,
        help="Disable secondary non-waste verifier (human/animal filtering).",
    )
    parser.add_argument(
        "--verifier-weights",
        type=str,
        default=None,
        help="Path to secondary non-waste verifier weights (default: autodetected).",
    )
    return parser.parse_args()


def _create_detector(args: argparse.Namespace, config: AppConfig) -> "WasteDetector":
    """Lazily import and initialize WasteDetector with configured primary & verifier models."""
    from app.detector import WasteDetector

    verifier_path = Path(args.verifier_weights) if args.verifier_weights else config.verifier_weights_path
    return WasteDetector(
        confidence_threshold=args.conf,
        image_size=args.imgsz,
        device=args.device,
        verifier_weights_path=verifier_path,
        verifier_confidence=config.verifier_confidence,
        verifier_enabled=(not args.no_verifier) and config.verifier_enabled,
    )


def main() -> None:
    """Primary application entry point."""
    args = parse_args()
    config = AppConfig()
    show_preview = not args.no_preview
    detector_factory = lambda: _create_detector(args, config)

    # CLI Direct Source Modes
    if args.source_image:
        try:
            detector = detector_factory()
        except Exception as e:
            print(f"\n[CRITICAL] Failed to initialize detector: {e}")
            logger.critical("Model initialization failed: %s", e, exc_info=True)
            sys.exit(1)
        print_banner(detector)
        resolved, msg = resolve_image_path(args.source_image)
        if not resolved:
            print(f"[ERROR] {msg}")
            sys.exit(1)
        if msg:
            print(f"[INFO] {msg}")
        source = ImageSource(file_path=resolved)
        run_session(source, detector, config, show_preview=show_preview)

    elif args.source:
        src = args.source.strip()
        if src.lower() == "webcam":
            source = WebcamSource(
                camera_index=args.camera_id,
                read_timeout_sec=config.camera_read_timeout_seconds,
            )
            run_session(
                source,
                detector=None,
                config=config,
                show_preview=show_preview,
                detector_factory=detector_factory,
            )
            return

        try:
            detector = detector_factory()
        except Exception as e:
            print(f"\n[CRITICAL] Failed to initialize detector: {e}")
            logger.critical("Model initialization failed: %s", e, exc_info=True)
            sys.exit(1)
        print_banner(detector)

        if (
            src.lower() == "rtsp"
            or src.lower().startswith("rtsp://")
            or src.lower().startswith("http://")
            or src.lower().startswith("https://")
        ):
            raw_url = src if src.lower().startswith(("rtsp://", "http://", "https://")) else (args.rtsp_url or config.rtsp_url)
            if not raw_url:
                print("[ERROR] Network camera requested but no URL provided.")
                sys.exit(1)
            resolved_url, msg = resolve_stream_url(raw_url)
            if msg:
                print(f"[INFO] {msg}")
            source = RtspSource(
                rtsp_url=resolved_url,
                max_reconnect_attempts=config.max_reconnect_attempts,
                initial_delay_sec=config.reconnect_initial_delay_sec,
                max_delay_sec=config.reconnect_max_delay_sec,
            )
            run_session(source, detector, config, show_preview=show_preview)
        else:
            resolved, msg = resolve_video_path(src)
            if not resolved:
                print(f"[ERROR] {msg}")
                sys.exit(1)
            if msg:
                print(f"[INFO] {msg}")
            source = VideoFileSource(file_path=resolved)
            run_session(source, detector, config, show_preview=show_preview)

    else:
        # No arguments provided -> launch interactive terminal menu immediately
        interactive_menu(detector=None, config=config, detector_factory=detector_factory)


if __name__ == "__main__":
    main()

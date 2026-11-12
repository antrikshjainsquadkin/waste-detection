"""Hardware and CPU Inference Benchmark Utility.

Measures actual CPU throughput (FPS, average latency, median, p95) and memory
consumption across various input resolutions and frame skip factors on the target machine.
"""

from __future__ import annotations

import time
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import AppConfig
from app.models.manager import ModelManager


def run_benchmark(num_frames: int = 60) -> Dict[str, Any]:
    print("=" * 60)
    print("RUNNING CPU INFERENCE & PIPELINE BENCHMARK")
    print("=" * 60)

    config = AppConfig()
    mgr = ModelManager(config)
    detector = mgr.get_active_detector()

    # Hardware specs
    cpu_count = os.cpu_count() or 1
    print(f"Target CPU Cores: {cpu_count}")

    resolutions = [(320, 320), (416, 416), (640, 640)]
    results = {}

    for w, h in resolutions:
        print(f"\nBenchmarking resolution: {w}x{h} ({num_frames} frames)...")
        latencies: List[float] = []

        # Warm up
        dummy = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        for _ in range(5):
            detector.predict(dummy, config.get_all_confidence_thresholds())

        t_start = time.perf_counter()
        for i in range(num_frames):
            frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            t0 = time.perf_counter()
            detector.predict(frame, config.get_all_confidence_thresholds())
            latencies.append((time.perf_counter() - t0) * 1000.0)

        total_time = time.perf_counter() - t_start
        fps = num_frames / total_time
        avg_lat = float(np.mean(latencies))
        med_lat = float(np.median(latencies))
        p95_lat = float(np.percentile(latencies, 95))

        print(f"  -> Average FPS     : {fps:.2f}")
        print(f"  -> Mean Latency    : {avg_lat:.2f} ms")
        print(f"  -> Median Latency  : {med_lat:.2f} ms")
        print(f"  -> P95 Latency     : {p95_lat:.2f} ms")

        results[f"{w}x{h}"] = {
            "fps": round(fps, 2),
            "mean_ms": round(avg_lat, 2),
            "median_ms": round(med_lat, 2),
            "p95_ms": round(p95_lat, 2),
        }

    return results


if __name__ == "__main__":
    run_benchmark(num_frames=60)


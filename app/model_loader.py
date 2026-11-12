"""Model loading, integrity verification, and lifecycle management.

Downloads and validates the required pretrained checkpoint:
khanhromvn/waste-detection-yolo11 (waste_detection_yolo11m.pt)
verifying SHA-256 checksum and model class mapping.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import sys
import threading
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_CACHE_LOCK = threading.Lock()
_VERIFIED_SHA256_CACHE: Dict[Tuple[str, int, int], str] = {}
_PRIMARY_MODEL_CACHE: Dict[Tuple[str, int, int], Tuple[Any, Path, str, Dict[int, str]]] = {}


def _file_signature(file_path: Path) -> Optional[Tuple[str, int, int]]:
    """Return a cache key (resolved_path, size, mtime_ns) if file exists."""
    try:
        resolved = file_path.resolve()
        st = resolved.stat()
        return (str(resolved), int(st.st_size), int(st.st_mtime_ns))
    except OSError:
        return None

REPO_ID = "khanhromvn/waste-detection-yolo11"
FILENAME = "waste_detection_yolo11m.pt"
EXPECTED_SHA256 = "9f2d19a7ce386c6dbb95a725e6197f7b6fb699cd601bbeca9407034396f2ad6a"
EXPECTED_CLASSES: Dict[int, str] = {
    0: "organic",
    1: "recyclable",
    2: "residual",
}


def compute_sha256(file_path: Path) -> str:
    """Calculate the SHA-256 checksum of a file."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()


def download_model(
    destination_path: Path,
    repo_id: str = REPO_ID,
    filename: str = FILENAME,
    expected_sha256: str = EXPECTED_SHA256,
) -> Path:
    """Download checkpoint from Hugging Face if not already present or invalid.

    Uses huggingface_hub if available; otherwise uses streaming urllib.request fallback.
    Writes to a temporary file first and verifies SHA-256 before atomic rename.
    """
    destination_path = Path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    # If file exists and matches checksum, reuse it (offline support)
    if destination_path.exists() and destination_path.stat().st_size > 0:
        sig = _file_signature(destination_path)
        if sig is not None:
            with _CACHE_LOCK:
                cached_sha = _VERIFIED_SHA256_CACHE.get(sig)
            if cached_sha and cached_sha.lower() == expected_sha256.lower():
                return destination_path

        logger.info("Verifying existing model checkpoint at %s...", destination_path)
        actual_sha = compute_sha256(destination_path)
        if actual_sha.lower() == expected_sha256.lower():
            if sig is not None:
                with _CACHE_LOCK:
                    _VERIFIED_SHA256_CACHE[sig] = actual_sha
            logger.info("Existing checkpoint verified successfully (SHA-256 match).")
            return destination_path
        else:
            logger.warning(
                "Existing checkpoint checksum mismatch! Expected %s, got %s. Re-downloading...",
                expected_sha256,
                actual_sha,
            )
            destination_path.unlink()

    temp_path = destination_path.with_suffix(".tmp")
    if temp_path.exists():
        temp_path.unlink()

    logger.info("Downloading checkpoint %s from repo %s...", filename, repo_id)

    downloaded = False
    try:
        from huggingface_hub import hf_hub_download

        logger.info("Using huggingface_hub for download...")
        cached_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
        )
        shutil.copy2(cached_file, temp_path)
        downloaded = True
    except ImportError:
        logger.info("huggingface_hub not installed; using direct HTTPS streaming download...")
    except Exception as e:
        logger.warning("huggingface_hub download failed: %s; falling back to direct download...", e)

    if not downloaded:
        direct_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        logger.info("Fetching from %s ...", direct_url)
        headers = {"User-Agent": "WasteDetectionSystem/1.0"}
        req = urllib.request.Request(direct_url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as response, open(temp_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file, length=1024 * 1024)

    if not temp_path.exists() or temp_path.stat().st_size == 0:
        raise RuntimeError(f"Downloaded model file {temp_path} is missing or empty.")

    actual_sha = compute_sha256(temp_path)
    if actual_sha.lower() != expected_sha256.lower():
        temp_path.unlink()
        raise ValueError(
            f"Checkpoint SHA-256 verification failed!\n"
            f"Expected: {expected_sha256}\n"
            f"Actual:   {actual_sha}\n"
            f"Aborted loading untrusted or corrupted checkpoint."
        )

    temp_path.replace(destination_path)
    sig = _file_signature(destination_path)
    if sig is not None:
        with _CACHE_LOCK:
            _VERIFIED_SHA256_CACHE[sig] = actual_sha
    logger.info(
        "Checkpoint downloaded and verified: %s (size: %.1f MB, SHA-256: %s)",
        destination_path.name,
        destination_path.stat().st_size / (1024 * 1024),
        actual_sha,
    )
    return destination_path


def load_waste_model(
    weights_path: Optional[Path] = None,
    device: str = "cpu",
    force_reload: bool = False,
) -> Tuple[Any, Path, str, Dict[int, str]]:
    """Load and validate the Ultralytics YOLO11 waste detection model.

    Reuses already-verified checksums and loaded model instances within the
    current process to avoid redundant disk reads, hashing, and memory usage.

    Returns:
        (model, weights_path, sha256_checksum, class_mapping)
    """
    project_root = Path(__file__).resolve().parent.parent
    if weights_path is None:
        weights_path = project_root / "models" / "pretrained" / FILENAME

    weights_path = download_model(weights_path)
    sig = _file_signature(weights_path)

    if not force_reload and sig is not None:
        with _CACHE_LOCK:
            cached = _PRIMARY_MODEL_CACHE.get(sig)
        if cached is not None:
            model, cached_path, cached_sha, cached_mapping = cached
            logger.info("Reusing cached YOLO model for %s", cached_path.name)
            return model, cached_path, cached_sha, dict(cached_mapping)

    sha256: Optional[str] = None
    if sig is not None:
        with _CACHE_LOCK:
            sha256 = _VERIFIED_SHA256_CACHE.get(sig)
    if not sha256:
        sha256 = compute_sha256(weights_path)
        if sig is not None:
            with _CACHE_LOCK:
                _VERIFIED_SHA256_CACHE[sig] = sha256

    logger.info("Loading YOLO model from %s on device=%s...", weights_path, device)
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as e:
        missing_module = getattr(e, "name", "")
        if missing_module == "ultralytics":
            raise ModuleNotFoundError(
                f"The 'ultralytics' package is not installed in the active Python environment.\n"
                f"Active interpreter: {sys.executable}\n"
                f"To resolve, install dependencies into this environment:\n"
                f"  {sys.executable} -m pip install -r requirements.txt\n"
                f"or using uv:\n"
                f"  uv pip install -r requirements.txt"
            ) from e
        else:
            raise ModuleNotFoundError(
                f"Ultralytics failed to import because its dependency '{missing_module}' is missing in the active environment.\n"
                f"Active interpreter: {sys.executable}\n"
                f"Original error: {e}"
            ) from e
    except ImportError as e:
        raise ImportError(
            f"Failed to import Ultralytics in active Python environment ({sys.executable}).\n"
            f"Underlying error (e.g. DLL/C-extension load failure): {e}"
        ) from e

    model = YOLO(str(weights_path))

    # Verify model classes
    raw_names = getattr(model, "names", {})
    if isinstance(raw_names, dict):
        class_mapping = {int(k): str(v) for k, v in raw_names.items()}
    elif isinstance(raw_names, list):
        class_mapping = {i: str(v) for i, v in enumerate(raw_names)}
    else:
        class_mapping = {}

    logger.info("Loaded model class mapping: %s", class_mapping)

    # Validate against expected classes
    for cid, expected_name in EXPECTED_CLASSES.items():
        actual_name = class_mapping.get(cid)
        if actual_name != expected_name:
            raise ValueError(
                f"Model class mismatch at ID {cid}! Expected '{expected_name}', got '{actual_name}'. "
                f"This checkpoint does not match the required waste detection classes."
            )

    if sig is not None:
        with _CACHE_LOCK:
            _PRIMARY_MODEL_CACHE[sig] = (model, weights_path, sha256, dict(class_mapping))

    return model, weights_path, sha256, class_mapping


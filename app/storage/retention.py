"""Storage management, disk-space monitoring, and retention subsystem.

Ensures the user retains complete manual control over data deletion while
monitoring storage thresholds to protect laptop storage.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.core.config import AppConfig

logger = logging.getLogger(__name__)


class StorageManager:
    """Monitors disk space, computes session sizes, and executes controlled deletions."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.recordings_root = config.recordings_dir

    def get_disk_space_info(self) -> Dict[str, float]:
        """Return free, total, and used disk space in megabytes (MB) on the recordings drive."""
        try:
            target = self.recordings_root if self.recordings_root.exists() else self.recordings_root.parent
            total, used, free = shutil.disk_usage(target)
            return {
                "total_mb": round(total / (1024 * 1024), 1),
                "used_mb": round(used / (1024 * 1024), 1),
                "free_mb": round(free / (1024 * 1024), 1),
            }
        except Exception as e:
            logger.error("Failed querying disk space: %s", e)
            return {"total_mb": 0.0, "used_mb": 0.0, "free_mb": 0.0}

    def get_recordings_total_size_mb(self) -> float:
        """Calculate total size of all session recordings and evidence in MB."""
        if not self.recordings_root.exists():
            return 0.0

        total_bytes = 0
        for p in self.recordings_root.rglob("*"):
            if p.is_file():
                try:
                    total_bytes += p.stat().st_size
                except OSError:
                    pass
        return round(total_bytes / (1024 * 1024), 2)

    def get_session_size_mb(self, session_dir: Path) -> float:
        """Calculate total disk space consumed by a single session directory in MB."""
        if not session_dir.exists() or not session_dir.is_dir():
            return 0.0

        total_bytes = 0
        for p in session_dir.rglob("*"):
            if p.is_file():
                try:
                    total_bytes += p.stat().st_size
                except OSError:
                    pass
        return round(total_bytes / (1024 * 1024), 2)

    def is_storage_low(self) -> Tuple[bool, float]:
        """Check if available free disk space is below warning threshold."""
        info = self.get_disk_space_info()
        free_mb = info.get("free_mb", 0.0)
        threshold_mb = self.config._raw_data.get("storage", {}).get("storage_warning_threshold_mb", 1000)
        is_low = free_mb < threshold_mb
        return is_low, free_mb

    def delete_session(self, session_id: str, confirm: bool = False) -> Tuple[bool, str]:
        """Manually delete a session directory. Requires explicit confirm=True."""
        if not confirm:
            return False, "Deletion rejected: explicit user confirmation is mandatory."

        session_dir = self.recordings_root / session_id
        if not session_dir.exists() or not session_dir.is_dir():
            return False, f"Session directory {session_id} not found."

        try:
            shutil.rmtree(session_dir)
            logger.info("Session %s successfully deleted by user.", session_id)
            return True, f"Session {session_id} successfully deleted."
        except Exception as e:
            logger.error("Failed to delete session %s: %s", session_id, e)
            return False, f"Error deleting session: {e}"


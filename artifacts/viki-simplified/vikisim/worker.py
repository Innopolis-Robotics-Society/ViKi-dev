"""Simplified SkeletonWorker (mirrors viki/server/skeleton_worker.py)."""

from __future__ import annotations

import threading
from typing import Any, Optional

from .calibration import CalibrationManager
from .manager import CameraManager
from .models import HandDetection, SkeletonFrame
from .pipeline import PipelineResult, SkeletonPipeline
from .sync import MultiCameraSync


class SkeletonWorker(threading.Thread):
    """Background thread: pulls synced frames, runs the pipeline, caches result."""

    def __init__(
        self,
        manager: CameraManager,
        sync: MultiCameraSync,
        pipeline: SkeletonPipeline,
        recorder: Any,
        target_fps: float = 15.0,
    ) -> None:
        super().__init__(daemon=True)
        self.manager = manager
        self.sync = sync
        self.pipeline = pipeline
        self.recorder = recorder
        self._enabled = False
        self._latest_result: Optional[PipelineResult] = None
        self._lock = threading.Lock()

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def run(self) -> None:
        while True:
            group = self.sync.get_synced_frame()
            if group is not None and self._enabled:
                result = self.pipeline.process(group)
                with self._lock:
                    self._latest_result = result

    def get_latest_frame(self) -> Optional[SkeletonFrame]:
        with self._lock:
            return self._latest_result.fused_frame if self._latest_result else None

    def get_latest_result(self) -> Optional[PipelineResult]:
        with self._lock:
            return self._latest_result

    def set_depth_debug(self, enabled: bool) -> None:
        self.pipeline.set_depth_debug(enabled)

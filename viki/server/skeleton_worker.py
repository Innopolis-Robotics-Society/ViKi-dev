"""
viki.server.skeleton_worker
--------------------------
Background worker that runs the skeleton pipeline on demand.
"""

from __future__ import annotations

# from asyncio.windows_events import NULL
import threading
import time
import os
from typing import Optional

import numpy as np
from viki.capture.base import SyncedFrameGroup
from viki.capture.kinect import KinectBackend
from viki.skeleton.models import SkeletonFrame, HandDetection
from viki.skeleton.pipeline import SkeletonPipeline, PipelineResult
from viki.skeleton.recorder import SkeletonRecorder
from viki.capture.sync import MultiCameraSync
from viki.capture.manager import CameraManager
from viki.capture.recorder import RGBDRecorder
from viki.skeleton.viz import visualize_color_depth_mapping
from viki.config import DEPTH_PROJECTION_DEBUG


class SkeletonWorker:
    """
    Manages a background thread that periodically runs the skeleton pipeline.
    """

    def __init__(
        self,
        manager: CameraManager,
        sync: MultiCameraSync,
        pipeline: SkeletonPipeline,
        recorder: SkeletonRecorder,
        target_fps: float = 15.0,
    ) -> None:
        self._manager = manager
        self._sync = sync
        self._pipeline = pipeline
        self._recorder = recorder
        self._target_fps = target_fps
        self._interval = 1.0 / target_fps

        self._enabled = False
        self._recording = False

        # RGB-D Recording state
        self._rgbd_recording = False
        self._rgbd_recorder: Optional[RGBDRecorder] = None
        self._rgbd_stop_time: float = 0.0

        self._latest_result: Optional[PipelineResult] = None
        self._lock = threading.Lock()
        self._last_viz_time: float = 0.0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the background worker thread."""
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background worker thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable skeleton estimation."""
        self._enabled = enabled
        if not enabled and self._recording:
            self.set_recording(False)

    def set_recording(self, recording: bool) -> None:
        """Enable or disable recording to disk."""
        if recording and not self._enabled:
            self._enabled = True  # Must be enabled to record

        if recording and not self._recording:
            self._recorder.start()
        elif not recording and self._recording:
            self._recorder.stop()

        self._recording = recording

    def set_rgbd_recording(
        self, enabled: bool, duration: float = 10.0, output_dir: str = "data/videos"
    ) -> None:
        """Enable or disable synchronized RGB-D recording."""
        with self._lock:
            if enabled:
                self._rgbd_recorder = RGBDRecorder(
                    self._manager, output_base_dir=output_dir
                )
                self._rgbd_recording = True
                self._rgbd_stop_time = time.monotonic() + duration
            else:
                if self._rgbd_recorder:
                    self._rgbd_recorder.stop()
                self._rgbd_recorder = None
                self._rgbd_recording = False

    def get_latest_frame(self) -> Optional[SkeletonFrame]:
        """Return the most recently processed skeleton frame."""
        with self._lock:
            return self._latest_result.fused_frame if self._latest_result else None

    def get_latest_detections(self) -> dict[str, HandDetection | None]:
        """Return the most recent 2D detections per camera."""
        with self._lock:
            return self._latest_result.detections if self._latest_result else {}

    def _run(self) -> None:
        """Main loop of the worker thread."""
        import logging

        logger = logging.getLogger(__name__)
        while not self._stop_event.is_set():
            start_time = time.monotonic()

            # Check for RGB-D recording timeout
            if self._rgbd_recording and time.monotonic() > self._rgbd_stop_time:
                logger.info("RGB-D recording duration elapsed. Stopping...")
                self.set_rgbd_recording(False)

            if self._enabled or self._rgbd_recording:
                try:
                    # 1. Get synced frames
                    group = self._sync.get_synced_frame()
                    if group:
                        # A. If RGB-D recording is active, save the raw synced group
                        if self._rgbd_recording and self._rgbd_recorder:
                            self._rgbd_recorder.save_group(group, int(self._target_fps))

                        # B. Process skeleton if enabled
                        if self._enabled:
                            result = self._pipeline.process(group)
                            with self._lock:
                                self._latest_result = result
                            if self._recording and result.fused_frame:
                                self._recorder.record(result.fused_frame)
                    else:
                        # No synced frames - if recording, we could write a duplicate here
                        # but for now we just let it be (MultiCameraSync returns None)
                        pass
                except Exception as e:
                    logger.exception(f"Worker pipeline error: {e}")
                    time.sleep(1)

            # Maintain target FPS
            elapsed = time.monotonic() - start_time
            sleep_time = self._interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def is_recording(self) -> bool:
        return self._recording

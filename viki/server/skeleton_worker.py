"""
viki.server.skeleton_worker
--------------------------
Background worker that runs the skeleton pipeline on demand.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
from viki.skeleton.models import SkeletonFrame, HandDetection
from viki.skeleton.pipeline import SkeletonPipeline, PipelineResult
from viki.skeleton.recorder import SkeletonRecorder
from viki.capture.sync import MultiCameraSync
from viki.capture.manager import CameraManager


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
        target_fps: float = 15.0
    ) -> None:
        self._manager = manager
        self._sync = sync
        self._pipeline = pipeline
        self._recorder = recorder
        self._target_fps = target_fps
        self._interval = 1.0 / target_fps

        self._enabled = False
        self._recording = False
        
        self._latest_result: Optional[PipelineResult] = None
        self._lock = threading.Lock()
        
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
            self._thread.join()
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
            
            if self._enabled:
                try:
                    # 1. Get synced frames
                    group = self._sync.get_synced_frame()
                    if group:
                        # 2. Process skeleton
                        result = self._pipeline.process(group)
                        
                        with self._lock:
                            self._latest_result = result

                        if result.fused_frame is None:
                            # Log occasionally that we got frames but no detection
                            if np.random.random() < 0.01:
                                logger.debug("SkeletonWorker: Received synced frames but pipeline returned no fused frame (no detection or missing calib).")
                        else:
                            # 3. Record if enabled
                            if self._recording:
                                self._recorder.record(result.fused_frame)
                    else:
                        # This happens if cameras aren't producing frames or sync is failing
                        if np.random.random() < 0.01:
                            logger.warning("SkeletonWorker: No synced frames available from MultiCameraSync.")
                except Exception as e:
                    logger.exception(f"Skeleton worker pipeline error: {e}")
            
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

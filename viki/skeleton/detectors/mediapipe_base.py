"""
viki.skeleton.detectors.mediapipe_base
--------------------------------------
Shared MediaPipe Tasks infrastructure that is identical across
HandLandmarker / PoseLandmarker / any other task so delibeerated here
"""

from __future__ import annotations

import logging
import threading
import urllib.request
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR_DEFAULT: str = "models"


def ensure_model(
    filename: str,
    url: str,
    models_dir: str = MODELS_DIR_DEFAULT,
) -> str:
    """
    Download a MediaPipe .task file once and cache it locally.

    parameters
    ----------
    filename   : target file name under `models_dir`.
    url        : remote source URL.
    models_dir : local cache directory.

    returns
    -------
    Absolute filesystem path to the local model file.
    """
    path = Path(models_dir) / filename
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading MediaPipe model %s …", filename)
        urllib.request.urlretrieve(url, path)
        logger.info("Saved MediaPipe model to %s", path)
    return str(path)


class MediaPipeTaskRunner:
    """Mode-aware wrapper around one MediaPipe Tasks vision model."""

    def __init__(
        self,
        task_factory: Callable[..., Any],
        model_path: str,
        mode: Literal["image", "video", "live"] = "image",
    ) -> None:
        """
        parameters
        ----------
        task_factory : (base_options, running_mode, result_callback_or_None) -> task.
        model_path   : path to the .task model file.
        mode         : "image", "video", "live".
        """

        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self._mode = mode
        self._lock = threading.Lock()
        self._live_last_result: Any = None
        self._live_last_returned_ts_ms: int = -1
        self._last_submitted_ts_ms: int = -1

        running_mode = self._map_mode(mode, vision)

        callback: Optional[Callable] = None
        if mode == "live":
            def _cb(result, _img, _ts):
                with self._lock:
                    self._live_last_result = result
            callback = _cb

        base_options = python.BaseOptions(model_asset_path=model_path)
        self._task = task_factory(base_options, running_mode, callback)

    @staticmethod
    def _map_mode(mode: str, vision) -> Any:
        if mode == "video":
            return vision.RunningMode.VIDEO
        if mode == "live":
            return vision.RunningMode.LIVE_STREAM
        return vision.RunningMode.IMAGE

    def submit(self, rgb: np.ndarray, timestamp_us: int) -> Optional[Any]:
        """
        Submit one frame to the underlying MediaPipe task.

        parameters
        ----------
        rgb          : (H, W, 3) uint8 RGB image.
        timestamp_us : frame timestamp in microseconds.

        returns
        -------
        Raw MediaPipe result object (task-specific type), or None when no
        result is available yet — first frames in LIVE mode, or stale
        cached result for the same timestamp.
        """
        # Lazy: avoid module-level mediapipe import cost.
        import mediapipe as mp

        timestamp_ms = timestamp_us // 1000
        # VIDEO and LIVE require strictly increasing timestamps.
        if timestamp_ms <= self._last_submitted_ts_ms:
            timestamp_ms = self._last_submitted_ts_ms + 1
        self._last_submitted_ts_ms = timestamp_ms

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )

        if self._mode == "image":
            return self._task.detect(mp_image)
        if self._mode == "video":
            return self._task.detect_for_video(mp_image, timestamp_ms)

        # LIVE: submit async, return latest cached result if fresh.
        self._task.detect_async(mp_image, timestamp_ms)
        with self._lock:
            cached = self._live_last_result
            last_returned = self._live_last_returned_ts_ms
        if cached is None:
            return None


        if last_returned == timestamp_ms:
            return None
        with self._lock:
            self._live_last_returned_ts_ms = timestamp_ms
        return cached

    def close(self) -> None:
        """Release the underlying MediaPipe task (frees native resources)."""
        task = getattr(self, "_task", None)
        if task is not None:
            task.close()
            self._task = None

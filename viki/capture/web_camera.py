from __future__ import annotations

import logging
import time
import numpy as np
import cv2

from typing import Dict, List, Optional

from .base import CameraBackend, Frame


class WebCameraBackend(CameraBackend):

    def __init__(
        self,
        idx: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[float] = None,
    ) -> None:
        self.idx = idx
        self._logger = logging.getLogger(__name__)
        self._running = False
        self._width = width
        self._height = height
        self._fps = fps
        self._cap = None

        # We don't open the camera in __init__ to avoid resource leaks 
        # and allow fresh connection on start()


    def __del__(self) -> None:
        if hasattr(self, "_cap") and self._cap is not None:
            self._cap.release()

    def start(self) -> None:
        if self._cap is not None:
            self._cap.release()

        self._logger.info(f"Starting WebCameraBackend on index {self.idx}...")
        
        # Try CAP_V4L2 first
        self._cap = cv2.VideoCapture(self.idx, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            self._logger.warning("CAP_V4L2 failed, trying default backend")
            self._cap = cv2.VideoCapture(self.idx)

        if not self._cap.isOpened():
            self._logger.error(f"Could not open webcam {self.idx}")
            self._running = False
            return

        # 1. Try to force MJPG format - often fixes V4L2 select() timeouts
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # 2. Set buffer size to 1 to prevent lag/timeouts
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if self._width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        if self._height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if self._fps:
            self._cap.set(cv2.CAP_PROP_FPS, self._fps)

        self._running = True
        self._logger.info(f"WebCameraBackend started: {self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} @ {self._cap.get(cv2.CAP_PROP_FPS)}fps")

    def stop(self) -> None:
        self._running = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def get_frame(self) -> Frame:
        if not self._running or self._cap is None:
            msg = "WebCameraBackend is not started or cap is None."
            self._logger.debug(msg)
            time.sleep(1)
            raise RuntimeError(msg)

        ret, frame = self._cap.read()
        if not ret:
            msg = "Failed to retrieve frames from Web Camera."
            self._logger.warning(msg)
            # Attempt to reconnect once
            self._logger.info("Attempting to reconnect to webcam...")
            self.start()
            if not self._running:
                raise RuntimeError(msg)
            
            # Try one more read after reconnect
            ret, frame = self._cap.read()
            if not ret:
                raise RuntimeError(msg)

        color = np.array(frame)
        depth = np.zeros_like(color)

        timestamp_us = int(time.time() * 1_000_000)

        return Frame(
            color=color,
            depth=depth,
            timestamp_us=timestamp_us,
            device_id=self.device_id,
        )

    @property
    def device_id(self) -> str:
        return WebCameraBackend.device_id_static(self.idx)

    @property
    def is_running(self) -> bool:
        return self._running

    @staticmethod
    def device_id_static(idx: int) -> str:
        return f"web_camera_{idx}"

    @staticmethod
    def list_devices() -> List[str]:
        return [WebCameraBackend.device_id_static(0)]
        # MAX_IDX = 5
        # available: List[str] = []

        # for i in range(MAX_IDX):
        #     try:
        #         cap = cv2.VideoCapture(i)
        #         if cap.isOpened() and cap.grab():
        #             available.append(WebCameraBackend.device_id_static(i))
        #         cap.release()
        #     except Exception:
        #         continue

        # return available

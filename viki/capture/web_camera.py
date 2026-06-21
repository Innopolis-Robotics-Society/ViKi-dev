from __future__ import annotations

import logging
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
        self._cap = cv2.VideoCapture(idx)

        if width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            self._cap.set(cv2.CAP_PROP_FPS, fps)

        self._frame_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS))

    def __del__(self) -> None:
        if hasattr(self, "_cap") and self._cap is not None:
            self._cap.release()

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def get_frame(self) -> Frame:
        if not self._running:
            msg = "WebCameraBackend is not started. Call start() first."
            self._logger.debug(msg)
            raise RuntimeError(msg)

        ret, frame = self._cap.read()
        color = np.array(frame)
        depth = np.zeros_like(color)

        if not ret:
            msg = "Failed to retrieve frames from Web Camera."
            self._logger.warning(msg)
            raise RuntimeError(msg)

        timestamp_us = int(self._cap.get(cv2.CAP_PROP_POS_MSEC) * 1000)  # ms -> us

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

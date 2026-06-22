from abc import ABC, abstractmethod
import threading
import cv2
import numpy as np
import logging
from typing import Dict, List
from viki.capture.base import Frame
from viki.capture.manager import CameraManager
from viki.calibration.models import (
    BoardParameters,
    CalibrationSample,
    CalibrationIntrinsics,
    CalibrationExtrinsics,
)


class _CalibrationWorker(ABC):
    def __init__(
        self, mgr: CameraManager, device_id: str, board_params: BoardParameters
    ):
        self._mgr = mgr
        self.device_id = device_id
        self._logger = logging.getLogger(__name__)
        self._samples: List[CalibrationSample] = []
        self._board_params = board_params

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def set_board_params(self, board_params: BoardParameters) -> None:
        with self._lock:
            self._board_params = board_params

    @property
    def board_params(self) -> BoardParameters:
        with self._lock:
            return self._board_params

    @property
    def samples_count(self) -> int:
        with self._lock:
            return len(self._samples)

    @property
    def samples(self) -> List[CalibrationSample]:
        with self._lock:
            return self._samples.copy()

    @abstractmethod
    def add_sample(self, frame: Frame) -> None:
        pass

    @abstractmethod
    def intrinsics_calibration(
        self, samples: List[CalibrationSample] | None = None
    ) -> CalibrationIntrinsics:
        pass

    @abstractmethod
    def extrinsics_calibration(
        self,
        intrinsics: CalibrationIntrinsics,
        sample: CalibrationSample | None = None,
    ) -> CalibrationExtrinsics:
        pass

    def clear(self):
        with self._lock:
            self._samples = []

    def capture(self) -> None:
        frame = self._mgr.latest_frame(self.device_id)
        if frame is None:
            return
        self.add_sample(frame)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.capture()
            except TimeoutError:
                pass
            except Exception as e:
                self._logger.error(f"{self.device_id} calibration worker error: {e}")

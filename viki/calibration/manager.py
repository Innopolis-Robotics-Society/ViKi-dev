import cv2
import logging
from typing import Dict, List
from viki.capture.manager import CameraManager
from viki.calibration.models import (
    ArucoBoardParameters,
    BoardParameters,
    CalibrationSample,
    CalibrationIntrinsics,
    CalibrationExtrinsics,
)
from viki.config import (
    INTRINSICS_FILENAME,
    EXTRINSICS_FILENAME,
    CALIB_MODE,
    CALIB_BOARD_TYPE,
    CALIB_CHESS_BOARD_SIZE,
    CALIB_CHESS_SQUARE_SIZE,
    CALIB_ARUCO_BOARD_SIZE,
    CALIB_ARUCO_SQUARE_SIZE,
    CALIB_ARUCO_MARKER_SIZE,
    CALIB_ARUCO_DICT,
)
from viki.calibration.file import (
    read_device_intrinsics,
    read_device_extrinsics,
    write_device_intrinsics,
    write_device_extrinsics,
)
from viki.calibration.worker import _CalibrationWorker
from viki.calibration.chessboard_worker import ChessboardWorker
from viki.calibration.aruco_worker import ArucoWorker


class CalibrationManager:
    def __init__(self, mgr: CameraManager):
        self._mgr = mgr
        self._logger = logging.getLogger(__name__)
        self._intrinsics: Dict[str, CalibrationIntrinsics] = {}
        self._extrinsics: Dict[str, CalibrationExtrinsics] = {}
        self._workers: Dict[str, _CalibrationWorker] = {}

    def start(
        self,
        device_id: str,
        mode=CALIB_MODE,
        board_type=CALIB_BOARD_TYPE,
        board_size=None,
        square_size=None,
        marker_size=None,
        aruco_dict=CALIB_ARUCO_DICT,
    ) -> None:
        """
        mode: str = ["auto", "manual"], manual - capture image manually, via add_sample(), auto - worker will try to capture image itself
        """
        if device_id in self._workers:
            self._logger.warning(
                f"CalibrationManager start: {device_id} has already started"
            )
            return

        if board_type == "chess":
            bs = board_size or CALIB_CHESS_BOARD_SIZE
            ss = square_size or CALIB_CHESS_SQUARE_SIZE
            board_params = BoardParameters(bs, ss)
            worker = ChessboardWorker(self._mgr, device_id, board_params)
        else:
            bs = board_size or CALIB_ARUCO_BOARD_SIZE
            ss = square_size or CALIB_ARUCO_SQUARE_SIZE
            ms = marker_size or CALIB_ARUCO_MARKER_SIZE
            board_params = ArucoBoardParameters(
                bs, ss, ms, aruco_dict
            )
            worker = ArucoWorker(self._mgr, device_id, board_params)
        if mode == "auto":
            worker.start()
        self._workers[device_id] = worker

    def stop(self, device_id: str) -> None:
        worker = self._workers.pop(device_id, None)
        if worker:
            worker.stop()
            return
        self._logger.warning(
            f"CalibrationManager stop: {device_id} is not in worker list"
        )

    def stop_all(self) -> None:
        for device_id in list(self._workers):
            self.stop(device_id)

    def sync_params(
        self,
        board_type: str,
        board_size: tuple[int, int],
        square_size: float,
        marker_size: float = CALIB_ARUCO_MARKER_SIZE,
        aruco_dict: int = CALIB_ARUCO_DICT,
    ) -> None:
        if board_type == "chess":
            params = BoardParameters(board_size, square_size)
        else:
            params = ArucoBoardParameters(board_size, square_size, marker_size, aruco_dict)
        
        for worker in self._workers.values():
            worker.set_board_params(params)

    def is_device_active(self, device_id: str) -> bool:
        if self._workers.get(device_id):
            return True
        return False

    def clear(self, device_id: str) -> None:
        worker = self._workers.get(device_id)
        if not worker:
            self._logger.warning(
                f"CalibrationManager status: {device_id} is not in worker list"
            )
            return
        worker.clear()

    def intrinsics_calibration(
        self,
        device_id: str,
        results_path: str = INTRINSICS_FILENAME,
        samples: List[CalibrationSample] | None = None,
    ) -> CalibrationIntrinsics:

        worker = self._workers.get(device_id)
        if not worker:
            msg = f"CalibrationManager intrinsics_calibration: {device_id} is not in worker list"
            self._logger.warning(msg)
            raise RuntimeError(msg)

        intrinsics = (
            worker.intrinsics_calibration(samples)
            if samples
            else worker.intrinsics_calibration()
        )

        write_device_intrinsics(device_id, intrinsics, results_path)

        return intrinsics

    def write_intrinsics(
        self,
        device_id: str,
        intrinsics: CalibrationIntrinsics,
        path: str = INTRINSICS_FILENAME,
    ):
        write_device_intrinsics(device_id, intrinsics, path)

    def load_intrinsics(self, device_id: str, path: str = INTRINSICS_FILENAME) -> None:

        intrinsics = read_device_intrinsics(device_id, path)
        if not intrinsics:
            return
        self._intrinsics[device_id] = intrinsics

    def set_intrinsics(
        self, device_id: str, intrinsics: CalibrationIntrinsics, path: str = ""
    ) -> None:
        if path != "":
            self.write_intrinsics(device_id, intrinsics, path)
        self._intrinsics[device_id] = intrinsics

    def get_intrinsics(
        self, device_id: str, path: str = ""
    ) -> CalibrationIntrinsics | None:
        if path != "":
            self.load_intrinsics(device_id, path)
        intrinsics = self._intrinsics.get(device_id)
        return intrinsics

    def extrinsics_calibration(
        self,
        device_id: str,
        results_path: str = EXTRINSICS_FILENAME,
        sample: CalibrationSample | None = None,
        intrinsics: CalibrationIntrinsics | None = None,
    ) -> CalibrationExtrinsics:

        worker = self._workers.get(device_id)
        if not worker:
            msg = f"CalibrationManager extrinsics_calibration: {device_id} is not in worker list"
            self._logger.warning(msg)
            raise RuntimeError(msg)

        if not intrinsics:
            intrinsics = self.get_intrinsics(device_id)
            if not intrinsics:
                msg = f"CalibrationManager extrinsics_calibration: no intrinsics"
                self._logger.warning(msg)
                raise RuntimeError(msg)

        extrinsics = (
            worker.extrinsics_calibration(intrinsics, sample)
            if sample
            else worker.extrinsics_calibration(intrinsics)
        )

        write_device_extrinsics(device_id, extrinsics, results_path)

        return extrinsics

    def write_extrinsics(
        self,
        device_id: str,
        extrinsics: CalibrationExtrinsics,
        path: str = EXTRINSICS_FILENAME,
    ):
        write_device_extrinsics(device_id, extrinsics, path)

    def load_extrinsics(self, device_id: str, path: str = EXTRINSICS_FILENAME) -> None:

        extrinsics = read_device_extrinsics(device_id, path)
        if not extrinsics:
            return
        self._extrinsics[device_id] = extrinsics

    def set_extrinsics(
        self, device_id: str, extrinsics: CalibrationExtrinsics, path: str = ""
    ) -> None:
        if path != "":
            self.write_extrinsics(device_id, extrinsics, path)
        self._extrinsics[device_id] = extrinsics

    def get_extrinsics(
        self, device_id: str, path: str = ""
    ) -> CalibrationExtrinsics | None:
        if path != "":
            self.load_extrinsics(device_id, path)
        extrinsics = self._extrinsics.get(device_id)
        if not extrinsics:
            self._logger.debug(
                f"CalibrationManager get_extrinsics: {device_id} not in extrinsics list"
            )
        return extrinsics

    def capture_all(self) -> None:
        for device_id, worker in self._workers.items():
            worker.capture()

    def capture(self, device_id: str) -> None:
        worker = self._workers.get(device_id)
        if not worker:
            self._logger.warning(
                f"CalibrationManager capture: {device_id} not in workers list"
            )
            return
        worker.capture()

    def samples_count(self, device_id: str) -> int:
        worker = self._workers.get(device_id)
        if not worker:
            self._logger.debug(
                f"CalibrationManager samples_amount: {device_id} is not in worker list"
            )
            return 0
        return worker.samples_count

    def status(self, device_id: str) -> dict:
        worker = self._workers.get(device_id)
        if not worker:
            return {"samples_count": 0, "started": False}
        return {"samples_count": worker.samples_count, "started": True}

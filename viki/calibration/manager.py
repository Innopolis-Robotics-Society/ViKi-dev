import threading
import cv2
import numpy as np
import json
import logging
from typing import Dict, List, Tuple
from viki.capture.base import Frame
from viki.capture.manager import CameraManager
from viki.calibration.models import (
    CalibrationSample,
    CalibrationIntrinsics,
    CalibrationExtrinsics,
)


class _CalibrationWorker:
    def __init__(
        self,
        mgr: CameraManager,
        device_id: str,
        chessboard_size: Tuple[int, int],
        square_size: float,
    ):
        self._mgr = mgr
        self.device_id = device_id
        self._logger = logging.getLogger(__name__)
        self._samples: List[CalibrationSample] = []
        self._chessboard_size = chessboard_size
        self._square_size = square_size

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def set_board_params(
        self, chessboard_size: Tuple[int, int], square_size: float
    ) -> None:
        with self._lock:
            self._chessboard_size = chessboard_size
            self._square_size = square_size

    @property
    def board_params(self) -> Tuple[Tuple[int, int], float]:
        with self._lock:
            return self._chessboard_size, self._square_size

    @property
    def samples_count(self) -> int:
        with self._lock:
            return len(self._samples)

    @property
    def samples(self) -> List[CalibrationSample]:
        with self._lock:
            return self._samples.copy()

    def add_sample(self, frame: Frame) -> None:

        chessboard_size, square_size = self.board_params

        subpix_criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )

        gray = cv2.cvtColor(frame.color, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if not ret:
            self._logger.debug(
                f"{self.device_id} add sample: cv2.findChessboardCorners failed, chessboard_size: {chessboard_size}"
            )
            return

        refined_corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), subpix_criteria
        )

        w, h = frame.color.shape[:2]

        sample = CalibrationSample(
            frame, refined_corners, (w, h), chessboard_size, square_size
        )
        with self._lock:
            self._samples.append(sample)

        self._logger.debug(f"{self.device_id} add sample: success")

    def intrinsics_calibration(
        self, samples: List[CalibrationSample] | None = None
    ) -> Dict:

        if not samples:
            samples = self.samples
        count = len(samples)

        if count < 20:
            msg = f"{self.device_id} intrinsics calibration: not enough samples"
            self._logger.debug(msg)
            return {"status": "failed", "msg": msg}

        res = samples[0].resolution
        if not all(res == sample.resolution for sample in samples):
            msg = f"{self.device_id} intrinsics calibration: varying resolutions detected, expected same for all images, {set(sample.resolution for sample in self._samples)}"
            self._logger.debug(msg)
            return {
                "status": "failed",
                "msg": msg,
            }

        w, h = res
        object_points = []
        image_points = []

        for sample in samples:
            square_size = sample.square_size
            w, h = sample.chessboard_size

            objp = np.zeros((w * h, 3), np.float32)
            objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
            objp *= square_size

            object_points.append(objp)
            image_points.append(sample.corners)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, (w, h), None, None  # pyright: ignore
        )

        if not ret:
            msg = f"{self.device_id} intrinsics calibration: cv2.calibrateCamera failed"
            self._logger.debug(msg)
            return {"status": "failed", "msg": msg}

        self._logger.debug(f"{self.device_id} intrinsics calibration: success")
        return {
            "status": "success",
            "camera_matrix": mtx,
            "dist_coeffs": dist,
            "reprojection_error": float(ret),
            "samples_used": count,
            "resolution": (w, h),
            "rvecs": rvecs,
            "tvecs": tvecs,
        }

    def extrinsics_calibration(
        self,
        intrinsics: CalibrationIntrinsics,
        sample: CalibrationSample | None = None,
    ) -> Dict:

        if not sample:
            if self.samples_count < 1:
                msg = f"{self.device_id} extrinsics_calibration: no sample"
                self._logger.debug(msg)
                return {"status": "failed", "msg": msg}
            sample = self.samples[-1]

        square_size = sample.square_size
        w, h = sample.chessboard_size

        objp = np.zeros((w * h, 3), np.float32)
        objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
        objp *= square_size

        camera_matrix = intrinsics.camera_matrix
        dist_coeffs = intrinsics.dist_coeffs
        ret, rvec, tvec = cv2.solvePnP(objp, sample.corners, camera_matrix, dist_coeffs)

        if not ret:
            msg = f"{self.device_id} extrinsics calibration: cv2.solvePnP failed"
            self._logger.debug(msg)
            return {"status": "failed", "msg": msg}

        self._logger.debug(f"{self.device_id} extrinsics calibration: success")

        return {
            "status": "success",
            "rvec": rvec,
            "tvec": tvec,
        }

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


class CalibrationManager:
    def __init__(self, mgr: CameraManager):
        self._mgr = mgr
        self._logger = logging.getLogger(__name__)
        self._intrinsics: Dict[str, CalibrationIntrinsics] = {}
        self._extrinsics: Dict[str, CalibrationExtrinsics] = {}
        self._workers: Dict[str, _CalibrationWorker] = {}

    def start(
        self, device_id: str, chessboard_size=(8, 6), square_size=0.025, mode="auto"
    ) -> None:
        """
        mode: str = ["auto", "manual"], manual - capture image manually, via add_sample(), auto - worker will try to capture image itself
        """
        if device_id in self._workers:
            self._logger.warning(
                f"CalibrationManager start: {device_id} has already started"
            )
            return

        worker = _CalibrationWorker(self._mgr, device_id, chessboard_size, square_size)
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
        results_path: str = "data/intrinsics.json",
        samples: List[CalibrationSample] | None = None,
    ) -> Dict:

        worker = self._workers.get(device_id)
        if not worker:
            msg = f"CalibrationManager intrinsics_calibration: {device_id} is not in worker list"
            self._logger.warning(msg)
            return {"status": "failed", "msg": msg}

        result = (
            worker.intrinsics_calibration(samples)
            if samples
            else worker.intrinsics_calibration()
        )
        if result.get("status", "failed") != "success":
            msg = f"CalibrationManager intrinsics_calibration: worker calibration failed: {result.get('msg', 'no message')}"
            self._logger.debug(msg)
            return {"status": "failed", "msg": msg}

        data = {
            "device_id": device_id,
            "camera_matrix": result.get("camera_matrix"),
            "dist_coeffs": result.get("dist_coeffs"),
        }

        with open(results_path, "w") as f:
            json.dump(data, f)

        return result

    def load_intrinsics(
        self, device_id: str, path: str = "data/intrinsics_calibration.json"
    ) -> None:

        with open(path, "r") as f:
            data = json.load(f)

            mtx = np.array(data["camera_matrix"])
            fx, fy, cx, cy = mtx[0, 0], mtx[1, 1], mtx[0, 2], mtx[1, 2]

            dist_coeffs = np.array(data["dist_coeffs"])

            intrinsics = CalibrationIntrinsics(fx, fy, cx, cy, dist_coeffs)
            self._intrinsics[device_id] = intrinsics

    def set_intrinsics(self, device_id: str, intrinsics: CalibrationIntrinsics) -> None:
        self._intrinsics[device_id] = intrinsics

    def get_intrinsics(self, device_id: str) -> CalibrationIntrinsics | None:
        intrinsics = self._intrinsics.get(device_id)
        if not intrinsics:
            self._logger.debug(
                f"CalibrationManager get_intrinsics: {device_id} not in intrinsics list"
            )
        return intrinsics

    def extrinsics_calibration(
        self,
        device_id: str,
        results_path: str = "data/extrinsics.json",
        sample: CalibrationSample | None = None,
        intrinsics: CalibrationIntrinsics | None = None,
    ) -> Dict:

        worker = self._workers.get(device_id)
        if not worker:
            msg = f"CalibrationManager extrinsics_calibration: {device_id} is not in worker list"
            self._logger.warning(msg)
            return {"status": "failed", "msg": msg}

        if not intrinsics:
            intrinsics = self.get_intrinsics(device_id)
            if not intrinsics:
                msg = f"CalibrationManager extrinsics_calibration: no intrinsics"
                self._logger.warning(msg)
                return {"status": "failed", "msg": msg}

        result = (
            worker.extrinsics_calibration(intrinsics, sample)
            if sample
            else worker.extrinsics_calibration(intrinsics)
        )
        if result.get("status", "failed") != "success":
            msg = f"CalibrationManager extrinsics_calibration: worker calibration failed: {result.get('msg', 'no message')}"
            self._logger.debug(msg)
            return {"status": "failed", "msg": msg}

        data = {
            "device_id": device_id,
            "rvec": result.get("rvec"),
            "tvec": result.get("tvec"),
        }

        with open(results_path, "w") as f:
            json.dump(data, f)

        return result

    def load_extrinsics(
        self, device_id: str, path: str = "data/extrinsics_calibration.json"
    ) -> None:

        with open(path, "r") as f:
            data = json.load(f)

            rvec = np.ndarray(data["rvec"])
            tvec = np.ndarray(data["tvec"])
            extrinsics = CalibrationExtrinsics(rvec, tvec)
            self._extrinsics[device_id] = extrinsics

    def set_extrinsics(self, device_id: str, extrinsics: CalibrationExtrinsics) -> None:
        self._extrinsics[device_id] = extrinsics

    def get_extrinsics(self, device_id: str) -> CalibrationExtrinsics | None:
        extrinsics = self._extrinsics.get(device_id)
        if not extrinsics:
            self._logger.debug(
                f"CalibrationManager get_extrinsics: {device_id} not in extrinsics list"
            )
        return extrinsics

    def capture_all(self) -> None:
        for _, worker in self._workers.items():
            worker.capture()
        return

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
            self._logger.warning(
                f"CalibrationManager samples_amount: {device_id} not in workers list"
            )
            return 0
        return worker.samples_count

    def status(self, device_id: str) -> int:
        return self.samples_count(device_id)

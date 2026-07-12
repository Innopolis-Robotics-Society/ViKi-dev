import cv2
import numpy as np
from typing import List
from viki.capture.base import Frame
from viki.calibration.models import (
    CalibrationSample,
    CalibrationIntrinsics,
    CalibrationExtrinsics,
)
from viki.calibration.worker import _CalibrationWorker


class ChessboardWorker(_CalibrationWorker):

    def add_sample(self, frame: Frame) -> None:

        board_params = self.board_params
        board_size = board_params.board_size

        subpix_criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )

        gray = cv2.cvtColor(frame.color, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        if not ret:
            self._logger.debug(
                f"{self.device_id} add sample: cv2.findChessboardCorners failed, chessboard_size: {board_size}"
            )
            return

        refined_corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), subpix_criteria
        )

        w, h = frame.color.shape[:2]

        sample = CalibrationSample(frame, refined_corners, (w, h), board_params)
        with self._lock:
            self._samples.append(sample)

        self._logger.debug(f"{self.device_id} add sample: success")

    def intrinsics_calibration(
        self, samples: List[CalibrationSample] | None = None
    ) -> CalibrationIntrinsics:

        if not samples:
            samples = self.samples
        count = len(samples)

        if count < 20:
            msg = f"{self.device_id} intrinsics calibration: not enough samples"
            self._logger.debug(msg)
            raise RuntimeError(msg)

        res = samples[0].resolution
        if not all(res == sample.resolution for sample in samples):
            msg = f"{self.device_id} intrinsics calibration: varying resolutions detected, expected same for all images, {set(sample.resolution for sample in self._samples)}"
            self._logger.debug(msg)
            raise RuntimeError(msg)

        w, h = res
        object_points = []
        image_points = []
 
        for sample in samples:
            square_size = sample.board_params.square_size
            w, h = sample.board_params.board_size
 
            objp = np.zeros((w * h, 3), np.float32)
            for i in range(h):
                for j in range(w):
                    objp[i * w + j] = [j * square_size, i * square_size, 0]
 
            object_points.append(objp)
            image_points.append(sample.corners)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, (w, h), None, None  # pyright: ignore
        )

        if not ret:
            msg = f"{self.device_id} intrinsics calibration: cv2.calibrateCamera failed"
            self._logger.debug(msg)
            raise RuntimeError(msg)

        self._logger.debug(f"{self.device_id} intrinsics calibration: success")
        return CalibrationIntrinsics(
            fx=mtx[0, 0],
            fy=mtx[1, 1],
            cx=mtx[0, 2],
            cy=mtx[1, 2],
            dist_coeffs=dist.flatten(),
        )

    def extrinsics_calibration(
        self,
        intrinsics: CalibrationIntrinsics,
        sample: CalibrationSample | None = None,
    ) -> CalibrationExtrinsics:

        if not sample:
            if self.samples_count < 1:
                msg = f"{self.device_id} extrinsics_calibration: no sample"
                self._logger.debug(msg)
                raise RuntimeError(msg)
            sample = self.samples[-1]
 
        square_size = sample.board_params.square_size
        w, h = sample.board_params.board_size
 
        objp = np.zeros((w * h, 3), np.float32)
        for i in range(h):
            for j in range(w):
                objp[i * w + j] = [j * square_size, i * square_size, 0]
 
        camera_matrix = intrinsics.camera_matrix
        dist_coeffs = intrinsics.dist_coeffs
        ret, rvec, tvec = cv2.solvePnP(objp, sample.corners, camera_matrix, dist_coeffs)

        if not ret:
            msg = f"{self.device_id} extrinsics calibration: cv2.solvePnP failed"
            self._logger.debug(msg)
            raise RuntimeError(msg)

        self._logger.debug(f"{self.device_id} extrinsics calibration: success")

        return CalibrationExtrinsics(rvec=rvec, tvec=tvec)

    def mark_board(self, frame: Frame) -> np.ndarray:
        pattern_size = self._board_params.board_size
        frm = frame.color
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK
        )
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None, flags)
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            debug_frm = frm.copy()
            cv2.drawChessboardCorners(debug_frm, pattern_size, corners_sub, ret)
            frm = debug_frm
        return frm

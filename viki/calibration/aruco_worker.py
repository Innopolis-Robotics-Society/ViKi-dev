import cv2
import numpy as np
from typing import List, Sequence

from cv2.typing import MatLike
from viki.capture.base import Frame
from viki.capture.manager import CameraManager
from viki.calibration.models import (
    ArucoBoardParameters,
    BoardParameters,
    ArucoCalibrationSample,
    CalibrationSample,
    CalibrationIntrinsics,
    CalibrationExtrinsics,
)
from viki.calibration.worker import _CalibrationWorker


class ArucoWorker(_CalibrationWorker):
    def __init__(
        self,
        mgr: CameraManager,
        device_id: str,
        aruco_board_params: ArucoBoardParameters,
    ):
        super().__init__(mgr, device_id, aruco_board_params)
        self.device_id = device_id

        dict_id = aruco_board_params.aruco_dict
        self.dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        self.aruco_detector = cv2.aruco.ArucoDetector(self.dictionary)

        # Create the ChArUco board object once (reused for all detections)
        self.board = cv2.aruco.CharucoBoard(
            aruco_board_params.board_size,
            aruco_board_params.square_size,
            aruco_board_params.marker_size,
            self.dictionary,
        )

        self.detector = cv2.aruco.CharucoDetector(self.board)

    def set_board_params(self, board_params: BoardParameters) -> None:
        # super().set_board_params(board_params)
        if not isinstance(board_params, ArucoBoardParameters):
            return
        with self._lock:
            self.board = cv2.aruco.CharucoBoard(
                board_params.board_size,
                board_params.square_size,
                board_params.marker_size,
                self.dictionary,
            )
            self.detector = cv2.aruco.CharucoDetector(self.board)

    def add_sample(self, frame: Frame) -> None:
        gray = cv2.cvtColor(frame.color, cv2.COLOR_BGR2GRAY)

        # Debug: detect markers separately to see what's actually visible
        markers_raw, ids_raw, _ = self.aruco_detector.detectMarkers(gray)
        if ids_raw is not None:
            self._logger.debug(f"{self.device_id} markers visible: {len(ids_raw)}")
        else:
            self._logger.debug(f"{self.device_id} no markers visible at all")

        # 1. Detect ArUco markers
        corners, c_ids, markers, m_ids = self.detector.detectBoard(gray)

        if m_ids is None or len(m_ids) == 0:
            self._logger.debug(
                f"{self.device_id} add_sample: detectBoard failed to find markers. Board size: {self.board_params.board_size}. "
                f"Raw markers found: {len(ids_raw) if ids_raw is not None else 0}"
            )
            return
        if c_ids is None or len(c_ids) == 0:
            self._logger.debug(
                f"{self.device_id} add_sample: detectBoard found markers but no corners. Board size: {self.board_params.board_size}"
            )
            return

        self._logger.debug(
            f"{self.device_id} add_sample: success. Corners: {len(corners)}, Markers: {len(m_ids)}"
        )

        h, w = frame.color.shape[:2]

        # Store the detected corners and IDs inside the sample
        sample = ArucoCalibrationSample(
            frame=frame,
            corners=corners,
            resolution=(w, h),
            board_params=self.board_params,
            markers=markers,
            c_ids=c_ids,
            m_ids=m_ids,
        )

        with self._lock:
            self._samples.append(sample)

        self._logger.debug(
            f"{self.device_id} add_sample: success)"  # (ids: {corners, c_ids})"
        )

    def intrinsics_calibration(
        self, samples: List[CalibrationSample] | None = None
    ) -> CalibrationIntrinsics:

        if samples is None:
            samples = self._samples
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
        all_charuco_corners = []
        all_charuco_ids = []

        for sample in samples:
            if not type(sample) is ArucoCalibrationSample:
                continue
            if (
                sample.corners is not None
                and sample.c_ids is not None
                and len(sample.corners) > 8
            ):
                all_charuco_corners.append(sample.corners)
                all_charuco_ids.append(sample.c_ids)

        if len(all_charuco_corners) < 20:
            msg = (
                f"{self.device_id} intrinsics: not enough valid CharUco "
                f"samples ({len(all_charuco_corners)})"
            )
            self._logger.debug(msg)
            raise RuntimeError(msg)

        try:
            ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                all_charuco_corners,
                all_charuco_ids,
                self.board,
                (w, h),
                None,  # pyright: ignore
                None,  # pyright: ignore
            )
        except AttributeError:
            # Fallback: Use cv2.calibrateCamera
            all_obj_points = []
            all_img_points = []
            board_corners_3d = self.board.getChessboardCorners()

            for corners, ids in zip(all_charuco_corners, all_charuco_ids):
                obj_pts = board_corners_3d[ids]
                all_obj_points.append(obj_pts)
                all_img_points.append(corners)

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                all_obj_points, all_img_points, (w, h), None, None
            )

        if not ret:
            msg = f"{self.device_id} intrinsics: calibration failed"
            self._logger.debug(msg)
            raise RuntimeError(msg)

        self._logger.debug(
            f"{self.device_id} intrinsics: success (RMS error: {ret:.3f})"
        )
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
                msg = f"{self.device_id} extrinsics: no sample available"
                self._logger.debug(msg)
                raise RuntimeError(msg)
            sample = self._samples[-1]
        if not type(sample) is ArucoCalibrationSample:
            msg = f"ArucoWorker extrinsics_calibration: sample is not CharUco sample"
            self._logger.debug(msg)
            raise RuntimeError(msg)
        camera_matrix = intrinsics.camera_matrix
        dist_coeffs = intrinsics.dist_coeffs
        if sample.corners is None:
            msg = f"ArucoWorker extrinsics_calibration: corners are None"
            self._logger.debug(msg)
            raise RuntimeError(msg)

        try:
            ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                sample.corners,
                sample.c_ids,
                self.board,
                camera_matrix,
                dist_coeffs,
                None,  # pyright: ignore
                None,  # pyright: ignore,
            )
        except AttributeError:
            all_corners_3d = self.board.getChessboardCorners()
            object_points = all_corners_3d[sample.c_ids].astype(np.float32)
            image_points = sample.corners.astype(np.float32)

            ret, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        if not ret:
            msg = f"{self.device_id} extrinsics: pose estimation failed"
            self._logger.debug(msg)
            raise RuntimeError(msg)

        # Transform from OpenCV world (Y-down, Z-down) to real world (Y-up, Z-up)
        # This is a 180-degree rotation around the X-axis
        R, _ = cv2.Rodrigues(rvec)
        rx180 = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ], dtype=np.float32)
        R_world = R @ rx180
        rvec_world, _ = cv2.Rodrigues(R_world)

        self._logger.debug(f"{self.device_id} extrinsics: success")
        return CalibrationExtrinsics(rvec=rvec_world, tvec=tvec)

    def mark_board(self, frame: Frame) -> np.ndarray:
        gray = cv2.cvtColor(frame.color, cv2.COLOR_BGR2GRAY)
        markers_raw, ids_raw, _ = self.aruco_detector.detectMarkers(gray)
        if ids_raw is None:
            return frame.color
        corners, c_ids, markers, m_ids = self.detector.detectBoard(gray)
        if m_ids is None or len(m_ids) == 0 or c_ids is None or len(c_ids) == 0:
            return frame.color
        debug_img = frame.color.copy()
        try:
            cv2.aruco.drawDetectedMarkers(
                debug_img, markers, m_ids, borderColor=(0, 255, 0)
            )
            corners_pts = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
            corner_ids = np.asarray(c_ids, dtype=np.int32).reshape(-1, 1)
            cv2.aruco.drawDetectedCornersCharuco(
                debug_img, corners_pts, corner_ids, (0, 0, 255)
            )
        except:
            pass
        return debug_img

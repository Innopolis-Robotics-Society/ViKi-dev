import cv2
from typing import List, Sequence

from cv2.typing import MatLike
from viki.capture.base import Frame
from viki.capture.manager import CameraManager
from viki.calibration.models import (
    ArucoBoardParameters,
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

        # Create the ChArUco board object once (reused for all detections)
        self.board = cv2.aruco.CharucoBoard(
            aruco_board_params.board_size,
            aruco_board_params.square_size,
            aruco_board_params.marker_size,
            self.dictionary,
        )

        self.detector = cv2.aruco.CharucoDetector(self.board)

    def set_board_params(self, board_params: ArucoBoardParameters) -> None:
        super().set_board_params(board_params)
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
        markers_raw, ids_raw, _ = cv2.aruco.detectMarkers(gray, self.dictionary)
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
        
        self._logger.debug(f"{self.device_id} add_sample: success. Corners: {len(corners)}, Markers: {len(m_ids)}")
        
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
            f"{self.device_id} add_sample: success)" # (ids: {corners, c_ids})"
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
            print(sample.corners)
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

        ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            all_charuco_corners,
            all_charuco_ids,
            self.board,
            (w, h),
            None,  # pyright: ignore
            None,  # pyright: ignore
        )

        if not ret:
            msg = (
                f"{self.device_id} intrinsics: cv2.aruco.calibrateCameraCharuco failed"
            )
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
 
        ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            sample.corners,
            sample.c_ids,
            self.board,
            camera_matrix,
            dist_coeffs,
            None,  # pyright: ignore
            None,  # pyright: ignore
        )

        if not ret:
            msg = f"{self.device_id} extrinsics: cv2.aruco.estimatePoseCharucoBoard failed"
            self._logger.debug(msg)
            raise RuntimeError(msg)

        self._logger.debug(f"{self.device_id} extrinsics: success")
        return CalibrationExtrinsics(rvec=rvec, tvec=tvec)

from dataclasses import dataclass, field
import numpy as np
import cv2
from typing import Tuple
from viki.capture.base import Frame


@dataclass
class CalibrationSample:
    frame: Frame
    corners: np.ndarray
    resolution: Tuple[int, int]
    chessboard_size: Tuple[int, int]
    square_size: float


@dataclass
class CalibrationIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(5))

    @property
    def camera_matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )


@dataclass
class CalibrationExtrinsics:
    rvec: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tvec: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def rotation_matrix(self) -> np.ndarray:
        R, _ = cv2.Rodrigues(self.rvec)
        return R

    @property
    def trasnform_matrix(self) -> np.ndarray:
        R = self.rotation_matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = self.tvec.flatten()
        return T

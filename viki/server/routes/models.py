from typing import Tuple, List
from pydantic import BaseModel
import numpy as np
import cv2


class BoardParametersData(BaseModel):
    board_size: Tuple[int, int]
    square_size: float


class ArucoBoardParametersData(BoardParametersData):
    marker_size: float
    aruco_dict: str


class IntrinsicsResponse(BaseModel):
    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: List[float]

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

    @property
    def dist_coeffs_np(self) -> np.ndarray:
        return np.array(self.dist_coeffs, dtype=np.float32)


class ExtrinsicsResponse(BaseModel):
    device_id: str
    rvec: List[float]
    tvec: List[float]

    @property
    def rvec_np(self) -> np.ndarray:
        return np.array(self.rvec, dtype=np.float32)

    @property
    def tvec_np(self) -> np.ndarray:
        return np.array(self.tvec, dtype=np.float32)

    @property
    def rotation_matrix(self) -> np.ndarray:
        R, _ = cv2.Rodrigues(self.rvec_np)
        return R

    @property
    def trasnform_matrix(self) -> np.ndarray:
        R = self.rotation_matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = self.tvec_np.flatten()
        return T

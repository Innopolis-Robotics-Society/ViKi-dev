"""
viki.config
-----------
Centralised tunables for the ViKi capture server.
Loaded from data/user_configuration.json.
"""

import json
import os
import shutil
from typing import Any
import numpy as np

DEFAULT_CONFIG_PATH = "data/default_configuration.json"
USER_CONFIG_PATH = "data/user_configuration.json"

# Duck variables which LSP can catch and use
INTRINSICS_FILENAME: str
EXTRINSICS_FILENAME: str
DEFAULT_FPS: int
DEFAULT_COLOR_WIDTH: int
DEFAULT_COLOR_HEIGHT: int
DEFAULT_DEPTH_MODE: str
JPEG_QUALITY: int
STREAM_IDLE_SLEEP: float
PLACEHOLDER_SIZE: list[int]
DEFAULT_SYNCHRONIZED_IMAGES_ONLY: bool
FRAME_BUFFER_SIZE: int
RECORD_DEPTH: bool
DEPTH_PROJECTION_DEBUG: bool
SKELETON_DEPTH_SAMP_RADIUS: int
SKELETON_DEPTH_BASE_DIR: str
SKELETON_RECS_DIR: str
SKELETON_SMOOTHED_DIR: str
SKELETON_ENABLE_DEPTH_VALIDATION: bool
SKELETON_DEPTH_SUBTRACT_THRESHOLD: float
HAND_TO_DETECT: str
CALIB_MODE: str
CALIB_BOARD_TYPE: str
BONE_LENGTHS: dict[str, Any]
BONE_TOLERANCE: float
CALIB_CHESS_BOARD_SIZE: list[int]
CALIB_CHESS_SQUARE_SIZE: float
CALIB_ARUCO_BOARD_SIZE: list[int]
CALIB_ARUCO_SQUARE_SIZE: float
CALIB_ARUCO_MARKER_SIZE: float
CALIB_ARUCO_DICT: int
RECORDING_DURATION: int
RECORDING_FPS: int
REALSENSE_RESOLUTIONS: list[str]
REALSENSE_FPS: list[int]
KINECT_RESOLUTIONS: list[str]
KINECT_FPS: list[int]
KINECT_DEPTH_MODES: list[str]
KINECT_DEPTH_MODE_MAX_FPS: dict[str, int]
SKELETON_SAVE_JSON_DEBUG: bool
RETARGET_DEFAULT_ROBOT: str
RETARGET_LANDMARK_SG_WINDOW: int
RETARGET_LANDMARK_SG_POLYORDER: int
RETARGET_IK_POSITION_COST: float
RETARGET_IK_ORIENTATION_COST: float
RETARGET_IK_POSTURE_COST: float
RETARGET_TARGET_MODE: str
RETARGET_IK_SUBSTEPS: int
RETARGET_IK_SOLVER: str
RETARGET_APPROACH_SEC: float
RETARGET_JOINT_SG_WINDOW: int
RETARGET_JOINT_SG_POLYORDER: int
RETARGET_RECENTER_TO_NEUTRAL: bool
RETARGET_WRIST_SCALE: float
RETARGET_BASE_EULER_ANGLES: list[float]
RETARGET_BASE_TRANSLATION: list[float]
MODELS_DIR: str



def euler_to_rotation_matrix(euler_angles_deg: list[float], order: str = 'xyz') -> np.ndarray:
    """Convert Euler angles (degrees) to a rotation matrix."""
    assert len(euler_angles_deg) == 3, "Euler angles must be a list of 3 floats"
    
    roll, pitch, yaw = np.deg2rad(euler_angles_deg)

    # Rotation matrix around X-axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    # Rotation matrix around Y-axis
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    # Rotation matrix around Z-axis
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Combine rotations. Assuming XYZ intrinsic rotations (body-fixed).
    # This means R = R_z(yaw) @ R_y(pitch) @ R_x(roll)
    # If it was extrinsic (fixed-axis), it would be R = R_x(roll) @ R_y(pitch) @ R_z(yaw)
    # The user asked for [X, Y, Z] as [roll, pitch, yaw], which is typically XYZ intrinsic.
    if order.lower() == 'xyz':
        R = Rz @ Ry @ Rx
    elif order.lower() == 'zyx': # Common for aerospace, roll-pitch-yaw
        R = Rx @ Ry @ Rz
    else:
        raise ValueError(f"Unsupported Euler angle order: {order}. Choose 'xyz' or 'zyx'.")

    return R

def _load_config():
    if not os.path.exists(USER_CONFIG_PATH):
        if os.path.exists(DEFAULT_CONFIG_PATH):
            shutil.copy(DEFAULT_CONFIG_PATH, USER_CONFIG_PATH)
        else:
            # Fallback if even default is missing (shouldn't happen with our setup)
            return {}

    with open(USER_CONFIG_PATH, "r") as f:
        return json.load(f)


_config = _load_config()

# We assign these to globals so that 'from viki.config import CONSTANT' still works
globals().update(_config)

# Compute RETARGET_BASE_ROTATION from Euler angles
if "RETARGET_BASE_EULER_ANGLES" in _config:
    RETARGET_BASE_ROTATION = euler_to_rotation_matrix(_config["RETARGET_BASE_EULER_ANGLES"])
    globals()["RETARGET_BASE_ROTATION"] = RETARGET_BASE_ROTATION


# Keep a reference to the paths for the API
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_PATH
USER_CONFIG_PATH = USER_CONFIG_PATH

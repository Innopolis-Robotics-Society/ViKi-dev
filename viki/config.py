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
RETARGET_TRAJECTORY_SCALE: float
SKELETON_COORDINATE_FRAME: str
MODELS_DIR: str


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

# Keep a reference to the paths for the API
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_PATH
USER_CONFIG_PATH = USER_CONFIG_PATH

"""viki.streamlit_app.state
---------------------------
Session bootstrap + config derivation -- the Streamlit equivalent of the React
Zustand store's config/camera slices. ``st.session_state`` is the single source
of truth for the discovered device list, per-card start config and UI toggles.

Kept in its own module (rather than ``app.py``) so ``sections/*`` can import the
derivation helpers without a circular import.

``derive_camera_config`` / ``derive_frontend_config`` and ``FALLBACK_CONFIG`` are
ported from the React ``config.slice.ts``.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from .api import ViKiApi, ViKiApiError

# Hardcoded UI default (React DEFAULT_ARUCO_DICT).
DEFAULT_ARUCO_DICT = "DICT_5X5_50"

# Used if the initial /api/config fetch fails (ported from React FALLBACK_CONFIG).
FALLBACK_CONFIG: dict[str, Any] = {
    "DEFAULT_FPS": 15,
    "DEFAULT_COLOR_WIDTH": 1280,
    "DEFAULT_COLOR_HEIGHT": 720,
    "DEFAULT_DEPTH_MODE": "NFOV_UNBINNED",
    "CALIB_CHESS_BOARD_SIZE": [8, 6],
    "CALIB_CHESS_SQUARE_SIZE": 0.025,
    "CALIB_ARUCO_BOARD_SIZE": [10, 8],
    "CALIB_ARUCO_SQUARE_SIZE": 0.05,
    "CALIB_ARUCO_MARKER_SIZE": 0.035,
    "RECORDING_DURATION": 10.0,
    "RECORDING_FPS": 15,
    "REALSENSE_RESOLUTIONS": ["640x480", "1280x720", "1920x1080"],
    "REALSENSE_FPS": [15, 30],
    "KINECT_RESOLUTIONS": ["1280x720", "1920x1080", "2048x1536"],
    "KINECT_FPS": [5, 15, 30],
    "KINECT_DEPTH_MODES": [
        "NFOV_UNBINNED", "NFOV_2X2BINNED", "WFOV_UNBINNED", "WFOV_2X2BINNED"
    ],
    "KINECT_DEPTH_MODE_MAX_FPS": {"WFOV_UNBINNED": 15},
}

# OpenCV ArUco dictionary names (ported from calibration.types.ts ARUCO_DICTS).
ARUCO_DICTS = [
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
    "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
    "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
    "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL",
]


def _cfg(c: dict, key: str) -> Any:
    return c.get(key, FALLBACK_CONFIG.get(key))


def derive_camera_config(c: dict) -> dict:
    """Per-camera-type control options, ported from React deriveCameraConfig."""
    default_res = f"{_cfg(c, 'DEFAULT_COLOR_WIDTH')}x{_cfg(c, 'DEFAULT_COLOR_HEIGHT')}"
    return {
        "realsense": {
            "resolutions": _cfg(c, "REALSENSE_RESOLUTIONS"),
            "defaultRes": default_res,
            "fps": _cfg(c, "REALSENSE_FPS"),
            "defaultFps": _cfg(c, "DEFAULT_FPS"),
            "depthModes": None,
            "defaultDepth": _cfg(c, "DEFAULT_DEPTH_MODE"),
            "depthModeMaxFps": {},
        },
        "kinect": {
            "resolutions": _cfg(c, "KINECT_RESOLUTIONS"),
            "defaultRes": default_res,
            "fps": _cfg(c, "KINECT_FPS"),
            "defaultFps": _cfg(c, "DEFAULT_FPS"),
            "depthModes": _cfg(c, "KINECT_DEPTH_MODES"),
            "defaultDepth": _cfg(c, "DEFAULT_DEPTH_MODE"),
            "depthModeMaxFps": _cfg(c, "KINECT_DEPTH_MODE_MAX_FPS") or {},
        },
    }


def derive_frontend_config(c: dict) -> dict:
    """Calibration + recording defaults, ported from React deriveFrontendConfig."""
    return {
        "calibration": {
            "chess": {
                "boardSize": _cfg(c, "CALIB_CHESS_BOARD_SIZE"),
                "squareSize": _cfg(c, "CALIB_CHESS_SQUARE_SIZE"),
            },
            "aruco": {
                "boardSize": _cfg(c, "CALIB_ARUCO_BOARD_SIZE"),
                "squareSize": _cfg(c, "CALIB_ARUCO_SQUARE_SIZE"),
                "markerSize": _cfg(c, "CALIB_ARUCO_MARKER_SIZE"),
                "defaultDict": DEFAULT_ARUCO_DICT,
            },
        },
        "recording": {
            "duration": _cfg(c, "RECORDING_DURATION"),
            "fps": _cfg(c, "RECORDING_FPS"),
        },
    }


def default_card_config(type_cfg: dict) -> dict:
    """StartConfig for a fresh card (ports React defaultCardConfig)."""
    w, h = (int(x) for x in str(type_cfg["defaultRes"]).split("x"))
    return {
        "color_width": w,
        "color_height": h,
        "fps": type_cfg["defaultFps"],
        "depth_mode": type_cfg.get("defaultDepth") or "NFOV_UNBINNED",
    }


def get_api() -> ViKiApi:
    if "api" not in st.session_state:
        st.session_state.api = ViKiApi()
    return st.session_state.api


def init_state() -> None:
    """Initialise ``st.session_state`` once. Loads server config (falling back to
    ``FALLBACK_CONFIG`` if the backend is unreachable) and derives the camera /
    calibration defaults, then does one device scan. Never raises."""
    if st.session_state.get("_viki_initialized"):
        return

    api = get_api()

    # Load config (best-effort).
    server_config = FALLBACK_CONFIG
    try:
        server_config = api.get_config()
    except ViKiApiError:
        server_config = FALLBACK_CONFIG

    st.session_state.server_config = server_config
    st.session_state.camera_config = derive_camera_config(server_config)
    st.session_state.frontend_config = derive_frontend_config(server_config)

    # Cameras.
    st.session_state.setdefault("devices", [])
    st.session_state.setdefault("running", {})
    st.session_state.setdefault("card_config", {})
    st.session_state.setdefault("info", {})

    # Config editor text.
    st.session_state.setdefault("config_text", "")

    # Calibration board params (ported defaults from calibration.slice.ts + config).
    fc = st.session_state.frontend_config
    st.session_state.setdefault("board_type", "chess")
    st.session_state.setdefault("board_width", fc["calibration"]["chess"]["boardSize"][0])
    st.session_state.setdefault("board_height", fc["calibration"]["chess"]["boardSize"][1])
    st.session_state.setdefault("square_size", fc["calibration"]["chess"]["squareSize"])
    st.session_state.setdefault("marker_size", fc["calibration"]["aruco"]["markerSize"])
    st.session_state.setdefault("aruco_dict", fc["calibration"]["aruco"]["defaultDict"])
    st.session_state.setdefault("calib_session_started", False)

    # Skeleton.
    st.session_state.setdefault("extrinsics", {})
    st.session_state.setdefault("viz_cam", None)
    st.session_state.setdefault("view_mode", "projections")

    st.session_state._viki_initialized = True

    # First scan (best-effort so the empty state renders on a fresh, camera-less
    # backend without an error).
    scan_devices(quiet=True)


def scan_devices(quiet: bool = False) -> None:
    """Refresh the device list + running state from the backend (ports
    scanDevices). Populates a fresh card config for any newly seen device while
    preserving edits to previously seen ones."""
    api = get_api()
    try:
        data = api.list_devices()
    except ViKiApiError as exc:
        if not quiet:
            st.error(f"Scan failed: {exc}")
        st.session_state.devices = []
        return

    devices: list[dict] = []
    for did in data.get("realsense", []) or []:
        devices.append({"id": did, "type": "realsense"})
    for did in data.get("kinect", []) or []:
        devices.append({"id": did, "type": "kinect"})

    active = data.get("active", []) or []
    cam_cfg = st.session_state.camera_config
    prev_cfg = st.session_state.get("card_config", {})

    running: dict[str, bool] = {}
    card_config: dict[str, dict] = {}
    for d in devices:
        running[d["id"]] = d["id"] in active
        type_cfg = cam_cfg["kinect"] if d["type"] == "kinect" else cam_cfg["realsense"]
        card_config[d["id"]] = prev_cfg.get(d["id"]) or default_card_config(type_cfg)

    st.session_state.devices = devices
    st.session_state.running = running
    st.session_state.card_config = card_config

    # Default viz camera to the first device if unset.
    if st.session_state.get("viz_cam") is None and devices:
        st.session_state.viz_cam = devices[0]["id"]

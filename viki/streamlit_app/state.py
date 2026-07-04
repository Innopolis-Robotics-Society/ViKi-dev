"""viki.streamlit_app.state
---------------------------
Session bootstrap -- the Streamlit equivalent of the React
Zustand store's config/camera slices. ``st.session_state`` is the single source
of truth for the discovered device list, per-card start config and UI toggles.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from .api import ViKiApi, ViKiApiError

# OpenCV ArUco dictionary names.
ARUCO_DICTS = [
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
    "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
    "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
    "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL",
]


def get_api() -> ViKiApi:
    if "api" not in st.session_state:
        st.session_state.api = ViKiApi()
    return st.session_state.api


def init_state() -> None:
    """Initialise ``st.session_state`` once. Loads server config
    and does one device scan. Never raises."""
    if st.session_state.get("_viki_initialized"):
        return

    api = get_api()

    # Load config from backend.
    try:
        server_config = api.get_config()
    except ViKiApiError:
        server_config = {}

    st.session_state.server_config = server_config

    # Cameras.
    st.session_state.setdefault("devices", [])
    st.session_state.setdefault("running", {})
    st.session_state.setdefault("card_config", {})
    st.session_state.setdefault("info", {})

    # Config editor text.
    st.session_state.setdefault("config_text", "")

    # Calibration board params.
    # Use server_config or hardcoded defaults if config is missing.
    sc = server_config
    st.session_state.setdefault("board_type", "chess")
    st.session_state.setdefault("board_width", sc.get("CALIB_CHESS_BOARD_SIZE", [8, 6])[0])
    st.session_state.setdefault("board_height", sc.get("CALIB_CHESS_BOARD_SIZE", [8, 6])[1])
    st.session_state.setdefault("square_size", sc.get("CALIB_CHESS_SQUARE_SIZE", 0.025))
    st.session_state.setdefault("marker_size", sc.get("CALIB_ARUCO_MARKER_SIZE", 0.035))
    st.session_state.setdefault("aruco_dict", sc.get("CALIB_ARUCO_DICT", "DICT_5X5_50"))
    st.session_state.setdefault("calib_session_started", False)

    # Skeleton.
    st.session_state.setdefault("extrinsics", {})
    st.session_state.setdefault("viz_cam", None)
    st.session_state.setdefault("view_mode", "projections")

    st.session_state._viki_initialized = True

    # First scan (best-effort).
    scan_devices(quiet=True)


def scan_devices(quiet: bool = False) -> None:
    """Refresh the device list + running state from the backend.
    Populates a fresh card config for any newly seen device while
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
    for did in data.get("web_camera", []) or []:
        devices.append({"id": did, "type": "web_camera"})

    active = data.get("active", []) or []
    server_config = st.session_state.get("server_config", {})
    prev_cfg = st.session_state.get("card_config", {})

    running: dict[str, bool] = {}
    card_config: dict[str, dict] = {}
    for d in devices:
        did = d["id"]
        running[did] = did in active
        if did not in prev_cfg:
            card_config[did] = {
                "color_width": server_config.get("DEFAULT_COLOR_WIDTH", 1280),
                "color_height": server_config.get("DEFAULT_COLOR_HEIGHT", 720),
                "fps": server_config.get("DEFAULT_FPS", 15),
                "depth_mode": server_config.get("DEFAULT_DEPTH_MODE", "NFOV_UNBINNED"),
            }
        else:
            card_config[did] = prev_cfg[did]

    st.session_state.devices = devices
    st.session_state.running = running
    st.session_state.card_config = card_config

    # Default viz camera to the first device if unset.
    if st.session_state.get("viz_cam") is None and devices:
        st.session_state.viz_cam = devices[0]["id"]

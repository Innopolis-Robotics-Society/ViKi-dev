"""Config tab.

Ports the React ConfigPanel: a raw JSON editor over ``/api/config`` with
Load / Reset to Defaults / Save / Restart Server actions and a Help expander.
``HELP_ITEMS`` is reused verbatim from ConfigPanel.tsx.
"""

from __future__ import annotations

import json

import streamlit as st

from ..api import ViKiApiError
from ..state import (
    get_api,
)

# Ported verbatim from ConfigPanel.tsx HELP_ITEMS.
HELP_ITEMS: list[tuple[str, str]] = [
    ("INTRINSICS_FILENAME", "Path to intrinsics calibration"),
    ("EXTRINSICS_FILENAME", "Path to extrinsics calibration"),
    ("DEFAULT_FPS", "Default frames per second for camera start"),
    ("DEFAULT_COLOR_WIDTH", "Default color image width"),
    ("DEFAULT_COLOR_HEIGHT", "Default color image height"),
    ("DEFAULT_DEPTH_MODE", "Default depth mode (NFOV_UNBINNED recommended)"),
    ("JPEG_QUALITY", "Quality of JPEG encoding (0-100)"),
    ("STREAM_IDLE_SLEEP", "Sleep duration for MJPEG stream generators (default value works well)"),
    ("PLACEHOLDER_SIZE", "Dimensions of the placeholder image when camera not started"),
    ("DEFAULT_SYNCHRONIZED_IMAGES_ONLY", "Only use synchronized frames (not tested with true)"),
    ("FRAME_BUFFER_SIZE", "Number of frames kept in ring buffer per camera"),
    ("RECORD_DEPTH", "Save depth frames as .npy files alongside the rgb-d videos (heavy on the disk)"),
    ("DEPTH_PROJECTION_DEBUG", "Enable debug plotting for depth projection (slows down live skeleton estimation)"),
    ("SKELETON_DEPTH_SAMP_RADIUS", "Radius for depth sampling during fusion (default worked well in testing)"),
    ("SKELETON_DEPTH_BASE_DIR", "Directory for background depth frames"),
    ("SKELETON_ENABLE_DEPTH_VALIDATION", "Toggle depth-based outlier rejection (disabling WILL lead to great instability)"),
    ("SKELETON_DEPTH_SUBTRACT_THRESHOLD", "Min depth diff for object detection (untested outside default value)"),
    ("HAND_TO_DETECT", "Hand side to estimate ('left' or 'right')"),
    ("CALIB_MODE", "Calibration mode ('manual' or 'auto') (auto is deprecated)"),
    ("CALIB_BOARD_TYPE", "Board type ('chess' or 'aruco')"),
    ("BONE_LENGTHS", "Manual bone length overrides {(Parent, Child): length_m} (default uses a moving mean, use this if you're sure you know what you're doing)"),
    ("BONE_TOLERANCE", "Tolerance for kinematic constraints (0.0-1.0) (default works well with mean for lenghts)"),
    ("CALIB_CHESS_BOARD_SIZE", "Default chessboard dimensions [cols, rows]"),
    ("CALIB_CHESS_SQUARE_SIZE", "Default chessboard square size in meters"),
    ("CALIB_ARUCO_BOARD_SIZE", "Default aruco board dimensions [cols, rows]"),
    ("CALIB_ARUCO_SQUARE_SIZE", "Default aruco square size in meters"),
    ("CALIB_ARUCO_MARKER_SIZE", "Default aruco marker size in meters"),
]


def _load_config() -> None:
    api = get_api()
    try:
        config = api.get_config()
        st.session_state.config_text = json.dumps(config, indent=2)
        # Update the shared server config.
        st.session_state.server_config = config
        st.toast("Configuration loaded", icon="✅")
    except ViKiApiError as exc:
        st.error(f"Failed to load config: {exc}")


def _save_config() -> None:
    api = get_api()
    try:
        config = json.loads(st.session_state.config_text)
    except json.JSONDecodeError as exc:
        st.error(f"Invalid JSON: {exc}")
        return
    try:
        api.save_config(config)
        st.toast("Configuration saved", icon="✅")
    except ViKiApiError as exc:
        st.error(f"Failed to save config: {exc}")


def _reset_config() -> None:
    api = get_api()
    try:
        api.reset_config()
        _load_config()
        st.toast("Configuration reset to defaults", icon="✅")
    except ViKiApiError as exc:
        st.error(f"Failed to reset config: {exc}")


def _restart_server() -> None:
    api = get_api()
    api.restart()  # swallows the expected dropped-connection
    st.toast("Restarting server... please wait a few seconds.", icon="⏳")


def render() -> None:
    st.subheader("Server Configuration")

    # Load the config text on first view of the tab.
    if not st.session_state.get("config_text"):
        _load_config()

    st.text_area(
        "Raw configuration (JSON)",
        key="config_text",
        height=420,
    )

    st.caption("Restart is required for changes to take effect.")

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.button("Load", use_container_width=True, on_click=_load_config)
    with b2:
        st.button("Reset to Defaults", use_container_width=True, on_click=_reset_config)
    with b3:
        st.button("Save", type="primary", use_container_width=True, on_click=_save_config)
    with b4:
        st.button("Restart Server", use_container_width=True, on_click=_restart_server)

    with st.expander("❔ Help"):
        for key, desc in HELP_ITEMS:
            st.markdown(f"**{key}**: {desc}")
        st.markdown(
            "**CALIB_ARUCO_DICT**: OpenCV Aruco dictionary ID (our main board has "
            "40 markers total, each 5x5, so DICT_5X5_50 is good)"
        )
        st.markdown("---")
        st.markdown("**Load**: pulls the user configuration from backend")
        st.markdown("**Reset to Defaults**: loads the default parameters for the system")
        st.markdown("**Save**: save the current configuration to the server")
        st.markdown("**Restart Server**: restarts docker to apply all the configuration changes")

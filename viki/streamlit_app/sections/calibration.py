"""Calibration tab.

Ports the React CalibrationPanel + calibration.slice.ts: board params
(chess/ChArUco), per-device live preview streams with sample counts, and the
Capture / Extrinsics / Clear actions.

The calibration session (the backend precondition for ``/api/calibration/capture``)
is applied **automatically**: it is (re)started on tab render when cameras are
present, and re-applied whenever any board parameter changes -- there is no manual
sync button. Applying does reset -> sync -> start-per-device so new params take
effect on freshly created workers. Sample counts are polled with
``@st.fragment(run_every="1s")`` so only the counts refresh -- the preview
``<img>`` streams are rendered outside the fragment and keep running untouched.
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from ..api import ViKiApiError
from ..embeds import mjpeg_img
from ..settings import browser_url
from ..state import ARUCO_DICTS, get_api


def _preview_url(device_id: str) -> str:
    # React uses colorStreamUrl(id, false) -> undistort disabled, no cache-buster
    # needed since we render it once per full rerun.
    return browser_url(f"/api/cameras/{device_id}/stream?undistort=false")


def _build_params() -> dict:
    base = {
        "board_size": [int(st.session_state.board_width), int(st.session_state.board_height)],
        "square_size": float(st.session_state.square_size),
    }
    if st.session_state.board_type == "aruco":
        base["marker_size"] = float(st.session_state.marker_size)
        base["aruco_dict"] = st.session_state.aruco_dict
    return base


def _apply_calibration_session(*, toast: bool = False) -> None:
    """(Re)apply the calibration session for the current board params.

    Does reset -> sync -> start-per-device so freshly created workers pick up the
    latest params. Called automatically on tab render and on every board-param
    change; no-op when no devices are present. Errors are always surfaced; the
    success toast is quiet by default so param edits don't spam notifications.
    """
    devices = st.session_state.get("devices", [])
    if not devices:
        return
    api = get_api()
    params = _build_params()
    board_type = st.session_state.board_type
    try:
        api.calibration_reset()
        api.calibration_sync(board_type, params)
        for d in devices:
            if board_type == "chess":
                api.calibration_start_chess(d["id"], params)
            else:
                api.calibration_start_aruco(d["id"], params)
        st.session_state.calib_session_started = True
        if toast:
            st.toast("Calibration session started", icon="✅")
    except ViKiApiError as exc:
        st.error(f"Calibration session start failed: {exc}")


def _capture_sample() -> None:
    api = get_api()
    try:
        api.calibration_capture_all()
        st.toast("Sample captured", icon="✅")
    except ViKiApiError as exc:
        st.error(f"Capture failed: {exc}")


def _calibrate_extrinsics() -> None:
    api = get_api()
    try:
        results = api.calibration_extrinsics()
        extrinsics = {
            r["device_id"]: {"rvec": r["rvec"], "tvec": r["tvec"]} for r in results
        }
        st.session_state.extrinsics = extrinsics
        st.toast("Extrinsics calibration successful", icon="✅")
    except ViKiApiError as exc:
        st.error(f"Extrinsics calibration failed: {exc}")


def _clear_samples() -> None:
    api = get_api()
    try:
        for d in st.session_state.get("devices", []):
            api.calibration_clear(d["id"])
        st.toast("Calibration samples cleared")
    except ViKiApiError as exc:
        st.error(f"Clear failed: {exc}")


@st.fragment(run_every="1s")
def _sample_counts() -> None:
    """Poll per-device sample counts once a second (ports updateStatus)."""
    api = get_api()
    devices = st.session_state.get("devices", [])
    if not devices:
        return
    cols = st.columns(len(devices))
    for col, d in zip(cols, devices):
        with col:
            try:
                data = api.calibration_status(d["id"])
                count = data.get("samples_count", 0)
                started = data.get("started", False)
                text = f"{count} samples" if started else "0 samples"
                st.metric(d["id"], text)
            except ViKiApiError:
                st.metric(d["id"], "error")


def _on_board_type_change() -> None:
    """Repopulate board dimension fields from config on type switch (ports the
    React initBoardFields called by setBoardType), then re-apply the session so
    the new board type takes effect immediately."""
    calib = st.session_state.frontend_config["calibration"]
    if st.session_state.board_type == "chess":
        st.session_state.board_width = calib["chess"]["boardSize"][0]
        st.session_state.board_height = calib["chess"]["boardSize"][1]
        st.session_state.square_size = calib["chess"]["squareSize"]
    else:
        st.session_state.board_width = calib["aruco"]["boardSize"][0]
        st.session_state.board_height = calib["aruco"]["boardSize"][1]
        st.session_state.square_size = calib["aruco"]["squareSize"]
        st.session_state.marker_size = calib["aruco"]["markerSize"]
    st.session_state.aruco_dict = calib["aruco"]["defaultDict"]
    # Board type change is a meaningful switch -> toast so the user sees it.
    _apply_calibration_session(toast=True)


def _on_board_param_change() -> None:
    """Re-apply the calibration session when a board dimension/param changes so
    updated params reach the backend workers without a manual sync click."""
    _apply_calibration_session()


def _board_params() -> None:
    st.markdown("**Board parameters**")
    st.selectbox(
        "Board type",
        ["chess", "aruco"],
        format_func=lambda v: "Chessboard" if v == "chess" else "ChArUco",
        key="board_type",
        on_change=_on_board_type_change,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input(
            "Board width", min_value=1, step=1, key="board_width",
            on_change=_on_board_param_change,
        )
    with c2:
        st.number_input(
            "Board height", min_value=1, step=1, key="board_height",
            on_change=_on_board_param_change,
        )
    with c3:
        st.number_input(
            "Square size (m)", min_value=0.001, step=0.001, format="%.3f",
            key="square_size", on_change=_on_board_param_change,
        )

    if st.session_state.board_type == "aruco":
        a1, a2 = st.columns(2)
        with a1:
            st.number_input(
                "Marker size (m)", min_value=0.001, step=0.001, format="%.3f",
                key="marker_size", on_change=_on_board_param_change,
            )
        with a2:
            # Value comes from session_state (seeded in init_state); passing an
            # index as well would conflict with the pre-set key.
            if st.session_state.aruco_dict not in ARUCO_DICTS:
                st.session_state.aruco_dict = ARUCO_DICTS[0]
            st.selectbox(
                "Dictionary", ARUCO_DICTS, key="aruco_dict",
                on_change=_on_board_param_change,
            )


def render() -> None:
    st.subheader("Camera Calibration")
    devices = st.session_state.get("devices", [])

    # Auto-start the session once when cameras are present so "Capture Sample"
    # works without any click. Re-arm when the device set changes so a rescan
    # re-applies; guarded so we don't reset/restart on every rerun.
    device_ids = tuple(d["id"] for d in devices)
    if devices and st.session_state.get("calib_session_devices") != device_ids:
        _apply_calibration_session()
        st.session_state.calib_session_devices = device_ids
    elif not devices:
        st.session_state.calib_session_devices = None
        st.session_state.calib_session_started = False

    # --- preview streams (rendered once; keep running across fragment polls) ---
    if not devices:
        st.info("No cameras detected. Scan and start cameras before calibrating.")
    else:
        cols = st.columns(min(len(devices), 3))
        for i, d in enumerate(devices):
            with cols[i % len(cols)]:
                if st.session_state.running.get(d["id"]):
                    components.html(
                        mjpeg_img(_preview_url(d["id"]), height=200, label=d["id"]),
                        height=220,
                    )
                else:
                    st.caption(f"{d['id']} (not running)")

        st.markdown("**Samples collected**")
        _sample_counts()

    st.divider()

    _board_params()
    st.caption("Board parameters sync automatically when changed -- no manual step needed.")

    st.divider()

    b1, b2 = st.columns(2)
    with b1:
        st.button(
            "📸 Capture Sample",
            type="primary",
            use_container_width=True,
            on_click=_capture_sample,
            disabled=not devices,
        )
    with b2:
        st.button(
            "🗑 Clear Samples",
            use_container_width=True,
            on_click=_clear_samples,
            disabled=not devices,
        )

    st.button(
        "Calibrate Extrinsics",
        type="primary",
        use_container_width=True,
        on_click=_calibrate_extrinsics,
        disabled=not devices,
    )

    st.caption("Extrinsic calibration requires at least 1 sample per camera.")

    if st.session_state.get("extrinsics"):
        with st.expander("Computed extrinsics"):
            st.json(st.session_state.extrinsics)

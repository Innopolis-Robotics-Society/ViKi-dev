"""Calibration tab.

Ports the React CalibrationPanel + calibration.slice.ts: board params
(chess/ChArUco), per-device live preview streams with sample counts, and the
Sync / Capture / Intrinsics / Extrinsics / Clear actions.

The React panel auto-starts a session when opened; Streamlit has no open/close
lifecycle, so the session is started explicitly via **Sync & Start Session**
(which also satisfies the backend's "Sync Parameters first" precondition on
``/api/calibration/capture``). Sample counts are polled with
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


def _sync_and_start() -> None:
    api = get_api()
    devices = st.session_state.get("devices", [])
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


def _calibrate_intrinsics() -> None:
    api = get_api()
    ok, fail = [], []
    for d in st.session_state.get("devices", []):
        try:
            api.calibration_intrinsics(d["id"])
            ok.append(d["id"])
        except ViKiApiError as exc:
            fail.append(f"{d['id']}: {exc}")
    if ok:
        st.toast(f"Intrinsics calibrated: {', '.join(ok)}", icon="✅")
    for f in fail:
        st.error(f"Intrinsics failed for {f}")


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
    """Repopulate board dimension fields from config on type switch."""
    sc = st.session_state.get("server_config", {})
    if st.session_state.board_type == "chess":
        st.session_state.board_width = sc.get("CALIB_CHESS_BOARD_SIZE", [8, 6])[0]
        st.session_state.board_height = sc.get("CALIB_CHESS_BOARD_SIZE", [8, 6])[1]
        st.session_state.square_size = sc.get("CALIB_CHESS_SQUARE_SIZE", 0.025)
    else:
        st.session_state.board_width = sc.get("CALIB_ARUCO_BOARD_SIZE", [8, 10])[0]
        st.session_state.board_height = sc.get("CALIB_ARUCO_BOARD_SIZE", [8, 10])[1]
        st.session_state.square_size = sc.get("CALIB_ARUCO_SQUARE_SIZE", 0.05)
        st.session_state.marker_size = sc.get("CALIB_ARUCO_MARKER_SIZE", 0.035)
    st.session_state.aruco_dict = sc.get("CALIB_ARUCO_DICT", "DICT_5X5_50")


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
        st.number_input("Board width", min_value=1, step=1, key="board_width")
    with c2:
        st.number_input("Board height", min_value=1, step=1, key="board_height")
    with c3:
        st.number_input("Square size (m)", min_value=0.001, step=0.001, format="%.3f", key="square_size")

    if st.session_state.board_type == "aruco":
        a1, a2 = st.columns(2)
        with a1:
            st.number_input(
                "Marker size (m)", min_value=0.001, step=0.001, format="%.3f", key="marker_size"
            )
        with a2:
            # Value comes from session_state (seeded in init_state); passing an
            # index as well would conflict with the pre-set key.
            if st.session_state.aruco_dict not in ARUCO_DICTS:
                st.session_state.aruco_dict = ARUCO_DICTS[0]
            st.selectbox("Dictionary", ARUCO_DICTS, key="aruco_dict")


def render() -> None:
    st.subheader("Camera Calibration")
    devices = st.session_state.get("devices", [])

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
    st.caption("Board parameters are applied when you click Sync & Start Session.")

    st.divider()

    b1, b2, b3 = st.columns(3)
    with b1:
        st.button(
            "🔄 Sync & Start Session",
            use_container_width=True,
            on_click=_sync_and_start,
            disabled=not devices,
        )
    with b2:
        st.button(
            "📸 Capture Sample",
            type="primary",
            use_container_width=True,
            on_click=_capture_sample,
            disabled=not devices,
        )
    with b3:
        st.button(
            "🗑 Clear Samples",
            use_container_width=True,
            on_click=_clear_samples,
            disabled=not devices,
        )

    c1, c2 = st.columns(2)
    with c1:
        st.button(
            "Calibrate Intrinsics",
            use_container_width=True,
            on_click=_calibrate_intrinsics,
            disabled=not devices,
        )
    with c2:
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

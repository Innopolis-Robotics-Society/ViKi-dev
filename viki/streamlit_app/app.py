"""viki.streamlit_app.app
--------------------------
Entry point for the ViKi Streamlit frontend.

Layout: a topbar row (server-alive indicator, Scan devices, Start all, Stop all,
Record RGB-D) followed by four tabs (Cameras / Calibration / Skeleton / Config).
All backend I/O goes through :class:`~viki.streamlit_app.api.ViKiApi`; every call
is wrapped so an unreachable/erroring backend surfaces a message instead of
crashing the page.

Run with:
    streamlit run viki/streamlit_app/app.py --server.port 8501 \
        --server.address 0.0.0.0 --server.headless true
"""

from __future__ import annotations

import streamlit as st

# `streamlit run app.py` executes this file as a top-level script (no package
# context), so relative imports fail here. Use absolute package imports — the
# container sets PYTHONPATH=/app, making `viki.streamlit_app` importable. The
# submodules below are then loaded as package members, so their own relative
# imports work.
from viki.streamlit_app.api import ViKiApiError
from viki.streamlit_app.sections import calibration, cameras, config_panel, skeleton
from viki.streamlit_app.state import get_api, init_state, scan_devices


# --- topbar actions ---------------------------------------------------------
def _start_all() -> None:
    api = get_api()
    for d in st.session_state.get("devices", []):
        if not st.session_state.running.get(d["id"]):
            cfg = st.session_state.card_config.get(d["id"])
            if not cfg:
                continue
            try:
                api.start_camera(d["id"], cfg)
                st.session_state.running[d["id"]] = True
            except ViKiApiError as exc:
                st.error(f"Failed to start {d['id']}: {exc}")


def _stop_all() -> None:
    api = get_api()
    for d in st.session_state.get("devices", []):
        if st.session_state.running.get(d["id"]):
            try:
                api.stop_camera(d["id"])
            except ViKiApiError as exc:
                st.error(f"Failed to stop {d['id']}: {exc}")
            st.session_state.running[d["id"]] = False
            st.session_state.info[d["id"]] = None


def _record_rgbd() -> None:
    api = get_api()
    sc = st.session_state.get("server_config", {})
    duration = sc.get("RECORDING_DURATION", 10.0)
    fps = sc.get("RECORDING_FPS", 15)
    try:
        res = api.record_start(duration=duration, fps=fps)
        st.toast(f"RGB-D recording started ({duration}s)", icon="🔴")
        if isinstance(res, dict) and res.get("status"):
            st.session_state["_last_record_status"] = res["status"]
    except ViKiApiError as exc:
        st.error(f"Recording failed: {exc}")


@st.fragment(run_every="5s")
def _server_indicator() -> None:
    alive = get_api().is_alive()
    if alive:
        st.markdown("🟢 **server online**")
    else:
        st.markdown("🔴 **server offline**")


def _topbar() -> None:
    st.title("ViKi — capture server")
    cols = st.columns([1.4, 1.2, 1, 1, 1.4])
    with cols[0]:
        _server_indicator()
    with cols[1]:
        st.button(
            "⟳ Scan devices",
            type="primary",
            use_container_width=True,
            on_click=scan_devices,
        )
    with cols[2]:
        st.button(
            "▶ Start all",
            use_container_width=True,
            on_click=_start_all,
            disabled=not st.session_state.get("devices"),
        )
    with cols[3]:
        st.button(
            "■ Stop all",
            use_container_width=True,
            on_click=_stop_all,
            disabled=not st.session_state.get("devices"),
        )
    with cols[4]:
        st.button(
            "🔴 Record RGB-D",
            use_container_width=True,
            on_click=_record_rgbd,
            disabled=not st.session_state.get("devices"),
        )


def main() -> None:
    st.set_page_config(page_title="ViKi capture server", layout="wide")
    init_state()
    _topbar()
    st.divider()

    tab_cameras, tab_calib, tab_skeleton, tab_config = st.tabs(
        ["Cameras", "Calibration", "Skeleton", "Config"]
    )
    with tab_cameras:
        cameras.render()
    with tab_calib:
        calibration.render()
    with tab_skeleton:
        skeleton.render()
    with tab_config:
        config_panel.render()


main()

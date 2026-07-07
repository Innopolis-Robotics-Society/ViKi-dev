"""Skeleton tab.

Ports the React SkeletonPanel: Start/Stop estimation, Record/Stop recording, the
embedded live 3D canvas, the per-device detected status and the joint table.

The detected status and joint table are driven by the ``/api/skeleton/stream``
WebSocket, which is consumed *in the browser* by the embedded
:func:`~viki.streamlit_app.embeds.skeleton_canvas` component -- that data does not
exist on any HTTP endpoint, so it is rendered inside the component (browser JS),
not server-side. The ``@st.fragment(run_every="1s")`` poll of
``/api/skeleton/status`` drives the server-side Running/Idle + Recording text.
The view-mode toggle and visualize-camera selector are also inside the component
so switching them does not trigger a Streamlit rerun (which would reset the WS).
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from ..api import ViKiApiError
from ..embeds import skeleton_canvas
from ..settings import ws_url
from ..state import get_api


def _toggle_estimation(enable: bool) -> None:
    api = get_api()
    try:
        api.skeleton_toggle(enable)
        st.toast(f"Estimation {'ON' if enable else 'OFF'}", icon="✅")
    except ViKiApiError as exc:
        st.error(f"Estimation toggle failed: {exc}")


def _toggle_recording(enable: bool) -> None:
    api = get_api()
    try:
        api.skeleton_record(enable)
        st.toast(f"Recording {'ON' if enable else 'OFF'}", icon="✅")
    except ViKiApiError as exc:
        st.error(f"Recording toggle failed: {exc}")


@st.fragment(run_every="1s")
def _status() -> None:
    """Poll /api/skeleton/status for the Running/Idle + Recording indicator."""
    api = get_api()
    try:
        data = api.skeleton_status()
    except ViKiApiError:
        st.warning("Skeleton status unavailable (backend unreachable).")
        return
    enabled = data.get("enabled", False)
    recording = data.get("recording", False)
    bits = ["🟢 Running" if enabled else "⚪ Idle"]
    if recording:
        bits.append("🔴 RECORDING")
    st.markdown("**Status:** " + "  ·  ".join(bits))


def render() -> None:
    st.subheader("3D Skeleton Estimation")
    devices = st.session_state.get("devices", [])

    main_col, side_col = st.columns([3, 1])

    with main_col:
        _status()

        st.caption(
            "The live pose, per-camera detected state and joint table below update in "
            "real time from the skeleton WebSocket (in your browser)."
        )

        html = skeleton_canvas(
            ws_url=ws_url("/api/skeleton/stream"),
            devices=devices,
            extrinsics=st.session_state.get("extrinsics", {}),
            viz_cam=st.session_state.get("viz_cam"),
            view_mode=st.session_state.get("view_mode", "projections"),
            height=380,
        )
        # Canvas + detection list + joint table + WS all live inside this component.
        components.html(html, height=740, scrolling=True)

    with side_col:
        st.button(
            "▶ Start Estimation",
            type="primary",
            use_container_width=True,
            on_click=_toggle_estimation,
            args=(True,),
        )
        st.button(
            "■ Stop Estimation",
            use_container_width=True,
            on_click=_toggle_estimation,
            args=(False,),
        )
        st.divider()
        st.button(
            "🔴 Record",
            type="primary",
            use_container_width=True,
            on_click=_toggle_recording,
            args=(True,),
        )
        st.button(
            "⏹ Stop Recording",
            use_container_width=True,
            on_click=_toggle_recording,
            args=(False,),
        )

    st.divider()
    st.subheader("Smoothing")

    api = get_api()
    try:
        recs = api.skeleton_list_recordings()
    except ViKiApiError:
        recs = []

    if recs:
        smooth_rec = st.selectbox("Recording to smooth", recs, key="smooth_rec_select")
        col_a, col_b = st.columns([1, 1])
        with col_a:
            win = st.number_input("Window", min_value=3, max_value=31, value=7, step=2, key="smooth_win")
        with col_b:
            order = st.number_input("Polyorder", min_value=1, max_value=5, value=2, key="smooth_order")

        if st.button("Smooth", type="primary", use_container_width=True):
            try:
                res = api.smooth_recording(smooth_rec, window_length=win, polyorder=order)
                cln_name = Path(res["path"]).name
                st.session_state["_last_smoothed"] = cln_name
                st.success(f"Smoothed → {cln_name}")
            except ViKiApiError as exc:
                st.error(f"Smoothing failed: {exc}")

        last = st.session_state.get("_last_smoothed")
        if last:
            st.image(api.smooth_plot_url(last), use_container_width=True)
    else:
        st.caption("No recordings found. Record a skeleton session first.")

    if not devices:
        st.info(
            "No cameras detected. The skeleton view will show data once cameras "
            "are running and estimation is started."
        )

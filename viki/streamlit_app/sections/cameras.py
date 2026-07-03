"""Cameras tab -- one card per discovered device.

Ports the React CameraCard/CameraGrid: per-camera resolution/fps/depth selects
(single source of truth in ``st.session_state.card_config``), Start/Stop, an info
readout, and the embedded colour + depth MJPEG streams (browser-pulled, only
while running). Empty-state "No cameras detected" when none.
"""

from __future__ import annotations

import time

import streamlit as st
import streamlit.components.v1 as components

from ..api import ViKiApiError
from ..embeds import mjpeg_img
from ..settings import browser_url
from ..state import get_api

_INFINITY = float("inf")


def _color_stream_url(device_id: str, undistort: bool = True) -> str:
    return browser_url(
        f"/api/cameras/{device_id}/stream?undistort={str(undistort).lower()}&t={int(time.time()*1000)}"
    )


def _depth_stream_url(device_id: str) -> str:
    return browser_url(f"/api/cameras/{device_id}/depth?t={int(time.time()*1000)}")


def _start_camera(device_id: str) -> None:
    api = get_api()
    cfg = st.session_state.card_config.get(device_id)
    if not cfg:
        return
    try:
        api.start_camera(device_id, cfg)
        st.session_state.running[device_id] = True
        st.toast(f"{device_id} started", icon="✅")
    except ViKiApiError as exc:
        st.error(f"Failed to start {device_id}: {exc}")


def _stop_camera(device_id: str) -> None:
    api = get_api()
    try:
        api.stop_camera(device_id)
        st.toast(f"{device_id} stopped")
    except ViKiApiError as exc:
        st.error(f"Failed to stop {device_id}: {exc}")
    st.session_state.running[device_id] = False
    st.session_state.info[device_id] = None


def _render_card(device: dict) -> None:
    device_id = device["id"]
    dtype = device["type"]
    running = st.session_state.running.get(device_id, False)
    cfg = st.session_state.card_config.get(device_id)
    if not cfg:
        return

    type_cfg = (
        st.session_state.camera_config["kinect"]
        if dtype == "kinect"
        else st.session_state.camera_config["realsense"]
    )

    with st.container(border=True):
        dot = "🟢" if running else "⚪"
        live = "  ·  **LIVE**" if running else ""
        st.markdown(f"{dot} **{device_id}**  `{dtype}`{live}")

        # --- controls ---
        c1, c2 = st.columns(2)
        with c1:
            st.button(
                "▶ Start",
                key=f"start_{device_id}",
                disabled=running,
                use_container_width=True,
                on_click=_start_camera,
                args=(device_id,),
            )
        with c2:
            st.button(
                "■ Stop",
                key=f"stop_{device_id}",
                disabled=not running,
                use_container_width=True,
                on_click=_stop_camera,
                args=(device_id,),
            )

        depth_max_fps = type_cfg.get("depthModeMaxFps") or {}
        has_depth = bool(type_cfg.get("depthModes"))
        cols = st.columns(3 if has_depth else 2)

        # resolution
        resolutions = type_cfg["resolutions"]
        res_value = f"{cfg['color_width']}x{cfg['color_height']}"
        if res_value not in resolutions:
            resolutions = [res_value, *resolutions]
        with cols[0]:
            new_res = st.selectbox(
                "res",
                resolutions,
                index=resolutions.index(res_value),
                key=f"res_{device_id}",
                disabled=running,
            )

        # depth mode (rendered before fps so the fps cap can react to it)
        new_depth = cfg["depth_mode"]
        if has_depth:
            depth_modes = type_cfg["depthModes"]
            with cols[2]:
                new_depth = st.selectbox(
                    "depth mode",
                    depth_modes,
                    index=depth_modes.index(new_depth) if new_depth in depth_modes else 0,
                    key=f"depth_{device_id}",
                    disabled=running,
                )

        # fps: forbid values above the depth-mode cap (backend rejects
        # WFOV_UNBINNED @ 30fps). Key includes the depth mode so the widget
        # re-instantiates cleanly when the option set shrinks/grows.
        cap = depth_max_fps.get(new_depth, _INFINITY)
        fps_options = [f for f in type_cfg["fps"] if f <= cap] or [type_cfg["fps"][0]]
        cur_fps = cfg["fps"] if cfg["fps"] in fps_options else fps_options[-1]
        with cols[1]:
            new_fps = st.selectbox(
                "fps",
                fps_options,
                index=fps_options.index(cur_fps),
                key=f"fps_{device_id}_{new_depth}",
                disabled=running,
            )

        # Persist control edits back into the single-source card config.
        w, h = (int(x) for x in new_res.split("x"))
        st.session_state.card_config[device_id] = {
            "color_width": w,
            "color_height": h,
            "fps": new_fps,
            "depth_mode": new_depth,
        }

        # --- streams (only while running) ---
        if running:
            info = _fetch_info(device_id)
            sc1, sc2 = st.columns(2)
            with sc1:
                components.html(
                    mjpeg_img(_color_stream_url(device_id), height=260, label="COLOR"),
                    height=280,
                )
            with sc2:
                components.html(
                    mjpeg_img(_depth_stream_url(device_id), height=260, label="DEPTH"),
                    height=280,
                )

            # --- info footer ---
            if info and info.get("color_shape"):
                cs = info["color_shape"]
                parts = [f"color: {cs[1]}×{cs[0]}"]
                if info.get("depth_shape"):
                    ds = info["depth_shape"]
                    parts.append(f"depth: {ds[1]}×{ds[0]}")
                ci = info.get("color_intrinsics")
                if ci:
                    parts.append(f"fx={ci['fx']:.1f} fy={ci['fy']:.1f}")
                st.caption("  ·  ".join(parts))
            else:
                st.caption("—")
        else:
            st.caption("Camera stopped — start it to view the live streams.")


def _fetch_info(device_id: str) -> dict | None:
    api = get_api()
    try:
        info = api.camera_info(device_id)
        st.session_state.info[device_id] = info
        return info
    except ViKiApiError:
        return st.session_state.info.get(device_id)


def render() -> None:
    st.subheader("Cameras")
    devices = st.session_state.get("devices", [])
    if not devices:
        st.info(
            "No cameras detected. Connect a RealSense or Kinect and click "
            "**⟳ Scan devices** above."
        )
        return

    for device in devices:
        _render_card(device)

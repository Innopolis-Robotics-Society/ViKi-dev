from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from ..api import ViKiApiError
from ..embeds import mjpeg_img
from ..state import get_api


@st.fragment(run_every="2s")
def _job_status(api, job_id: str, chosen: str):
    try:
        job = api.dataset_job_status(job_id)
    except ViKiApiError:
        st.caption("Polling...")
        return

    status = job.get("status", "unknown")
    st.session_state["_ds_job_status"] = status
    st.info(f"Job status: **{status}**")

    if status == "running":
        st.caption("Computing robot IK trajectory...")
    elif status == "completed":
        st.success("Conversion complete!")
        result = job.get("result", {})
        if result:
            cols = st.columns(3)
            with cols[0]:
                st.metric("Frames", result.get("frames", "?"))
            with cols[1]:
                st.metric("Mean pos error", f"{result.get('mean_not_aligned_pos_error_mm', 0):.1f} mm")
            with cols[2]:
                st.metric("Mean rot error", f"{result.get('mean_not_aligned_orientation_error_deg', 0):.1f}°")
        traj_path = result.get("traj_path", "") if result else ""
        h5_name = traj_path.rsplit("/", 1)[-1] if traj_path else chosen.replace(".npz", ".h5")
        viz_url = api.dataset_viz_stream_url(h5_name)
        st.subheader("Robot trajectory")
        components.html(mjpeg_img(viz_url, height=400), height=420)

        st.subheader("Debug: IK target overlay")
        img_url = api.dataset_debug_viz_url()
        try:
            st.image(img_url, use_container_width=True)
        except Exception:
            st.caption("Debug viz unavailable (no retargeting data yet).")
    elif status == "failed":
        st.error(f"Conversion failed: {job.get('error', 'unknown')}")


def render() -> None:
    st.subheader("Robot Dataset Conversion")
    api = get_api()

    recs = []
    try:
        recs = api._get("/api/dataset/recordings", params={"page": 0, "limit": 100}).get("recordings", [])
    except ViKiApiError as exc:
        st.warning(f"Cannot load smoothed recordings: {exc}")

    if recs:
        chosen = st.selectbox("Smoothed recording", recs, key="ds_rec_select")
        robot = st.selectbox("Robot", ["ur10", "iiwa14"], index=0, key="ds_robot_select")

        if st.button("Convert to robot trajectory", type="primary", use_container_width=True):
            try:
                res = api.dataset_optimize(chosen, robot=robot)
                st.session_state["_ds_job_id"] = res["job_id"]
                st.session_state["_ds_job_status"] = "queued"
                st.rerun()
            except ViKiApiError as exc:
                st.error(f"Failed to start: {exc}")

        job_id = st.session_state.get("_ds_job_id")
        if job_id:
            _job_status(api, job_id, chosen)
    else:
        st.caption("No smoothed recordings found. Smooth a recording in the Skeleton tab first.")

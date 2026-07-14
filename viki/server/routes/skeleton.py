"""
viki.server.routes.skeleton
--------------------------
Endpoints for controlling skeleton estimation and recording,
and a WebSocket for streaming the latest skeleton frame.
"""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
import time
import logging
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel
import numpy as np
from viki.skeleton.models import LM

from argparse import Namespace

import viki.config as config
from viki.server.deps import get_worker, get_processor
from viki.server.skeleton_worker import SkeletonWorker
from viki.optimization.preparation.processor import PreparationPipeline


def sanitize_nan(val):
    """Recursively replace NaN with None for JSON serialization."""
    if isinstance(val, dict):
        return {k: sanitize_nan(v) for k, v in val.items()}
    if isinstance(val, list):
        return [sanitize_nan(x) for x in val]
    if isinstance(val, np.ndarray):
        return sanitize_nan(val.tolist())
    if isinstance(val, float) and np.isnan(val):
        return None
    return val


router = APIRouter(prefix="/api/skeleton", tags=["skeleton"])
logger = logging.getLogger(__name__)


class ToggleRequest(BaseModel):
    enabled: bool


class SmoothRequest(BaseModel):
    filename: str
    window_length: int = 7
    polyorder: int = 2


@router.post("/toggle")
async def toggle_estimation(
    req: ToggleRequest, worker: SkeletonWorker = Depends(get_worker)
):
    """
    Enable or disable skeleton estimation.

    Parameters
    ----------
    req : ToggleRequest
        `enabled` boolean.

    Returns
    -------
    dict
        {"status": "updated", "enabled": bool}
    """
    worker.set_enabled(req.enabled)
    return {"status": "updated", "enabled": worker.is_enabled}


@router.post("/record")
async def toggle_recording(
    req: ToggleRequest, worker: SkeletonWorker = Depends(get_worker)
):
    """
    Enable or disable recording of skeleton data to disk.

    Parameters
    ----------
    req : ToggleRequest
        `enabled` boolean.

    Returns
    -------
    dict
        {"status": "updated", "recording": bool}
    """
    worker.set_recording(req.enabled)
    return {"status": "updated", "recording": worker.is_recording}


@router.post("/depth-debug")
async def toggle_depth_debug(
    req: ToggleRequest, worker: SkeletonWorker = Depends(get_worker)
):
    """
    Enable or disable depth-projection debug marks (red dots on the 3D panel).

    Parameters
    ----------
    req : ToggleRequest
        `enabled` boolean.

    Returns
    -------
    dict
        {"status": "updated", "depth_debug": bool}
    """
    worker.set_depth_debug(req.enabled)
    return {"status": "updated", "depth_debug": req.enabled}


@router.get("/status")
async def get_status(worker: SkeletonWorker = Depends(get_worker)):
    """
    Get current skeleton estimation and recording status.

    Returns
    -------
    dict
        {"enabled": bool, "recording": bool}
    """
    return {
        "enabled": worker.is_enabled,
        "recording": worker.is_recording,
    }

@router.get("/recordings")
async def list_recordings(
    page: int = 0, 
    limit: int = 10, 
    processor: PreparationPipeline = Depends(get_processor)
):
    """
    List recorded skeleton data files (paginated).

    Parameters
    ----------
    page : int, default=0
        Page number.
    limit : int, default=10
        Items per page.

    Returns
    -------
    dict
        {"recordings": list[str]} – list of filenames.
    """
    return {"recordings": processor.list_recordings(page=page, page_size=limit)}


@router.post("/smooth")
async def smooth_recording(
    req: SmoothRequest,
    processor: PreparationPipeline = Depends(get_processor)
):
    """
    Apply Savitzky-Golay smoothing to a recorded skeleton file.

    Parameters
    ----------
    req : SmoothRequest
        Filename, window length, and polynomial order.

    Returns
    -------
    dict
        {"status": "success", "path": str} – path to smoothed file.

    Raises
    ------
    HTTPException 404
        If file not found.
    HTTPException 400
        If smoothing parameters are invalid.
    HTTPException 500
        If an internal error occurs.
    """
    try:
        path, _ = processor.smooth_recording(
            req.filename,
            window_length=req.window_length,
            polyorder=req.polyorder
        )
        return {"status": "success", "path": path}
    except FileNotFoundError:
        raise HTTPException(404, f"Recording {req.filename} not found")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception("Smoothing failed")
        raise HTTPException(500, f"Smoothing failed: {str(e)}")


@router.get("/smooth-plot")
async def smooth_plot(filename: str):
    """
    Return a PNG image comparing raw and smoothed wrist trajectories.

    Parameters
    ----------
    filename : str
        Smoothed .npz file name.

    Returns
    -------
    Response
        PNG image.

    Raises
    ------
    HTTPException 404
        If file not found.
    """
    """Return a PNG comparison of raw vs smoothed wrist trajectory."""
    smoothed_dir = Path(config.SKELETON_SMOOTHED_DIR)
    npz_path = smoothed_dir / filename
    if not npz_path.exists():
        raise HTTPException(status_code=404, detail=f"Smoothed recording not found: {filename}")

    with np.load(npz_path) as data:
        positions = data["positions"]
        timestamps = data["timestamps"]
        raw_points = data.get("raw_points")
        landmark_ids = data.get("landmark_ids")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    t_sec = (timestamps - timestamps[0]) / 1_000_000

    labels = ["X", "Y", "Z"]
    colors_raw = ["#e74c3c", "#e67e22", "#3498db"]
    colors_smooth = ["#2ecc71", "#1abc9c", "#9b59b6"]

    for i, (ax, label, cr, cs) in enumerate(zip(axes, labels, colors_raw, colors_smooth)):
        ax.plot(t_sec, positions[:, i], color=cs, linewidth=2, label="Smoothed" if i == 0 else None)
        if raw_points is not None and landmark_ids is not None:
            wrist_col = int(np.where(landmark_ids == 0)[0][0])
            ax.plot(t_sec, raw_points[:, wrist_col, i], color=cr, linewidth=1, alpha=0.5, label="Raw" if i == 0 else None)
        ax.set_ylabel(f"{label} (m)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Smoothing comparison — {filename}", fontsize=12)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")


@router.websocket("/stream")
async def skeleton_stream(websocket: WebSocket):
    """
    WebSocket endpoint that streams the latest skeleton result.

    Sends JSON with one entry per camera in ``frames`` (each tagged with its
    ``device_id`` so the frontend can draw it in its own colour), the per‑camera
    2D ``detections``, and the (un‑fused) ``debug_depth_marks``. Updates at
    approximately 20 Hz.
    """
    await websocket.accept()
    worker: SkeletonWorker = websocket.app.state.skeleton_worker
    try:
        while True:
            result = worker.get_latest_result()
            detections = worker.get_latest_detections()

            if result or detections:
                debug_marks = (
                    result.debug_depth_marks if result is not None else None
                )
                ts = (
                    result.frames[0].timestamp_us
                    if result and result.frames
                    else time.time_ns() // 1000
                )
                data = {
                    "ts": ts,
                    "frames": [
                        {
                            "device_id": f.device_id,
                            "landmarks": sanitize_nan(
                                {lm.value: vec for lm, vec in f.points.items()}
                            ),
                            "end_effector": (
                                sanitize_nan(f.end_effector.as_dict())
                                if f.end_effector else None
                            ),
                        }
                        for f in (result.frames if result else [])
                    ],
                    "detections": {
                        dev_id: (sanitize_nan(det.points) if det else {})
                        for dev_id, det in detections.items()
                    },
                    "debug_depth_marks": (
                        {
                            dev_id: {
                                lm.value: sanitize_nan(vec.tolist())
                                for lm, vec in marks.items()
                            }
                            for dev_id, marks in debug_marks.items()
                        }
                        if debug_marks else {}
                    ),
                }
                await websocket.send_json(data)

            # Stream at ~20 fps (comment out for unbound stream)
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        pass

"""
viki.server.streams
--------------------
MJPEG stream generators. These poll the CameraManager (non-blocking) and
yield multipart JPEG chunks, delegating all pixel work to ``viki.viz``.

Kept separate from the route handlers so the endpoints stay thin and the
transport/timing logic lives in one place.
"""

from __future__ import annotations

import logging
import time
from typing import Iterator

import cv2
import numpy as np

from viki.calibration.manager import CalibrationManager
from viki.capture.manager import CameraManager
from viki.config import JPEG_QUALITY, PLACEHOLDER_SIZE, STREAM_IDLE_SLEEP
from viki.viz.depth import DepthColorizer, Undistorter
from viki.viz.mjpeg import mjpeg_chunk, placeholder
from viki.config import INTRINSICS_FILENAME


def camera_stream(
    mgr: CameraManager, cal: CalibrationManager, device_id: str, mode: str
) -> Iterator[bytes]:
    """
    Yield MJPEG chunks for one camera.

    ``mode`` is ``"color"`` (optionally undistorted) or ``"depth"`` (colour-mapped).
    Ends when the device is no longer active.
    """
    pw, ph = PLACEHOLDER_SIZE
    last_ts = -1

    # Fetch calibration once; build an undistorter only if it exists.
    try:
        cal.load_intrinsics(device_id, INTRINSICS_FILENAME)
    except:
        pass
    intrinsics = cal.get_intrinsics(device_id)
    if not intrinsics:
        msg = f"Could not create camera stream: No intrinsics available for {device_id}"
        logging.warning(msg)
        undistorter = None
    else:
        undistorter = Undistorter(intrinsics.camera_matrix, intrinsics.dist_coeffs)
    colorizer = DepthColorizer()

    while True:
        frame = mgr.latest_frame(device_id)

        if frame is None:
            if device_id not in mgr.active_device_ids():
                return
            img = placeholder(pw, ph, f"{device_id}: not started")
            last_ts = -1
        elif frame.host_timestamp_us == last_ts:
            time.sleep(STREAM_IDLE_SLEEP)
            continue
        else:
            last_ts = frame.host_timestamp_us
            if mode == "color":
                img = frame.color
                if undistorter is not None:
                    img = undistorter.apply(img)
            else:
                img = colorizer.colorize(frame.depth)
                if img is None:
                    time.sleep(STREAM_IDLE_SLEEP)
                    continue

        yield mjpeg_chunk(img, JPEG_QUALITY)


def calibration_mosaic(mgr: CameraManager) -> Iterator[bytes]:
    """Yield MJPEG chunks of a horizontal mosaic of all active colour streams."""
    pw, ph = PLACEHOLDER_SIZE

    while True:
        active_ids = mgr.active_device_ids()
        if not active_ids:
            yield mjpeg_chunk(placeholder(pw, ph, "No cameras active"), JPEG_QUALITY)
            time.sleep(0.1)
            continue

        frames = {}
        for dev_id in active_ids:
            frame = mgr.latest_frame(dev_id)
            if frame:
                frames[dev_id] = frame.color

        if not frames:
            time.sleep(0.01)
            continue

        resized = [cv2.resize(img, (pw, ph)) for img in frames.values()]
        combined = np.hstack(resized)
        yield mjpeg_chunk(combined, JPEG_QUALITY)
        time.sleep(0.03)

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
from viki.viz.depth import DepthColorizer, Undistorter, DepthStabilizer
from viki.viz.mjpeg import mjpeg_chunk, placeholder
from viki.config import INTRINSICS_FILENAME


def camera_stream(
    mgr: CameraManager, cal: CalibrationManager, device_id: str, mode: str, undistort: bool = True
) -> Iterator[bytes]:
    """
    Yield MJPEG chunks for one camera.

    ``mode`` is ``"color"`` (optionally undistorted) or ``"depth"`` (colour-mapped).
    Ends when the device is no longer active.
    """
    pw, ph = PLACEHOLDER_SIZE
    last_ts = -1

    # Fetch calibration once; build an undistorter only if it exists.
    intrinsics = cal.get_intrinsics(device_id, path=INTRINSICS_FILENAME)
    if not intrinsics:
        msg = f"Could not create camera stream: No intrinsics available for {device_id}"
        logging.warning(msg)
        undistorter = None
    else:
        undistorter = Undistorter(intrinsics.camera_matrix, intrinsics.dist_coeffs)
    colorizer = DepthColorizer()
    stabilizer = DepthStabilizer()


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
                if undistort and undistorter is not None:
                    img = undistorter.apply(img)
            else:
                depth = stabilizer.stabilize(frame.depth)
                img = colorizer.colorize(depth)
                if img is None:
                    time.sleep(STREAM_IDLE_SLEEP)
                    continue

        yield mjpeg_chunk(img, JPEG_QUALITY)

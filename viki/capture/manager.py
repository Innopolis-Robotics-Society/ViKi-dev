"""
viki.capture.manager
--------------------
CameraManager: detects, starts, and owns camera backends.
Each camera runs in a background thread; the latest frame is always
available for non-blocking reads by the MJPEG streamer.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

from .base import CameraBackend, Frame


class _CameraWorker:
    """Background thread that continuously reads frames from one camera."""

    def __init__(self, backend: CameraBackend) -> None:
        self.backend = backend
        self._latest: Optional[Frame] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self.backend.start()
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self.backend.stop()

    def latest(self) -> Optional[Frame]:
        with self._lock:
            return self._latest

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame = self.backend.get_frame()
                with self._lock:
                    self._latest = frame
            except Exception as exc:
                print(f"[worker:{self.backend.device_id}] error: {exc}")
                time.sleep(0.1)


class CameraManager:
    """Manages multiple camera backends and their worker threads."""

    def __init__(self) -> None:
        self._workers: dict[str, _CameraWorker] = {}

    # ── Device discovery ──────────────────────────────────────────────────────

    def list_devices(self) -> dict:
        """Return all detected camera device IDs grouped by type."""
        devices: dict = {
            "realsense": [],
            "kinect": [],
            "active": list(self._workers.keys()),
        }

        try:
            from .realsense import RealSenseBackend
            devices["realsense"] = RealSenseBackend.list_devices()
        except Exception as e:
            devices["realsense_error"] = str(e)

        try:
            from .kinect import KinectBackend
            count = KinectBackend.device_count()
            devices["kinect"] = [f"kinect_{i}" for i in range(count)]
        except Exception as e:
            devices["kinect_error"] = str(e)

        return devices

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(
        self,
        device_id: str,
        fps: int = 30,
        color_width: int = 640,
        color_height: int = 480,
        depth_mode: str = "NFOV_UNBINNED",
    ) -> None:
        if device_id in self._workers:
            return  # already running

        backend = self._make_backend(device_id, fps, color_width, color_height, depth_mode)
        worker = _CameraWorker(backend)
        worker.start()
        self._workers[device_id] = worker

    def stop(self, device_id: str) -> None:
        worker = self._workers.pop(device_id, None)
        if worker:
            worker.stop()

    def stop_all(self) -> None:
        for device_id in list(self._workers):
            self.stop(device_id)

    # ── Frame access ──────────────────────────────────────────────────────────

    def latest_frame(self, device_id: str) -> Optional[Frame]:
        worker = self._workers.get(device_id)
        return worker.latest() if worker else None

    def get_info(self, device_id: str) -> Optional[dict]:
        worker = self._workers.get(device_id)
        if not worker:
            return None
        frame = worker.latest()
        info: dict = {
            "device_id": device_id,
            "running": True,
            "has_frame": frame is not None,
        }
        if frame:
            info["color_shape"] = list(frame.color.shape)
            info["depth_shape"] = list(frame.depth.shape)
            info["timestamp_us"] = frame.timestamp_us
            if frame.color_intrinsics:
                ci = frame.color_intrinsics
                info["color_intrinsics"] = {
                    "fx": ci.fx, "fy": ci.fy,
                    "cx": ci.cx, "cy": ci.cy,
                    "width": ci.width, "height": ci.height,
                }
        return info

    # ── Backend factory ───────────────────────────────────────────────────────

    @staticmethod
    def _make_backend(
        device_id: str,
        fps: int,
        color_width: int,
        color_height: int,
        depth_mode: str,
    ) -> CameraBackend:
        if device_id.startswith("kinect_"):
            from .kinect import KinectBackend
            idx = int(device_id.split("_")[1])
            return KinectBackend(
                device_index=idx,
                color_resolution=(color_width, color_height),
                depth_mode=depth_mode,
                fps=fps,
            )
        else:
            from .realsense import RealSenseBackend
            return RealSenseBackend(
                serial=device_id,
                color_resolution=(color_width, color_height),
                depth_resolution=(color_width, color_height),
                fps=fps,
            )
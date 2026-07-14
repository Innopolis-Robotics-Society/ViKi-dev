"""Simplified CameraManager + per-camera worker (mirrors viki/capture/manager.py)."""

from __future__ import annotations

import threading
from collections import deque
from typing import Dict, Optional

from .capture_base import CameraBackend, Frame
from .sync import SyncedFrameGroup


class _CameraWorker(threading.Thread):
    """Daemon thread that pulls frames from one backend into a ring buffer."""

    def __init__(self, backend: CameraBackend, buffer_size: int = 4) -> None:
        super().__init__(daemon=True)
        self.backend = backend
        self._buffer: deque[Frame] = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            frame = self.backend.get_frame()
            if frame is not None:
                with self._lock:
                    self._buffer.append(frame)

    def latest(self) -> Optional[Frame]:
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def shutdown(self) -> None:
        self._stop.set()


class CameraManager:
    """Owns backends and workers; serves the latest frame on demand."""

    def __init__(self) -> None:
        self.backends: Dict[str, CameraBackend] = {}
        self.workers: Dict[str, _CameraWorker] = {}

    def list_devices(self) -> list[str]:
        ...  # device discovery

    def start_camera(self, device_id: str) -> None:
        backend = self._make_backend(device_id)
        self.backends[device_id] = backend
        backend.start()
        worker = _CameraWorker(backend)
        worker.start()
        self.workers[device_id] = worker

    def stop_camera(self, device_id: str) -> None:
        worker = self.workers.pop(device_id, None)
        if worker is not None:
            worker.shutdown()
            worker.join(timeout=1.0)
        backend = self.backends.pop(device_id, None)
        if backend is not None:
            backend.stop()

    def get_backend(self, device_id: str) -> Optional[CameraBackend]:
        return self.backends.get(device_id)

    def latest_frame(self, device_id: str) -> Optional[Frame]:
        worker = self.workers.get(device_id)
        return worker.latest() if worker else None

    def _make_backend(self, device_id: str) -> CameraBackend:
        ...  # RealSenseBackend / KinectBackend selection

"""viki.streamlit_app.api
--------------------------
``ViKiApi``: a thin ``requests.Session`` wrapper over the FastAPI ``/api/*``
surface. This is the ONLY module that knows endpoint paths; sections call typed
helper methods and handle the raised :class:`ViKiApiError` with
``st.error``/``st.toast`` so the page never crashes when the backend is down.

Endpoint shapes verified against ``viki/server/routes/*.py`` and the React
``*.api.ts`` modules.
"""

from __future__ import annotations

from typing import Any

import requests

from .settings import BACKEND_URL, REQUEST_TIMEOUT


class ViKiApiError(Exception):
    """Raised for any failed backend call (network error or non-2xx status)."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class ViKiApi:
    """Typed-ish HTTP helpers over the ViKi FastAPI backend."""

    def __init__(self, base_url: str = BACKEND_URL, timeout: float = REQUEST_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    # -- low-level -----------------------------------------------------------
    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Any | None = None,
        params: dict | None = None,
        timeout: float | None = None,
    ) -> Any:
        try:
            res = self._session.request(
                method,
                self._url(path),
                json=json,
                params=params,
                timeout=timeout if timeout is not None else self.timeout,
            )
        except requests.RequestException as exc:
            raise ViKiApiError(f"Backend unreachable: {exc}") from exc

        if not res.ok:
            detail = res.text
            try:
                body = res.json()
                if isinstance(body, dict) and "detail" in body:
                    detail = body["detail"]
            except ValueError:
                pass
            raise ViKiApiError(
                f"HTTP {res.status_code}: {detail or res.reason}", status=res.status_code
            )

        ctype = res.headers.get("content-type", "")
        if res.status_code == 204 or not res.content:
            return None
        if "application/json" in ctype:
            return res.json()
        return res.text

    def _get(self, path: str, **kw) -> Any:
        return self._request("GET", path, **kw)

    def _post(self, path: str, **kw) -> Any:
        return self._request("POST", path, **kw)

    # -- health --------------------------------------------------------------
    def is_alive(self) -> bool:
        """True if the backend responds to a device scan (used for the dot)."""
        try:
            self._get("/api/devices", timeout=3)
            return True
        except ViKiApiError:
            return False

    # -- cameras -------------------------------------------------------------
    def list_devices(self) -> dict:
        """GET /api/devices -> {realsense, kinect, web_camera, active, ...}."""
        return self._get("/api/devices")

    def start_camera(self, device_id: str, cfg: dict) -> dict:
        """POST /api/cameras/{id}/start (body: color_width/height, fps, depth_mode)."""
        return self._post(f"/api/cameras/{device_id}/start", json=cfg)

    def stop_camera(self, device_id: str) -> dict:
        return self._post(f"/api/cameras/{device_id}/stop")

    def camera_info(self, device_id: str) -> dict:
        return self._get(f"/api/cameras/{device_id}/info")

    # -- config / system -----------------------------------------------------
    def get_config(self) -> dict:
        return self._get("/api/config")

    def save_config(self, config: dict) -> dict:
        return self._post("/api/config", json=config)

    def reset_config(self) -> dict:
        return self._post("/api/config/reset")

    def restart(self) -> None:
        """POST /api/restart. The server calls ``os._exit`` so the connection
        drops mid-request; a dropped connection here is the expected success."""
        try:
            self._post("/api/restart", timeout=3)
        except ViKiApiError:
            # Connection reset because the process exited -- treat as success.
            pass

    # -- calibration ---------------------------------------------------------
    def calibration_reset(self) -> dict:
        return self._post("/api/calibration/reset")

    def calibration_sync(self, board_type: str, params: dict) -> dict:
        return self._post(
            "/api/calibration/sync", params={"board_type": board_type}, json=params
        )

    def calibration_start_chess(self, device_id: str, params: dict) -> Any:
        return self._post(
            f"/api/calibration/start/{device_id}",
            params={"mode": "manual"},
            json=params,
        )

    def calibration_start_aruco(self, device_id: str, params: dict) -> Any:
        return self._post(
            f"/api/calibration/start/aruco/{device_id}",
            params={"mode": "manual"},
            json=params,
        )

    def calibration_capture_all(self) -> Any:
        return self._post("/api/calibration/capture")

    def calibration_status(self, device_id: str) -> dict:
        """GET /api/calibration/status/{id} -> {samples_count, started}."""
        return self._get(f"/api/calibration/status/{device_id}")

    def calibration_clear(self, device_id: str) -> dict:
        return self._post(f"/api/calibration/clear/{device_id}")

    def calibration_intrinsics(self, device_id: str) -> dict:
        return self._post(f"/api/calibration/intrinsics/{device_id}")

    def calibration_extrinsics(self) -> list:
        """POST /api/calibration/extrinsics -> [{device_id, rvec, tvec}, ...]."""
        return self._post("/api/calibration/extrinsics")

    # -- skeleton ------------------------------------------------------------
    def skeleton_toggle(self, enabled: bool) -> dict:
        return self._post("/api/skeleton/toggle", json={"enabled": enabled})

    def skeleton_record(self, enabled: bool) -> dict:
        return self._post("/api/skeleton/record", json={"enabled": enabled})

    def skeleton_status(self) -> dict:
        """GET /api/skeleton/status -> {enabled, recording}."""
        return self._get("/api/skeleton/status")

    # -- recording -----------------------------------------------------------
    def record_start(self, duration: float, fps: int) -> dict:
        """POST /api/record/start (RGB-D recording in the background worker)."""
        return self._post(
            "/api/record/start", json={"duration": duration, "fps": fps}
        )

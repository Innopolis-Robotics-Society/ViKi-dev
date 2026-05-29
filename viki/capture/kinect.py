"""
viki.capture.kinect
-------------------
Azure Kinect DK backend using ctypes directly over libk4a.so.
No compilation required — only libk4a.so needs to be installed system-wide.

Tested with libk4a 1.4.1 on Ubuntu 22.04 inside Docker.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import numpy as np
from typing import Optional

from .base import CameraBackend, CameraIntrinsics, Frame

# ── Load libk4a ───────────────────────────────────────────────────────────────

def _load_libk4a() -> ctypes.CDLL:
    for name in ("libk4a.so", "libk4a.so.1.4", "libk4a.so.1"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise OSError(
        "libk4a.so not found. Make sure libk4a1.4 is installed:\n"
        "  apt-get install /path/to/libk4a1.4_1.4.1_amd64.deb"
    )

_lib = _load_libk4a()

# ── k4a constants ─────────────────────────────────────────────────────────────

K4A_RESULT_SUCCEEDED        = 0
K4A_WAIT_RESULT_SUCCEEDED   = 0
K4A_WAIT_RESULT_TIMEOUT     = 1

K4A_COLOR_RESOLUTION_720P   = 1   # 1280x720
K4A_COLOR_RESOLUTION_1080P  = 2   # 1920x1080
K4A_COLOR_RESOLUTION_1536P  = 4   # 2048x1536

K4A_DEPTH_MODE_NFOV_UNBINNED   = 3  # 640x576
K4A_DEPTH_MODE_NFOV_2X2BINNED  = 2  # 320x288
K4A_DEPTH_MODE_WFOV_UNBINNED   = 5  # 1024x1024
K4A_DEPTH_MODE_WFOV_2X2BINNED  = 4  # 512x512

K4A_FRAMES_PER_SECOND_5   = 0
K4A_FRAMES_PER_SECOND_15  = 1
K4A_FRAMES_PER_SECOND_30  = 2

K4A_IMAGE_FORMAT_COLOR_BGRA32 = 0
K4A_IMAGE_FORMAT_DEPTH16      = 3

K4A_CALIBRATION_TYPE_COLOR = 1
K4A_CALIBRATION_TYPE_DEPTH = 0

# ── ctypes structs ────────────────────────────────────────────────────────────

class K4ADeviceConfig(ctypes.Structure):
    _fields_ = [
        ("color_format",           ctypes.c_int),
        ("color_resolution",       ctypes.c_int),
        ("depth_mode",             ctypes.c_int),
        ("camera_fps",             ctypes.c_int),
        ("synchronized_images_only", ctypes.c_bool),
        ("depth_delay_off_color_usec", ctypes.c_int32),
        ("wired_sync_mode",        ctypes.c_int),
        ("subordinate_delay_off_master_usec", ctypes.c_uint32),
        ("disable_streaming_indicator", ctypes.c_bool),
    ]


# Opaque handle types
K4ADevice   = ctypes.c_void_p
K4ACapture  = ctypes.c_void_p
K4AImage    = ctypes.c_void_p

# ── Function signatures ───────────────────────────────────────────────────────

_lib.k4a_device_get_installed_count.restype  = ctypes.c_uint32
_lib.k4a_device_get_installed_count.argtypes = []

_lib.k4a_device_open.restype  = ctypes.c_int
_lib.k4a_device_open.argtypes = [ctypes.c_uint32, ctypes.POINTER(K4ADevice)]

_lib.k4a_device_close.restype  = None
_lib.k4a_device_close.argtypes = [K4ADevice]

_lib.k4a_device_start_cameras.restype  = ctypes.c_int
_lib.k4a_device_start_cameras.argtypes = [K4ADevice, ctypes.POINTER(K4ADeviceConfig)]

_lib.k4a_device_stop_cameras.restype  = None
_lib.k4a_device_stop_cameras.argtypes = [K4ADevice]

_lib.k4a_device_get_capture.restype  = ctypes.c_int
_lib.k4a_device_get_capture.argtypes = [K4ADevice, ctypes.POINTER(K4ACapture), ctypes.c_int32]

_lib.k4a_capture_release.restype  = None
_lib.k4a_capture_release.argtypes = [K4ACapture]

_lib.k4a_capture_get_color_image.restype  = K4AImage
_lib.k4a_capture_get_color_image.argtypes = [K4ACapture]

_lib.k4a_capture_get_depth_image.restype  = K4AImage
_lib.k4a_capture_get_depth_image.argtypes = [K4ACapture]

_lib.k4a_image_get_buffer.restype  = ctypes.c_void_p
_lib.k4a_image_get_buffer.argtypes = [K4AImage]

_lib.k4a_image_get_size.restype  = ctypes.c_size_t
_lib.k4a_image_get_size.argtypes = [K4AImage]

_lib.k4a_image_get_width_pixels.restype  = ctypes.c_int
_lib.k4a_image_get_width_pixels.argtypes = [K4AImage]

_lib.k4a_image_get_height_pixels.restype  = ctypes.c_int
_lib.k4a_image_get_height_pixels.argtypes = [K4AImage]

_lib.k4a_image_get_timestamp_usec.restype  = ctypes.c_uint64
_lib.k4a_image_get_timestamp_usec.argtypes = [K4AImage]

_lib.k4a_image_release.restype  = None
_lib.k4a_image_release.argtypes = [K4AImage]

_lib.k4a_device_get_serialnum.restype  = ctypes.c_int
_lib.k4a_device_get_serialnum.argtypes = [
    K4ADevice, ctypes.c_char_p, ctypes.POINTER(ctypes.c_size_t)
]

# ── Resolution maps ───────────────────────────────────────────────────────────

_COLOR_RES_MAP = {
    (1280, 720):  K4A_COLOR_RESOLUTION_720P,
    (1920, 1080): K4A_COLOR_RESOLUTION_1080P,
    (2048, 1536): K4A_COLOR_RESOLUTION_1536P,
}

_DEPTH_MODE_MAP = {
    "NFOV_UNBINNED":  K4A_DEPTH_MODE_NFOV_UNBINNED,
    "NFOV_2X2BINNED": K4A_DEPTH_MODE_NFOV_2X2BINNED,
    "WFOV_UNBINNED":  K4A_DEPTH_MODE_WFOV_UNBINNED,
    "WFOV_2X2BINNED": K4A_DEPTH_MODE_WFOV_2X2BINNED,
}

_FPS_MAP = {
    5:  K4A_FRAMES_PER_SECOND_5,
    15: K4A_FRAMES_PER_SECOND_15,
    30: K4A_FRAMES_PER_SECOND_30,
}


# ── Backend ───────────────────────────────────────────────────────────────────

class KinectBackend(CameraBackend):
    """
    Azure Kinect DK backend via ctypes over libk4a.so.
    No pyk4a required.

    Parameters
    ----------
    device_index : int
        Device index (0 for first device).
    color_resolution : tuple[int, int]
        Supported: (1280,720), (1920,1080), (2048,1536). Default: (1280,720).
    depth_mode : str
        One of: NFOV_UNBINNED, NFOV_2X2BINNED, WFOV_UNBINNED, WFOV_2X2BINNED.
        Default: NFOV_UNBINNED.
    fps : int
        5, 15, or 30. Default: 30.
    timeout_ms : int
        Frame wait timeout in milliseconds. Default: 5000.
    """

    def __init__(
        self,
        device_index: int = 0,
        color_resolution: tuple[int, int] = (1280, 720),
        depth_mode: str = "NFOV_UNBINNED",
        fps: int = 30,
        timeout_ms: int = 5000,
    ) -> None:
        if color_resolution not in _COLOR_RES_MAP:
            raise ValueError(f"Unsupported color_resolution {color_resolution}. "
                             f"Supported: {list(_COLOR_RES_MAP)}")
        if depth_mode not in _DEPTH_MODE_MAP:
            raise ValueError(f"Unknown depth_mode '{depth_mode}'. "
                             f"Supported: {list(_DEPTH_MODE_MAP)}")
        if fps not in _FPS_MAP:
            raise ValueError(f"Supported fps: 5, 15, 30. Got: {fps}")

        self._device_index    = device_index
        self._color_resolution = color_resolution
        self._depth_mode      = depth_mode
        self._fps             = fps
        self._timeout_ms      = timeout_ms

        self._handle: K4ADevice = K4ADevice(None)
        self._serial_str: str   = f"kinect_{device_index}"
        self._running           = False

    # ── CameraBackend interface ───────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return

        # Open device
        res = _lib.k4a_device_open(self._device_index, ctypes.byref(self._handle))
        if res != K4A_RESULT_SUCCEEDED:
            raise RuntimeError(f"k4a_device_open failed (result={res}). "
                               "Check udev rules and USB permissions.")

        # Read serial number
        size = ctypes.c_size_t(64)
        buf  = ctypes.create_string_buffer(64)
        _lib.k4a_device_get_serialnum(self._handle, buf, ctypes.byref(size))
        self._serial_str = buf.value.decode(errors="replace")

        # Build config
        config = K4ADeviceConfig(
            color_format           = K4A_IMAGE_FORMAT_COLOR_BGRA32,
            color_resolution       = _COLOR_RES_MAP[self._color_resolution],
            depth_mode             = _DEPTH_MODE_MAP[self._depth_mode],
            camera_fps             = _FPS_MAP[self._fps],
            synchronized_images_only = True,
            depth_delay_off_color_usec = 0,
            wired_sync_mode        = 0,
            subordinate_delay_off_master_usec = 0,
            disable_streaming_indicator = False,
        )

        res = _lib.k4a_device_start_cameras(self._handle, ctypes.byref(config))
        if res != K4A_RESULT_SUCCEEDED:
            _lib.k4a_device_close(self._handle)
            raise RuntimeError(f"k4a_device_start_cameras failed (result={res})")

        self._running = True

    def stop(self) -> None:
        if not self._running:
            return
        _lib.k4a_device_stop_cameras(self._handle)
        _lib.k4a_device_close(self._handle)
        self._handle  = K4ADevice(None)
        self._running = False

    def get_frame(self) -> Frame:
        if not self._running:
            raise RuntimeError("KinectBackend is not started. Call start() first.")

        capture = K4ACapture(None)
        res = _lib.k4a_device_get_capture(
            self._handle, ctypes.byref(capture), self._timeout_ms
        )
        if res == K4A_WAIT_RESULT_TIMEOUT:
            raise TimeoutError("Kinect capture timed out.")
        if res != K4A_WAIT_RESULT_SUCCEEDED:
            raise RuntimeError(f"k4a_device_get_capture failed (result={res})")

        try:
            color_img = _lib.k4a_capture_get_color_image(capture)
            depth_img = _lib.k4a_capture_get_depth_image(capture)

            color = self._image_to_numpy_bgr(color_img)
            depth = self._image_to_numpy_depth(depth_img)
            ts    = int(_lib.k4a_image_get_timestamp_usec(color_img))

            _lib.k4a_image_release(color_img)
            _lib.k4a_image_release(depth_img)
        finally:
            _lib.k4a_capture_release(capture)

        return Frame(
            color            = color,
            depth            = depth,
            timestamp_us     = ts,
            device_id        = self._serial_str,
            color_intrinsics = None,  # TODO: add via k4a_calibration
            depth_intrinsics = None,
        )

    @property
    def device_id(self) -> str:
        return self._serial_str

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _image_to_numpy_bgr(img: K4AImage) -> np.ndarray:
        w   = _lib.k4a_image_get_width_pixels(img)
        h   = _lib.k4a_image_get_height_pixels(img)
        buf = _lib.k4a_image_get_buffer(img)
        # BGRA32 -> copy to numpy -> drop alpha
        arr = np.ctypeslib.as_array(
            (ctypes.c_uint8 * (h * w * 4)).from_address(buf)
        ).reshape(h, w, 4).copy()
        return arr[:, :, :3]  # BGRA -> BGR

    @staticmethod
    def _image_to_numpy_depth(img: K4AImage) -> np.ndarray:
        w   = _lib.k4a_image_get_width_pixels(img)
        h   = _lib.k4a_image_get_height_pixels(img)
        buf = _lib.k4a_image_get_buffer(img)
        # DEPTH16 = uint16, millimetres
        return np.ctypeslib.as_array(
            (ctypes.c_uint16 * (h * w)).from_address(buf)
        ).reshape(h, w).copy()

    # ── Static utils ──────────────────────────────────────────────────────────

    @staticmethod
    def device_count() -> int:
        """Return number of connected Kinect devices."""
        return int(_lib.k4a_device_get_installed_count())
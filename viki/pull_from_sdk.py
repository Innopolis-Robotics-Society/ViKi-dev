#================================================================================================
# to pull intrinsicdata from sdk for kinect 0 run:
# docker compose run --rm terminal python3 viki/pull_from_sdk.py 0 > kinect_0_intrinsics.json
#================================================================================================
import ctypes
import sys
import json

# Load libk4a
try:
    lib = ctypes.CDLL("libk4a.so")
except OSError:
    print("Error: libk4a.so not found. Run this inside the docker container.")
    sys.exit(1)

# Constants
K4A_RESULT_SUCCEEDED = 0
K4A_DEPTH_MODE_NFOV_UNBINNED = 3
K4A_COLOR_RESOLUTION_720P = 1

# --- C-Struct Definitions ---
class IntrinsicsParam(ctypes.Structure):
    _fields_ = [
        ("fx", ctypes.c_float), ("fy", ctypes.c_float),
        ("cx", ctypes.c_float), ("cy", ctypes.c_float),
        ("k1", ctypes.c_float), ("k2", ctypes.c_float),
        ("p1", ctypes.c_float), ("p2", ctypes.c_float),
        ("k3", ctypes.c_float), ("k4", ctypes.c_float),
        ("k5", ctypes.c_float), ("k6", ctypes.c_float),
        ("codx", ctypes.c_float), ("cody", ctypes.c_float),
    ]

class Intrinsics(ctypes.Structure):
    _fields_ = [("parameters", IntrinsicsParam), ("model_type", ctypes.c_int)]

class CameraCalibration(ctypes.Structure):
    _fields_ = [("intrinsics", Intrinsics), ("extrinsics", ctypes.c_byte * 64)] # Simplified

class Calibration(ctypes.Structure):
    _fields_ = [
        ("depth_camera_calibration", CameraCalibration),
        ("color_camera_calibration", CameraCalibration),
        # ... remaining fields are ignored as we only need color
    ]

# Function signatures
lib.k4a_device_open.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
lib.k4a_device_get_calibration.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(Calibration)
]
lib.k4a_device_close.argtypes = [ctypes.c_void_p]

def main():
    device_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = ctypes.c_void_p()
    
    if lib.k4a_device_open(device_idx, ctypes.byref(device)) != K4A_RESULT_SUCCEEDED:
        print(f"Failed to open device {device_idx}")
        sys.exit(1)

    calib = Calibration()
    res = lib.k4a_device_get_calibration(
        device, K4A_DEPTH_MODE_NFOV_UNBINNED, K4A_COLOR_RESOLUTION_720P, ctypes.byref(calib)
    )

    if res != K4A_RESULT_SUCCEEDED:
        print("Failed to get calibration")
        lib.k4a_device_close(device)
        sys.exit(1)

    p = calib.color_camera_calibration.intrinsics.parameters
    
    data = {
        "fx": p.fx, "fy": p.fy, "cx": p.cx, "cy": p.cy,
        "dist_coeffs": [p.k1, p.k2, p.p1, p.p2, p.k3, p.k4, p.k5, p.k6]
    }
    
    print(json.dumps(data, indent=2))
    lib.k4a_device_close(device)

if __name__ == "__main__":
    main()

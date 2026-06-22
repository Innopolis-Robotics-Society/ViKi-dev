import json
import logging
import numpy as np
from viki.config import INTRINSICS_FILENAME, EXTRINSICS_FILENAME
from viki.calibration.models import CalibrationIntrinsics, CalibrationExtrinsics


def write_device_intrinsics(
    device_id: str, intrinsics: CalibrationIntrinsics, file: str = INTRINSICS_FILENAME
):
    try:
        with open(file, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = []
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    entry = {
        "device_id": device_id,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "cx": intrinsics.cx,
        "cy": intrinsics.cy,
        "dist_coeffs": intrinsics.dist_coeffs.tolist(),
    }

    logging.debug(entry)

    for i, entry in enumerate(data):
        if entry.get("device_id") == device_id:
            data[i] = entry
            break
    else:
        data.append(entry)

    with open(file, "w") as f:
        json.dump(data, f, indent=2)


def write_device_extrinsics(
    device_id: str, extrinsics: CalibrationExtrinsics, file: str = EXTRINSICS_FILENAME
):
    try:
        with open(file, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = []
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    entry = {
        "device_id": device_id,
        "rvec": extrinsics.rvec.tolist(),
        "tvec": extrinsics.tvec.tolist(),
    }

    logging.debug(entry)

    for i, entry in enumerate(data):
        if entry.get("device_id") == device_id:
            data[i] = entry
            break
    else:
        data.append(entry)

    with open(file, "w") as f:
        json.dump(data, f, indent=2)


def read_device_intrinsics(
    device_id: str, file: str = INTRINSICS_FILENAME
) -> CalibrationIntrinsics | None:
    try:
        with open(file, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    if not isinstance(data, list):
        return None

    for entry in data:
        if entry.get("device_id") == device_id:
            dist_coeffs = np.array(entry.get("dist_coeffs", [0.0] * 5))

            return CalibrationIntrinsics(
                fx=entry["fx"],
                fy=entry["fy"],
                cx=entry["cx"],
                cy=entry["cy"],
                dist_coeffs=dist_coeffs,
            )

    return None


def read_device_extrinsics(
    device_id: str, file: str = EXTRINSICS_FILENAME
) -> CalibrationExtrinsics | None:
    try:
        with open(file, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    if not isinstance(data, list):
        return None

    for entry in data:
        if entry.get("device_id") == device_id:
            rvec = np.array(entry.get("rvec", [0.0] * 3))
            tvec = np.array(entry.get("tvec", [0.0] * 3))

            return CalibrationExtrinsics(rvec=rvec, tvec=tvec)

    return None

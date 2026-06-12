import cv2
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from .base import Frame

class CalibrationManager:
    """Handles chessboard detection and calibration for multiple cameras with subpixel precision."""

    def __init__(self, chessboard_size=(8, 6), square_size=0.025):
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        # List of capture events. Each event is device_id -> corners_ndarray
        self.samples: List[Dict[str, np.ndarray]] = []
        # device_id -> (width, height)
        self.resolutions: Dict[str, Tuple[int, int]] = {}
        self.valid_indices: List[int] = []

    def add_sample(self, frames: Dict[str, Frame]) -> Dict[str, bool]:
        """
        Adds a sample to the calibration set for any camera where the chessboard is found.
        Refines corners to subpixel accuracy.
        Returns a mapping of device_id -> success.
        """
        if not frames:
            return {}

        captured_this_event = {}
        success_map = {}

        # Criteria for subpixel corner refinement
        subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        for dev_id, frame in frames.items():
            # Store resolution for this device
            if dev_id not in self.resolutions:
                h, w = frame.color.shape[:2]
                self.resolutions[dev_id] = (w, h)

            gray = cv2.cvtColor(frame.color, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

            if ret:
                # Refine pixel positions to subpixel float accuracy
                refined_corners = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), subpix_criteria
                )
                captured_this_event[dev_id] = refined_corners
                success_map[dev_id] = True
            else:
                success_map[dev_id] = False

        if captured_this_event:
            self.samples.append(captured_this_event)

        return success_map

    def run_calibration(self, results_path: str = "viki/capture/calibration_results.npz") -> Dict:
        """Performs intrinsics and extrinsics calibration."""
        # Identify all devices that have at least one sample
        all_device_ids = set()
        for sample in self.samples:
            all_device_ids.update(sample.keys())
        device_ids = sorted(list(all_device_ids))

        if not device_ids:
            raise RuntimeError("No calibration samples collected.")

        # Check that every device has enough samples
        for dev_id in device_ids:
            count = sum(1 for s in self.samples if dev_id in s)
            if count < 20:
                raise RuntimeError(f"Not enough samples for {dev_id}. Need 20, found {count}.")

        # Object points: (0,0,0), (1,0,0), ..., (8,5,0)
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size

        intrinsics = {}
        dist_coeffs = {}
        errors = {}

        # 1. Individual Camera Calibration
        for dev_id in device_ids:
            # Extract corners only for this device
            corners_list = [np.asarray(s[dev_id], dtype=np.float32) for s in self.samples if dev_id in s]
            w, h = self.resolutions[dev_id]

            # Need object points for the exact number of samples this camera has
            obj_points = [objp for _ in range(len(corners_list))]

            ret, mtx, dist, _, _ = cv2.calibrateCamera(
                obj_points, corners_list, (w, h), None, None
            )
            if not ret:
                raise RuntimeError(f"Calibration failed for {dev_id}")
            intrinsics[dev_id] = mtx
            dist_coeffs[dev_id] = dist
            errors[dev_id] = float(ret)

        # 2. Stereo Calibration for first two cameras
        extrinsic_data = None
        stereo_error = None
        if len(device_ids) >= 2:
            cam0, cam1 = device_ids[0], device_ids[1]

            # Filter events where both cameras detected the board
            paired_samples = [s for s in self.samples if cam0 in s and cam1 in s]
            if len(paired_samples) < 10:
                # Not enough paired samples for stereo, just skip it
                extrinsic_data = None
            else:
                img_points0 = [np.asarray(s[cam0], dtype=np.float32) for s in paired_samples]
                img_points1 = [np.asarray(s[cam1], dtype=np.float32) for s in paired_samples]

                w, h = self.resolutions[cam0]
                obj_points = [objp for _ in range(len(paired_samples))]

                stereo_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

                ret, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
                    obj_points, img_points0, img_points1,
                    intrinsics[cam0], dist_coeffs[cam0],
                    intrinsics[cam1], dist_coeffs[cam1],
                    (w, h),
                    flags=cv2.CALIB_FIX_INTRINSIC,
                    criteria=stereo_criteria
                )

                if ret:
                    extrinsic_data = {"pair": (cam0, cam1), "R": R, "T": T}
                    stereo_error = float(ret)
                else:
                    raise RuntimeError(f"Stereo calibration optimization failed between {cam0} and {cam1}")

        # Save binary npz deployment package
        np.savez(results_path, intrinsics=intrinsics, dist_coeffs=dist_coeffs, extrinsic_data=extrinsic_data)

        # Save as human-readable JSON
        json_path = results_path.replace(".npz", ".json")
        json_data = {
            "intrinsics": {k: v.tolist() for k, v in intrinsics.items()},
            "dist_coeffs": {k: v.tolist() for k, v in dist_coeffs.items()},
            "extrinsic_data": None
        }
        if extrinsic_data:
            json_data["extrinsic_data"] = {
                "pair": extrinsic_data["pair"],
                "R": extrinsic_data["R"].tolist(),
                "T": extrinsic_data["T"].tolist(),
            }

        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=4)

        return {
            "status": "success",
            "samples_used": len(self.samples),
            "cameras": device_ids,
            "stereo_calibrated": extrinsic_data is not None,
            "errors": errors,
            "stereo_error": stereo_error,
            "intrinsics": intrinsics,
            "dist_coeffs": dist_coeffs
        }


    def clear(self):
        self.samples = []
        self.resolutions = {}
        self.valid_indices = []

    @property
    def sample_count(self) -> int:
        return len(self.samples)

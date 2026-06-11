import cv2
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from .base import Frame

class CalibrationManager:
    """Handles chessboard detection and calibration for multiple cameras."""
    
    def __init__(self, chessboard_size=(8, 6), square_size=0.025):
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        # device_id -> List[Tuple[Frame, np.ndarray]] (frame, corners)
        self.samples: Dict[str, List[Tuple[Frame, np.ndarray]]] = {}
        self.valid_indices: List[int] = []
        
    def add_sample(self, frames: Dict[str, Frame]) -> bool:
        """
        Adds a sample to the calibration set if the chessboard is found in ALL provided cameras.
        Returns True if the sample was valid and added, False otherwise.
        """
        device_ids = list(frames.keys())
        if not device_ids:
            return False
            
        # Initialize lists for new devices
        for dev_id in device_ids:
            if dev_id not in self.samples:
                self.samples[dev_id] = []
        
        # Check for chessboard in all cameras
        captured_corners = {}
        all_found = True
        
        for dev_id, frame in frames.items():
            gray = cv2.cvtColor(frame.color, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
            if ret:
                captured_corners[dev_id] = corners
            else:
                all_found = False
                break
        
        if all_found:
            for dev_id, frame in frames.items():
                self.samples[dev_id].append((frame, captured_corners[dev_id]))
            return True
        
        return False

    def run_calibration(self, results_path: str = "viki/capture/calibration_results.npz") -> Dict:
        """Performs intrinsics and extrinsics calibration."""
        device_ids = list(self.samples.keys())
        if not device_ids or len(self.samples[device_ids[0]]) < 20:
            raise RuntimeError(f"Not enough valid samples. Need at least 20, found {len(self.samples[device_ids[0]]) if device_ids else 0}.")
            
        # Object points: (0,0,0), (1,0,0), ..., (8,5,0)
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size

        # Create a list of object points, one for each image
        num_samples = len(self.samples[device_ids[0]])
        obj_points = [objp for _ in range(num_samples)]

        intrinsics = {}
        dist_coeffs = {}
        
        # 1. Individual Camera Calibration
        for dev_id in device_ids:
            # Extract stored corners
            corners_list = [sample[1] for sample in self.samples[dev_id]]
            
            # Use the first frame to get image size
            h, w = self.samples[dev_id][0][0].color.shape[:2]
            
            ret, mtx, dist, _, _ = cv2.calibrateCamera(
                obj_points, corners_list, (w, h), None, None
            )
            if not ret:
                raise RuntimeError(f"Calibration failed for {dev_id}")
            intrinsics[dev_id] = mtx
            dist_coeffs[dev_id] = dist

        # 2. Stereo Calibration for first two cameras
        extrinsic_data = None
        if len(device_ids) >= 2:
            cam0, cam1 = device_ids[0], device_ids[1]
            
            img_points0 = [sample[1] for sample in self.samples[cam0]]
            img_points1 = [sample[1] for sample in self.samples[cam1]]
            
            h, w = self.samples[cam0][0][0].color.shape[:2]
            
            ret, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
                obj_points, img_points0, img_points1,
                intrinsics[cam0], dist_coeffs[cam0],
                intrinsics[cam1], dist_coeffs[cam1],
                (w, h),
                flags=cv2.CALIB_FIX_INTRINSIC
            )
            
            if ret:
                extrinsic_data = {"pair": (cam0, cam1), "R": R, "T": T}

        np.savez(results_path, intrinsics=intrinsics, dist_coeffs=dist_coeffs, extrinsic_data=extrinsic_data)
        
        # Also save as human-readable JSON
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
            "samples_used": num_samples,
            "cameras": device_ids,
            "stereo_calibrated": extrinsic_data is not None
        }

    def clear(self):
        self.samples = {}
        self.valid_indices = []

    @property
    def sample_count(self) -> int:
        if not self.samples:
            return 0
        return len(next(iter(self.samples.values())))

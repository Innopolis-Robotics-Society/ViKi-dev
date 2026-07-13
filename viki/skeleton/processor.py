"""
viki.skeleton.processor
----------------------
Business logic for processing skeleton recording files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
from viki.skeleton import utils
from viki.skeleton.smoothing import smooth_landmark_sequence, interpolate_nans
from viki.skeleton.hand_angles import compute_end_effector_pose
from viki.skeleton.models import LM
import viki.config as config

class SkeletonProcessor:
    """
    Handles listing and smoothing of skeleton recording files.
    """

    def __init__(self) -> None:
        self.recs_dir = Path(config.SKELETON_RECS_DIR)
        self.smoothed_dir = Path(config.SKELETON_SMOOTHED_DIR)
        
        self.recs_dir.mkdir(parents=True, exist_ok=True)
        self.smoothed_dir.mkdir(parents=True, exist_ok=True)

    def list_recordings(self, page: int = 0, page_size: int = 10) -> List[str]:
        """
        List all NPZ recording files in the recordings directory with pagination.
        """
        files = sorted([f.name for f in self.recs_dir.glob("rec-*.npz")], reverse=True)
        start = page * page_size
        end = start + page_size
        return files[start:end]

    def smooth_recording(
        self, 
        filename: str, 
        window_length: int = 7, 
        polyorder: int = 2
    ) -> tuple[str, np.ndarray]:
        """
        Load a recording, smooth its landmarks, and compute end-effector poses.
        Saves result to the smoothed directory.
        Returns (path to the smoothed file, smoothed_points array of shape (T, L, 3)).
        """
        input_path = self.recs_dir / filename
        if not input_path.exists():
            raise FileNotFoundError(f"Recording file {filename} not found.")

        with np.load(input_path) as data:
            timestamps = data["timestamps"]
            points = data["points"]
            landmark_ids = data["landmark_ids"]

        # Sort by timestamp to ensure monotonic time series for smoothing and plotting
        sort_idx = np.argsort(timestamps)
        timestamps = timestamps[sort_idx]
        points = points[sort_idx]

        if points.size == 0:
            raise ValueError("Recording file is empty.")

        # Backward compat: strip arm landmarks (21, 22) from old files
        hand_mask = landmark_ids < LM.N
        if not hand_mask.all():
            points = points[:, hand_mask, :]
            landmark_ids = landmark_ids[hand_mask]

        # 1. Fill NaN gaps via linear interpolation, then smooth
        filled = interpolate_nans(points)
        smoothed_points = smooth_landmark_sequence(
            filled,
            window_length=window_length,
            polyorder=polyorder,
        )

        # 2. Compute end-effector poses
        T = smoothed_points.shape[0]
        L = smoothed_points.shape[1]

        positions = np.zeros((T, 3), dtype=np.float32)
        rotations = np.zeros((T, 3, 3), dtype=np.float32)
        rpy = np.zeros((T, 3), dtype=np.float32)
        valid = np.zeros(T, dtype=bool)

        for t in range(T):
            current_mapping = {LM(landmark_ids[i]): smoothed_points[t, i] for i in range(L)}
            
            pose = compute_end_effector_pose(current_mapping, int(timestamps[t]))
            
            positions[t] = pose.position
            rotations[t] = pose.R_world_palm
            rpy[t] = pose.rpy_deg
            valid[t] = pose.valid

        # 3. Save to smoothed directory as cln-*.npz
        output_filename = filename.replace("rec-", "cln-")
        output_path = self.smoothed_dir / output_filename
        
        np.savez_compressed(
            output_path,
            positions=positions,
            rotations=rotations,
            rpy=rpy,
            valid=valid,
            timestamps=timestamps,
            raw_points=points.astype(np.float32),
            landmark_ids=landmark_ids,
        )

        if getattr(config, 'SKELETON_SAVE_JSON_DEBUG', False):
            json_path = output_path.with_suffix(".json")
            json_data = []
            for t in range(T):
                frame_pts = {int(landmark_ids[i]): smoothed_points[t, i].tolist() for i in range(L)}
                frame = {
                    "ts": int(timestamps[t]),
                    "landmarks": frame_pts,
                    "end_effector": {
                        "position": positions[t].tolist(),
                        "R_world_palm": rotations[t].tolist(),
                        "rpy_deg": rpy[t].tolist(),
                        "valid": bool(valid[t]),
                        "timestamp_us": int(timestamps[t]),
                    }
                }
                json_data.append(frame)
            with open(json_path, "w") as f:
                json.dump(json_data, f, indent=2)

        return str(output_path), smoothed_points

"""
viki.skeleton.processor
----------------------
Business logic for processing skeleton recording files.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from viki.skeleton import utils
from viki.skeleton.smoothing import smooth_landmark_sequence
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
    ) -> str:
        """
        Load a recording, smooth its landmarks, and compute end-effector poses.
        Saves result to the smoothed directory.
        Returns the path to the smoothed file.
        """
        input_path = self.recs_dir / filename
        if not input_path.exists():
            raise FileNotFoundError(f"Recording file {filename} not found.")

        with np.load(input_path) as data:
            timestamps = data["timestamps"]
            points = data["points"]
            landmark_ids = data["landmark_ids"]

        if points.size == 0:
            raise ValueError("Recording file is empty.")

        # 1. Smooth landmarks
        # points shape: (T, L, 3)
        smoothed_points = smooth_landmark_sequence(
            points,
            window_length=window_length,
            polyorder=polyorder,
        )

        # 2. Compute end-effector poses
        # We need to convert smoothed_points back to a Mapping[LM, np.ndarray] for each frame
        T = smoothed_points.shape[0]
        L = smoothed_points.shape[1]
        
        positions = np.zeros((T, 3), dtype=np.float32)
        rotations = np.zeros((T, 3, 3), dtype=np.float32)
        rpy = np.zeros((T, 3), dtype=np.float32)
        valid = np.zeros(T, dtype=bool)

        for t in range(T):
            # Create mapping for the current frame
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
            timestamps=timestamps
        )

        return str(output_path)

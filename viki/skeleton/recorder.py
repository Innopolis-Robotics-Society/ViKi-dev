"""
viki.skeleton.recorder
--------------------
Handles saving skeleton capture sessions to JSON files.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import List

import json
import numpy as np
from viki.skeleton.models import SkeletonFrame, LM
import viki.config as config


class SkeletonRecorder:
    """
    Records a sequence of SkeletonFrames to a JSON file.
    """

    def __init__(
        self,
        base_dir: str | Path = "data/skeleton_recs",
        filter_indices: list[LM] | None = None,
    ) -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._filter_indices = filter_indices
        self._current_file = None
        self._frames: List[SkeletonFrame] = []

    def start(self) -> str:
        """
        Start a new recording session.
        Returns the filename of the recording.
        """
        self._frames = []
        timestamp = datetime.now().strftime("%H.%M-%d.%m.%Y")
        filename = f"rec-{timestamp}.npz"
        self._current_file = self._base_dir / filename
        return filename

    def record(self, frame: SkeletonFrame) -> None:
        """
        Add a frame to the current recording session.
        """
        if self._current_file is None:
            return

        self._frames.append(frame)

    def stop(self) -> str | None:
        """
        Finalise the recording and write to disk as compressed NumPy arrays.
        Returns the path to the saved file.
        """
        if self._current_file is None:
            return None

        # Determine which indices to save
        indices = self._filter_indices if self._filter_indices else list(LM)
        landmark_ids = np.array([idx.value for idx in indices], dtype=np.int32)
        
        # Extract data
        timestamps = np.array([f.timestamp_us for f in self._frames], dtype=np.int64)
        
        # points shape: (N_frames, N_landmarks, 3)
        points = np.array(
            [[f.points[idx] for idx in indices] for f in self._frames], 
            dtype=np.float32
        )

        np.savez_compressed(
            self._current_file,
            timestamps=timestamps,
            points=points,
            landmark_ids=landmark_ids
        )

        if config.SKELETON_SAVE_JSON_DEBUG:
            json_path = self._current_file.with_suffix(".json")
            json_data = [
                {
                    "ts": f.timestamp_us,
                    "landmarks": {idx.value: f.points[idx].tolist() for idx in indices},
                    "end_effector": f.end_effector.as_dict() if f.end_effector else None
                }
                for f in self._frames
            ]
            with open(json_path, "w") as f:
                json.dump(json_data, f, indent=2)

        path = str(self._current_file)
        self._current_file = None
        self._frames = []
        return path

    @property
    def is_recording(self) -> bool:
        return self._current_file is not None

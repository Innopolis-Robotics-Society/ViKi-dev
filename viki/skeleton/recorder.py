"""
viki.skeleton.recorder
--------------------
Handles saving skeleton capture sessions to JSON files.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List

import numpy as np
from viki.skeleton.models import SkeletonFrame, LM


class SkeletonRecorder:
    """
    Records a sequence of SkeletonFrames to a JSON file.
    """

    def __init__(
        self, base_dir: str | Path = "data/skeleton_recs", filter_indices: list[LM] | None = None
    ) -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._filter_indices = filter_indices
        self._current_file = None
        self._frames: List[dict] = []

    def start(self) -> str:
        """
        Start a new recording session.
        Returns the filename of the recording.
        """
        self._frames = []
        timestamp = int(time.time())
        filename = f"rec_{timestamp}.json"
        self._current_file = self._base_dir / filename
        return filename

    def record(self, frame: SkeletonFrame) -> None:
        """
        Add a frame to the current recording session.
        Discards frames where all landmarks are NaN.
        """
        if self._current_file is None:
            return

        if np.isnan([vec for _, vec in frame.points.items()]).all():
            return

        # Filter and convert numpy arrays to lists for JSON serialization
        if self._filter_indices:
            landmark_data = [frame.points[index].tolist() for index in self._filter_indices]
        else:
            landmark_data = {index.value: vec.tolist() for index, vec in frame.points.items()}

        self._frames.append(
            {
                "ts": frame.timestamp_us,
                "landmarks": landmark_data,
            }
        )

    def stop(self) -> str | None:
        """
        Finalise the recording and write to disk.
        Returns the path to the saved file.
        """
        if self._current_file is None:
            return None

        with open(self._current_file, "w") as f:
            json.dump(self._frames, f, indent=2)

        path = str(self._current_file)
        self._current_file = None
        self._frames = []
        return path

    @property
    def is_recording(self) -> bool:
        return self._current_file is not None

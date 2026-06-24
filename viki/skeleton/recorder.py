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
from viki.skeleton.models import SkeletonFrame


class SkeletonRecorder:
    """
    Records a sequence of SkeletonFrames to a JSON file.
    """
 
    def __init__(self, base_dir: str | Path = "data", filter_indices: list[int] | None = None) -> None:
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

        if np.isnan(frame.landmarks).all():
            return

        # Filter landmarks if filter_indices is provided (e.g. arm only)
        landmarks = frame.landmarks
        source = frame.source
        confidence = frame.confidence
        origin = frame.origin
        
        if self._filter_indices:
            landmarks = landmarks[self._filter_indices]
            source = source[self._filter_indices]
            confidence = confidence[self._filter_indices]
            origin = origin[self._filter_indices]

        # Convert numpy arrays to lists for JSON serialization
        self._frames.append({
            "ts": frame.timestamp_us,
            "landmarks": landmarks.tolist(),
            "source": [str(s) for s in source],
            "confidence": confidence.tolist(),
            "origin": [str(o) for o in origin],
        })


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

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
    Records a sequence of SkeletonFrames to a compressed NPZ file.

    Attributes
    ----------
    _base_dir : Path
        Directory where recordings are saved.
    _filter_indices : list[LM] | None
        Unused; kept for API compatibility.
    _current_file : Path | None
        Path to the currently open recording file.
    _frames : List[SkeletonFrame]
        Buffer of frames for the current recording session.
    """

    def __init__(
        self,
        base_dir: str | Path = "data/skeleton_recs",
        filter_indices: list[LM] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        base_dir : str or Path, default="data/skeleton_recs"
            Root directory for recordings.
        filter_indices : list[LM], optional
            Not used; kept for API compatibility.
        """
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._filter_indices = filter_indices
        self._current_file = None
        self._frames: List[SkeletonFrame] = []

    def start(self) -> str:
        """
        Start a new recording session.

        Returns
        -------
        str
            Filename (e.g., "rec-12.34-12.12.2025.npz") without the full path.
        """
        self._frames = []
        timestamp = datetime.now().strftime("%H.%M-%d.%m.%Y")
        filename = f"rec-{timestamp}.npz"
        self._current_file = self._base_dir / filename
        return filename

    def record(self, frame: SkeletonFrame) -> None:
        """
        Add a frame to the current recording session.

        Parameters
        ----------
        frame : SkeletonFrame
            The frame to append.
        """
        if self._current_file is None:
            return

        self._frames.append(frame)

    def stop(self) -> str | None:
        """
        Finalise the recording and write to disk as compressed NumPy arrays.

        Saves all 23 landmarks; missing ones become NaN.
        If `SKELETON_SAVE_JSON_DEBUG` is True, also saves a JSON version.

        Returns
        -------
        str or None
            Path to the saved NPZ file, or None if no recording was active.
        """
        if self._current_file is None:
            return None

        # Sort frames by timestamp to ensure monotonic time series
        self._frames.sort(key=lambda f: f.timestamp_us)

        all_ids = list(range(LM.N))
        landmark_ids = np.array(all_ids, dtype=np.int32)
        nan3 = np.full(3, np.nan, dtype=np.float32)
        timestamps = np.array([f.timestamp_us for f in self._frames], dtype=np.int64)

        points = np.array(
            [[f.points.get(LM(idx), nan3) for idx in all_ids] for f in self._frames],
            dtype=np.float32,
        )

        np.savez_compressed(
            self._current_file,
            timestamps=timestamps,
            points=points,
            landmark_ids=landmark_ids,
        )

        if getattr(config, 'SKELETON_SAVE_JSON_DEBUG', False):
            json_path = self._current_file.with_suffix(".json")
            json_data = [
                {
                    "ts": f.timestamp_us,
                    "landmarks": {
                        idx: f.points.get(LM(idx), nan3).tolist() for idx in all_ids
                    },
                    "end_effector": f.end_effector.as_dict() if f.end_effector else None,
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
        """True if a recording session is currently active."""
        return self._current_file is not None

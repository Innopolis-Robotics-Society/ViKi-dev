"""
viki.skeleton.utils
------------------
Utilities for converting between skeleton frame formats and dense arrays.
"""

from __future__ import annotations

import numpy as np
from viki.skeleton.models import LM

def _landmark_index(raw_index) -> int | None:
    try:
        index = int(raw_index)
    except (TypeError, ValueError):
        return None
    if index < 0 or index >= LM.N:
        return None
    return index

def _landmark_vector(raw_vec) -> np.ndarray | None:
    try:
        vec = np.asarray(raw_vec, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if vec.shape != (3,):
        return None
    return vec

def frames_to_dense(frames: list[dict], filter_indices: list[LM] | None = None) -> np.ndarray:
    """
    Convert a list of frames (dict format) to a dense numpy array (T, L, 3).
    """
    dense = np.full((len(frames), LM.N, 3), np.nan, dtype=np.float64)

    for frame_idx, frame in enumerate(frames):
        landmarks = frame.get("landmarks", {})
        if isinstance(landmarks, dict):
            for raw_index, raw_vec in landmarks.items():
                index = _landmark_index(raw_index)
                vec = _landmark_vector(raw_vec)
                if index is not None and vec is not None:
                    dense[frame_idx, index, :] = vec
        elif isinstance(landmarks, list):
            for pos, raw_vec in enumerate(landmarks):
                index = _list_landmark_index(pos, filter_indices)
                vec = _landmark_vector(raw_vec)
                if index is not None and vec is not None:
                    dense[frame_idx, index, :] = vec

    return dense

def _list_landmark_index(position: int, filter_indices: list[LM] | None) -> int | None:
    if filter_indices:
        if position >= len(filter_indices):
            return None
        return int(filter_indices[position])
    if position >= LM.N:
        return None
    return position

def replace_landmarks(frames: list[dict], smoothed: np.ndarray, filter_indices: list[LM] | None = None) -> list[dict]:
    """
    Replace landmarks in original frames with smoothed ones from a dense array.
    """
    out_frames: list[dict] = []

    for frame_idx, frame in enumerate(frames):
        next_frame = dict(frame)
        landmarks = frame.get("landmarks", {})
        if isinstance(landmarks, dict):
            next_frame["landmarks"] = _replace_dict_landmarks(
                landmarks,
                smoothed[frame_idx],
            )
        elif isinstance(landmarks, list):
            next_frame["landmarks"] = _replace_list_landmarks(
                landmarks,
                smoothed[frame_idx],
                filter_indices,
            )
        out_frames.append(next_frame)

    return out_frames

def _replace_dict_landmarks(landmarks: dict, smoothed_frame: np.ndarray) -> dict:
    out = {}
    for raw_index, raw_vec in landmarks.items():
        index = _landmark_index(raw_index)
        if index is None or _landmark_vector(raw_vec) is None:
            out[raw_index] = raw_vec
        else:
            out[raw_index] = smoothed_frame[index].tolist()
    return out

def _replace_list_landmarks(landmarks: list, smoothed_frame: np.ndarray, filter_indices: list[LM] | None) -> list:
    out = []
    for pos, raw_vec in enumerate(landmarks):
        index = _list_landmark_index(pos, filter_indices)
        if index is None or _landmark_vector(raw_vec) is None:
            out.append(raw_vec)
        else:
            out.append(smoothed_frame[index].tolist())
    return out

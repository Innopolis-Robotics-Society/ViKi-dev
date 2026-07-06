"""
viki.skeleton.detectors.hand_pose
---------------------------------
Partial detector that emits 21 hand keypoints from MediaPipe HandLandmarker.
Writes into global slots 0..20 (WRIST..PINKY_TIP).

MediaPipe Hand landmark indices reference:
https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from viki.skeleton.detectors.base import (
    PartialDetection2D,
    PartialLandmarkDetector,
)
from viki.skeleton.detectors.mediapipe_base import (
    MODELS_DIR_DEFAULT,
    MediaPipeTaskRunner,
    ensure_model,
)
from viki.skeleton.models import PreparedFrame

_HAND_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

_LABEL_RIGHT = "Right"
_LABEL_LEFT = "Left"


class MediaPipeHand(PartialLandmarkDetector):
    """Partial detector for one hand."""

    name = "hand"
    indices = tuple(range(21))
    priority = 10

    def __init__(
        self,
        hand: Literal["right", "left"] = "right",
        mode: Literal["image", "video", "live"] = "image",
        hand_model: Optional[str] = None,
        models_dir: str = MODELS_DIR_DEFAULT,
        min_hand_confidence: float = 0.5,
    ) -> None:
        """
        parameters
        ----------
        hand                : "right" or "left" — which hand to keep.
        mode                : MediaPipe running mode ("image" / "video" / "live").
        hand_model          : explicit path to a hand_landmarker.task; auto-
                              downloaded into `models_dir` when None.
        models_dir          : local cache directory for downloaded models.
        min_hand_confidence : threshold reused for detection, presence,
                              and tracking confidence.
        """
        super().__init__()

        self._hand = hand
        self._target_label = _LABEL_RIGHT if hand == "right" else _LABEL_LEFT

        model_path = hand_model or ensure_model(
            "hand_landmarker.task",
            _HAND_URL,
            models_dir,
        )

        # Detector-specific config stays in the closure; runner only sees
        # shared infrastructure pieces.
        def _factory(base_options, running_mode, result_callback):
            from mediapipe.tasks.python import vision

            opts = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=running_mode,
                num_hands=1,
                min_hand_detection_confidence=min_hand_confidence,
                min_hand_presence_confidence=min_hand_confidence,
                min_tracking_confidence=min_hand_confidence,
                **({"result_callback": result_callback} if result_callback else {}),
            )
            return vision.HandLandmarker.create_from_options(opts)

        self._runner = MediaPipeTaskRunner(_factory, model_path, mode)

    def detect(self, frame: PreparedFrame) -> Optional[PartialDetection2D]:
        """
        parameters
        ----------
        frame : prepared camera frame (RGB + depth + K).

        returns
        -------
        PartialDetection2D over slots 0..20, or None when no hand of the
        requested handedness was detected (or LIVE result is not ready yet).
        """
        raw = self._runner.submit(frame.rgb, frame.timestamp_us)
        if raw is None or not raw.hand_landmarks:
            return None
        return self._extract(raw, frame)

    def close(self) -> None:
        """Release the underlying MediaPipe task."""
        self._runner.close()

    def _extract(self, raw, frame: PreparedFrame) -> Optional[PartialDetection2D]:
        """
        Build PartialDetection2D from a raw HandLandmarkerResult.
        """
        # Iterate handedness to find the hand whose label matches our target.
        match_idx: Optional[int] = None
        match_score: float = 0.0
        for i, handedness_list in enumerate(raw.handedness):
            label = handedness_list[0].category_name
            if label == self._target_label:
                match_idx = i
                match_score = float(handedness_list[0].score)
                break
        if match_idx is None:
            return None

        h, w = frame.rgb.shape[:2]
        lms = raw.hand_landmarks[match_idx]  # 21 NormalizedLandmark

        n = len(self.indices)
        px = np.zeros((n, 2), dtype=np.float32)
        z = np.zeros(n, dtype=np.float32)
        conf = np.full(n, match_score, dtype=np.float32)
        
        for k, idx in enumerate(self.indices):
            lm = lms[idx]
            px[k] = (lm.x * w, lm.y * h)
            z[k] = lm.z

        return PartialDetection2D(
            indices=self.indices,
            px=px,
            lm_z_rel=z,
            per_index_confidence=conf,
            device_id=frame.device_id,
            timestamp_us=frame.timestamp_us,
        )

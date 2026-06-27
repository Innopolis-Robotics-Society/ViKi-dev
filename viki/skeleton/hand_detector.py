"""
viki.skeleton.hand_detector
---------------------------
MediaPipe Tasks API wrapper for hand + pose detection.
Produces 23-landmark HandDetection: 21 hand landmarks + elbow + shoulder.

Running modes
-------------
IMAGE  - per-image detection.  
VIDEO  — uses temporal coherence between frames (tracking).
LIVE   — fully asynchronous, non-blocking.

Handedness
----------
MediaPipe was trained on selfie (mirrored) images.
So two modes implemented for test is mirrored=True and mirrored=False for main usage
"""

from __future__ import annotations

import threading
import urllib.request
from pathlib import Path
from typing import Literal, Optional
import logging

logger = logging.getLogger(__name__)

import numpy as np

from viki.skeleton.models import HandDetection, LM, PreparedFrame
import viki.config

# MediaPipe Pose landmark indices (33-point full-body model)
_POSE_LEFT_SHOULDER  = 11
_POSE_RIGHT_SHOULDER = 12
_POSE_LEFT_ELBOW     = 13
_POSE_RIGHT_ELBOW    = 14
_POSE_LEFT_WRIST     = 15
_POSE_RIGHT_WRIST    = 16

_MODEL_URLS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ),
    "pose_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
    ),
}


def _ensure_model(filename: str, models_dir: str | Path = "models") -> str:
    """Download model file if not present. Returns absolute path string."""
    path = Path(models_dir) / filename
    if not path.exists():
        url = _MODEL_URLS[filename]
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[HandDetector] Downloading {filename} …")
        urllib.request.urlretrieve(url, path)
        print(f"[HandDetector] Saved to {path}")
    return str(path)


class HandDetector:
    """
    Runs MediaPipe HandLandmarker + PoseLandmarker and merges
    results into 23 pixel-space landmarks.

    Parameters
    ----------
    hand : {"right", "left"}
        Which hand to track.
    mode : {"image", "video", "live"}
        "image" — independent per-frame detection (default).
        "video" — tracking mode, faster on video streams.
                  Timestamps must be strictly increasing (taken from frame.timestamp_us).
        "live"  — async non-blocking mode. detect() returns previous result immediately.
    hand_model : str | None
        Path to hand_landmarker.task. Auto-downloaded if not present.
    pose_model : str | None
        Path to pose_landmarker.task. Auto-downloaded if not present.
    models_dir : str
        Directory for auto-downloaded models. Default: "models/".
    min_hand_confidence : float
        Detection/tracking confidence threshold for hands [0, 1].
    min_pose_confidence : float
        Detection/tracking confidence threshold for pose [0, 1].
    mirrored : bool
        True  — selfie/front camera.
        False — rear/fixed camera (Kinect)
    """

    def __init__(
        self,
        hand: Literal["right", "left"] = viki.config.HAND_TO_DETECT, #TODO move this selection to frontend
        mode: Literal["image", "video", "live"] = "image",
        hand_model: str | None = None,
        pose_model: str | None = None,
        models_dir: str = "models",
        min_hand_confidence: float = 0.5,
        min_pose_confidence: float = 0.3,
        mirrored: bool = viki.config.CAMERAS_MIRRORED
    ) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
 
        self._hand = hand
        self._mode = mode
        self._detection_flag = False
 
        if mirrored:
            self._target_label = "Left" if hand == "right" else "Right"
        else:
            self._target_label = "Right" if hand == "right" else "Left" 

        hand_path = hand_model or _ensure_model("hand_landmarker.task", models_dir)
        pose_path = pose_model or _ensure_model("pose_landmarker.task", models_dir)

        if mode == "video":
            running_mode = vision.RunningMode.VIDEO
        elif mode == "live":
            running_mode = vision.RunningMode.LIVE_STREAM
        else:
            running_mode = vision.RunningMode.IMAGE


        
        self._lock = threading.Lock()
        self._live_hand_result = None
        self._live_pose_result = None 
        self._live_last_ts_ms  = -1  # timestamp of last result returned to caller
        self._last_timestamp_ms = -1 # track last timestamp sent to MediaPipe (for video mode)

        def _hand_cb(result, _img, _ts):

            with self._lock:
                self._live_hand_result = result

        def _pose_cb(result, _img, _ts):
            with self._lock:
                self._live_pose_result = result
                

        hand_path = hand_model or _ensure_model("hand_landmarker.task", models_dir)
        hand_opts = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=hand_path),
            running_mode=running_mode,
            num_hands=1,
            min_hand_detection_confidence=min_hand_confidence,
            min_hand_presence_confidence=min_hand_confidence,
            min_tracking_confidence=min_hand_confidence,
            **({"result_callback": _hand_cb} if mode == "live" else {}),
        )
        self._hands = vision.HandLandmarker.create_from_options(hand_opts)

        pose_opts = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=pose_path),
            running_mode=running_mode,
            min_pose_detection_confidence=min_pose_confidence,
            min_pose_presence_confidence=min_pose_confidence,
            min_tracking_confidence=min_pose_confidence,
            **({"result_callback": _pose_cb} if mode == "live" else {}),
        )
        self._pose = vision.PoseLandmarker.create_from_options(pose_opts)
        self._mp = mp

    def detect(self, frame: Optional[PreparedFrame]) -> Optional[HandDetection]:
        """
        Run detection on a PreparedFrame.
 
        IMAGE — blocking, returns result immediately.
        VIDEO — blocking with tracking; frame.timestamp_us must be strictly increasing.
        LIVE  — non-blocking; submits frame async and returns the PREVIOUS result.
                First call always returns None (no previous result yet).
 
        Returns None if hand is not detected.
        """
        if frame is None:
            return None
 
        h, w = frame.rgb.shape[:2]
 
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame.rgb),
        )
        timestamp_ms = frame.timestamp_us // 1000

 
        if self._mode == "video":
            # MediaPipe requires strictly increasing timestamps.
            if timestamp_ms <= self._last_timestamp_ms:
                timestamp_ms = self._last_timestamp_ms + 1
            self._last_timestamp_ms = timestamp_ms
 
            hand_result = self._hands.detect_for_video(mp_image, timestamp_ms) if self._hands else None
            pose_result = self._pose.detect_for_video(mp_image, timestamp_ms)

            
        elif self._mode == "live":
            # Submit async — results arrive in callbacks, return last known result
            if timestamp_ms <= self._last_timestamp_ms:
                timestamp_ms = self._last_timestamp_ms + 1
            self._last_timestamp_ms = timestamp_ms

            if self._hands:
                self._hands.detect_async(mp_image, timestamp_ms)
            
            self._pose.detect_async(mp_image, timestamp_ms)
            
            with self._lock:
                
                hand_result = self._live_hand_result
                pose_result = self._live_pose_result
                last_ts     = self._live_last_ts_ms
            if (hand_result is None) or pose_result is None:
                return None  # no result yet (first frame)
            # Callback hasn't fired for this frame yet — stale result, skip
            if last_ts == timestamp_ms:
                return None
            with self._lock:
                self._live_last_ts_ms = timestamp_ms

        else:  # image
            hand_result = self._hands.detect(mp_image) if self._hands else None
            pose_result = self._pose.detect(mp_image)
            print(f"DEBUG: MediaPipe raw results - hand: {hand_result is not None}, pose: {pose_result is not None}")

        # Debug logging for detection results
        hand_px, hand_z, confidence = self._extract_hand(hand_result, w, h)
        if hand_px is None:
            if self._detection_flag == True:
                # print(f"DEBUG: Pose detection failed: pose_result is {pose_result is None}")
                logger.debug(f"stopped seeing {self._hand} hand")
                self._detection_flag = False
            return None

        pose_px, pose_z = self._extract_pose(pose_result, w, h)
        if pose_px is None:
            if self._detection_flag == True:
                # print(f"DEBUG: Pose detection failed: pose_result is {pose_result is None}")
                logger.debug(f"pose of {self._hand} hand not detected")
                self._detection_flag = False
            return None
        
        px, lm_z_rel = self._merge(hand_px, hand_z, pose_px, pose_z)
        if self._detection_flag == False:
            logger.debug(f"seeing {self._hand} hand")
            self._detection_flag = True
        
        return HandDetection(
            px=px,
            lm_z_rel=lm_z_rel,
            confidence=confidence,
            device_id=frame.device_id,
            timestamp_us=frame.timestamp_us,
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()
        self._pose.close()

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def _extract_hand(
        self, result, w: int, h: int
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """Returns (px (21,2), z_rel (21,), confidence) or (None, None, 0.0)."""
        if not result.hand_landmarks:
            return None, None, 0.0

        # mark: handedness list of lists, one arm, so single inner list interesting for us
        for i, handedness_list in enumerate(result.handedness):
            label = handedness_list[0].category_name
            score = handedness_list[0].score
            if label == self._target_label:
                lms = result.hand_landmarks[i] # here 21 landmarks
                px = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)
                z  = np.array([lm.z for lm in lms], dtype=np.float32) # relative z
                return px, z, float(score)

        return None, None, 0.0

    def _extract_pose(
        self, result, w: int, h: int
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Returns (px (3,2), z_rel (3,)) for [wrist, elbow, shoulder] or (None, None)."""
        if result is None:
            print("DEBUG: _extract_pose: result is None")
            return None, None
        if not result.pose_landmarks:
            # print(f"DEBUG: _extract_pose: pose_landmarks is empty. Result type: {type(result)}")
            return None, None

        lms = result.pose_landmarks[0] # take only one person; 33 points

        if self._hand == "right":
            indices = [_POSE_RIGHT_WRIST, _POSE_RIGHT_ELBOW, _POSE_RIGHT_SHOULDER]
        else:
            indices = [_POSE_LEFT_WRIST, _POSE_LEFT_ELBOW, _POSE_LEFT_SHOULDER]

        px = np.array([[lms[i].x * w, lms[i].y * h] for i in indices], dtype=np.float32)
        z  = np.array([lms[i].z for i in indices], dtype=np.float32)
        return px, z

    @staticmethod
    def _merge(
        hand_px: np.ndarray | None,       # (21, 2)
        hand_z:  np.ndarray | None,       # (21,)
        pose_px: np.ndarray | None,       # (3, 2) [wrist, elbow, shoulder] or None
        pose_z:  np.ndarray | None,       # (3,) or None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build final (23, 2) px and (23,) z_rel arrays.
        
          [0]     WRIST     — overridden with Pose wrist if Pose available
          [1-20]  hand landmarks unchanged
          [21]    ELBOW     — from Pose, or nan if Pose not detected
          [22]    SHOULDER  — from Pose, or nan if Pose not detected
        """
        if hand_px is not None:
            px = hand_px.copy()
            z  = hand_z.copy()
        else:
            px = np.full((21, 2), np.nan, dtype=np.float32)
            z  = np.full(21, np.nan, dtype=np.float32)

        if pose_px is not None and pose_z is not None:
            # Override wrist (index 0) with pose wrist for arm-hand continuity
            px[0] = pose_px[0]
            z[0] = pose_z[0]
            
            elbow_px    = pose_px[1]
            shoulder_px = pose_px[2]
            elbow_z     = pose_z[1]
            shoulder_z  = pose_z[2]
        else:
            elbow_px    = np.full(2, np.nan, dtype=np.float32)
            shoulder_px = np.full(2, np.nan, dtype=np.float32)
            elbow_z     = np.nan
            shoulder_z  = np.nan

        px = np.vstack([px, elbow_px, shoulder_px])   # (23, 2)
        z  = np.append(z, [elbow_z, shoulder_z])       # (23,)

        return px, z

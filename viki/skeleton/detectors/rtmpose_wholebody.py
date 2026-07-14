"""
viki.skeleton.detectors.rtmpose_wholebody
-----------------------------------------
Partial detector that emits all 23 ViKi skeleton slots from a single
RTMPose Wholebody model (via rtmlib).

Output layout: writes into global slots 0..22, i.e. WRIST + 20 finger
landmarks + ELBOW + SHOULDER, all from one inference call.

COCO-WholeBody keypoint layout (133 total):
    0..16   body-17    (5=L-shoulder, 6=R-shoulder, 7=L-elbow, 8=R-elbow,
                        9=L-wrist,    10=R-wrist)
    17..22  feet-6
    23..90  face-68
    91..111 left hand  (91=wrist, 92..111=20 finger points, standard hand order)
    112..132 right hand (112=wrist, 113..132=20 finger points)

The hand-block wrist (91 / 112) is used as ViKi's WRIST slot, not the
body wrist, because it comes out of the same head as the fingers and
therefore aligns kinematically with them.
"""

from __future__ import annotations

from typing import Literal, Optional

import cv2
import numpy as np

from viki.skeleton.detectors.base import PartialDetection2D, PartialLandmarkDetector
from viki.skeleton.models import LM, PreparedFrame


# Anatomical-side mapping: (hand-block base index, elbow index, shoulder index).
_SIDE_MAP = {
    "right": (112, 8, 6),
    "left": (91, 7, 5),
}


def _build_slot_to_wb(hand: Literal["right", "left"]) -> np.ndarray:
    """
    Return a (23,) int array: slot_to_wb[viki_slot] = coco_wholebody_index.

    Slots 0..20 → hand block (base + 0..20, standard hand order).
    Slot 21   → elbow.
    Slot 22   → shoulder.
    """
    hand_base, elbow, shoulder = _SIDE_MAP[hand]
    m = np.empty(LM.N, dtype=np.int64)
    for i in range(21):
        m[i] = hand_base + i
    m[LM.ELBOW] = elbow
    m[LM.SHOULDER] = shoulder
    return m


class RTMPoseWholeBody(PartialLandmarkDetector):
    """
    Whole-body RTMPose partial detector.

    Fills every slot 0..22 in a single inference pass.

    Attributes
    ----------
    name : str
        "rtmpose_wholebody".
    indices : tuple[int, ...]
        (0, 1, ..., 22) — the full ViKi layout.
    priority : int
        0. There is no other detector to fight with in the RTMPose stack;
        priority is here only to satisfy the base class contract.
    """

    name = "rtmpose_wholebody"
    indices = tuple(range(LM.N))
    priority = 0

    def __init__(
        self,
        hand: Literal["right", "left"] = "right",
        model_mode: Literal["lightweight", "balanced", "performance"] = "balanced",
        device: Literal["cpu", "cuda", "mps"] = "cpu",
        min_confidence: float = 0.3,
    ) -> None:
        """
        Parameters
        ----------
        hand : "right" | "left"
            Anatomical side of the person to track.
        model_mode : "lightweight" | "balanced" | "performance"
            rtmlib preset trading accuracy for speed. "balanced" is the
            rtmlib default (RTMW-l @ 256x192).
        device : "cpu" | "cuda" | "mps"
            ONNXRuntime execution provider.
        min_confidence : float
            Per-keypoint score threshold. Below this the (u,v) is emitted
            as NaN so the composite treats it as missing.
        """
        super().__init__()

        # Lazy: importing rtmlib triggers onnxruntime init + model resolution;
        # we want that cost paid on construction, not on module import.
        from rtmlib import Wholebody

        self._hand = hand
        self._min_confidence = float(min_confidence)
        self._slot_to_wb = _build_slot_to_wb(hand)
        self._wholebody = Wholebody(mode=model_mode, device=device)

    def detect(self, frame: PreparedFrame) -> Optional[PartialDetection2D]:
        """
        Run whole-body inference on one prepared frame.

        Parameters
        ----------
        frame : PreparedFrame
            RGB is expected in ``frame.rgb`` (H, W, 3) uint8.

        Returns
        -------
        Optional[PartialDetection2D]
            Detection over all 23 slots, or None when no person is found.
            Per-slot confidences below ``min_confidence`` come back as NaN.
        """
        # rtmlib was written against OpenCV convention (BGR). PreparedFrame
        # holds RGB, so flip channels for the model.
        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        kpts, scores = self._wholebody(bgr)  # (N, 133, 2), (N, 133)

        if kpts.shape[0] == 0:
            return None

        # Pick the person with the highest mean score across our 23 slots
        # of interest — more robust than "first" for multi-person frames.
        mean_scores = scores[:, self._slot_to_wb].mean(axis=1)
        person = int(np.argmax(mean_scores))
        pts = kpts[person]        # (133, 2) float64
        confs = scores[person]    # (133,)   float32

        n = LM.N
        px = np.full((n, 2), np.nan, dtype=np.float32)
        per_conf = np.zeros(n, dtype=np.float32)
        for slot in range(n):
            wb = int(self._slot_to_wb[slot])
            c = float(confs[wb])
            if c < self._min_confidence:
                continue
            px[slot] = pts[wb].astype(np.float32)
            per_conf[slot] = c

        # RTMPose is a 2-D detector — no relative-z per landmark. Use zeros;
        # downstream lifting handles a NaN or 0 z_rel gracefully.
        return PartialDetection2D(
            indices=self.indices,
            px=px,
            lm_z_rel=np.zeros(n, dtype=np.float32),
            per_index_confidence=per_conf,
            device_id=frame.device_id,
            timestamp_us=frame.timestamp_us,
        )

    def close(self) -> None:
        """Release ONNXRuntime sessions."""
        # rtmlib's Wholebody wraps two rtmlib.tools.* models; ORT sessions
        # are freed when the wrapper is garbage-collected. Explicitly
        # drop the reference so a caller can rebuild fresh.
        self._wholebody = None

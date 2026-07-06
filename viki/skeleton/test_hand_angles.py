"""Tests for skeleton hand orientation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from viki.skeleton.detectors import MediaPipeArm, MediaPipeHand
from viki.skeleton.hand_angles import compute_end_effector_pose, compute_palm_rotation
from viki.skeleton.models import LM, SkeletonFrame
from viki.skeleton.recorder import SkeletonRecorder


def synthetic_points() -> dict[LM, np.ndarray]:
    points = {LM(i): np.zeros(3, dtype=np.float32) for i in range(LM.N)}
    points[LM.WRIST] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    points[LM.THUMB_CMC] = np.array([1.0, 3.0, 3.0], dtype=np.float32)
    points[LM.MIDDLE_MCP] = np.array([2.0, 2.0, 3.0], dtype=np.float32)
    return points


class HandAnglesTests(unittest.TestCase):
    def test_schema_preserves_arm_landmarks(self) -> None:
        self.assertEqual(LM.N, 23)
        self.assertEqual(LM.ELBOW, 21)
        self.assertEqual(LM.SHOULDER, 22)
        self.assertEqual(MediaPipeHand.indices, tuple(range(21)))
        self.assertEqual(MediaPipeArm.indices, (0, 21, 22))

    def test_palm_rotation_is_orthonormal_right_handed(self) -> None:
        points = synthetic_points()
        rotation = compute_palm_rotation(
            points[LM.WRIST],
            points[LM.THUMB_CMC],
            points[LM.MIDDLE_MCP],
        )

        self.assertIsNotNone(rotation)
        assert rotation is not None
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=6)
        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-6)

    def test_palm_rotation_rejects_invalid_inputs(self) -> None:
        points = synthetic_points()
        self.assertIsNone(compute_palm_rotation(points[LM.WRIST], points[LM.WRIST], points[LM.MIDDLE_MCP]))
        self.assertIsNone(compute_palm_rotation(points[LM.WRIST], points[LM.MIDDLE_MCP], points[LM.MIDDLE_MCP]))
        bad = points[LM.THUMB_CMC].copy()
        bad[0] = np.nan
        self.assertIsNone(compute_palm_rotation(points[LM.WRIST], bad, points[LM.MIDDLE_MCP]))

    def test_end_effector_pose_and_recorder_json(self) -> None:
        points = synthetic_points()
        pose = compute_end_effector_pose(points, timestamp_us=123)
        self.assertTrue(pose.valid)

        with tempfile.TemporaryDirectory() as tmp:
            recorder = SkeletonRecorder(base_dir=tmp)
            filename = recorder.start()
            recorder.record(SkeletonFrame(points=points, timestamp_us=123, end_effector=pose))
            saved = recorder.stop()

            self.assertIsNotNone(saved)
            self.assertTrue(filename.startswith("rec_"))
            data = json.loads(Path(saved).read_text(encoding="utf-8"))

        self.assertIn("landmarks", data[0])
        self.assertIn("end_effector", data[0])
        self.assertTrue(data[0]["end_effector"]["valid"])
        np.testing.assert_allclose(data[0]["end_effector"]["R_world_palm"], np.eye(3), atol=1e-6)


if __name__ == "__main__":
    unittest.main()

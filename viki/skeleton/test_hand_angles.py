"""Tests for skeleton hand orientation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from viki.skeleton.detectors import RTMPoseWholeBody
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
        # The whole-body detector covers every ViKi slot in one call.
        self.assertEqual(RTMPoseWholeBody.indices, tuple(range(23)))

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

    def test_end_effector_pose_and_recorder_npz(self) -> None:
        points = synthetic_points()
        pose = compute_end_effector_pose(points, timestamp_us=123)
        self.assertTrue(pose.valid)

        with tempfile.TemporaryDirectory() as tmp:
            recorder = SkeletonRecorder(base_dir=tmp)
            filename = recorder.start()
            recorder.record(SkeletonFrame(points=points, timestamp_us=123, end_effector=pose))
            saved = recorder.stop()

            self.assertIsNotNone(saved)
            self.assertTrue(filename.startswith("rec-"))
            self.assertTrue(str(saved).endswith(".npz"))

            with np.load(saved) as data:
                self.assertIn("timestamps", data)
                self.assertIn("points", data)
                self.assertIn("landmark_ids", data)
                self.assertEqual(data["points"].shape, (1, LM.N, 3))
                self.assertEqual(data["landmark_ids"].tolist(), list(range(LM.N)))
                np.testing.assert_allclose(data["points"][0, LM.WRIST.value], [1.0, 2.0, 3.0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()

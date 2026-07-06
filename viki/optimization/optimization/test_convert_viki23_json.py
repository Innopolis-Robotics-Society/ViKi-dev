"""Tests for ViKi-dev hand skeleton JSON conversion."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from viki.optimization.optimization.convert_viki23_json import convert


class ConvertVikiHandJsonTests(unittest.TestCase):
    def test_hand_only_conversion_maps_wrist_and_orientation_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "rec_test.json"
            output_path = root / "sample.npz"
            landmarks = [[0.0, 0.0, 0.0] for _ in range(21)]
            landmarks[0] = [1.0, 2.0, 3.0]
            landmarks[1] = [1.0, 3.0, 3.0]
            landmarks[9] = [2.0, 2.0, 3.0]
            input_path.write_text(
                json.dumps([{"ts": 1_000_000, "landmarks": landmarks}]),
                encoding="utf-8",
            )

            summary = convert(input_path, output_path, hand="right")
            with np.load(output_path, allow_pickle=True) as data:
                self.assertEqual(data["body"].shape, (1, 33, 3))
                self.assertEqual(data["right_hand"].shape, (1, 21, 3))
                np.testing.assert_allclose(data["body"][:, 16, :], [[1.0, 2.0, 3.0]])
                np.testing.assert_allclose(data["right_hand"][0], np.asarray(landmarks))
                self.assertNotIn("include_arm", data.files)
                self.assertTrue(bool(data["orientation_valid"][0]))

            self.assertEqual(summary["frames"], 1)
            self.assertEqual(summary["orientation_valid_frames"], 1)

    def test_legacy_23_landmark_recording_ignores_elbow_and_shoulder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "rec_legacy.json"
            output_path = root / "sample.npz"
            landmarks = [[0.0, 0.0, 0.0] for _ in range(23)]
            landmarks[0] = [1.0, 2.0, 3.0]
            landmarks[1] = [1.0, 3.0, 3.0]
            landmarks[9] = [2.0, 2.0, 3.0]
            landmarks[21] = [100.0, 100.0, 100.0]
            landmarks[22] = [200.0, 200.0, 200.0]
            input_path.write_text(
                json.dumps([{"ts": 1_000_000, "landmarks": landmarks}]),
                encoding="utf-8",
            )

            convert(input_path, output_path, hand="right", include_arm=True)
            with np.load(output_path, allow_pickle=True) as data:
                np.testing.assert_allclose(data["right_hand"][0], np.asarray(landmarks[:21]))
                self.assertNotIn("include_arm", data.files)
                self.assertFalse(np.any(data["body"] == 100.0))
                self.assertFalse(np.any(data["body"] == 200.0))

    def test_orientation_valid_uses_raw_recording_before_interpolation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "rec_missing.json"
            output_path = root / "sample.npz"
            valid_landmarks = [[0.0, 0.0, 0.0] for _ in range(21)]
            valid_landmarks[0] = [1.0, 2.0, 3.0]
            valid_landmarks[1] = [1.0, 3.0, 3.0]
            valid_landmarks[9] = [2.0, 2.0, 3.0]
            missing_landmarks = list(valid_landmarks)
            missing_landmarks[1] = [float("nan"), float("nan"), float("nan")]
            input_path.write_text(
                json.dumps(
                    [
                        {"ts": 1_000_000, "landmarks": valid_landmarks},
                        {"ts": 1_033_333, "landmarks": missing_landmarks},
                    ]
                ),
                encoding="utf-8",
            )

            convert(input_path, output_path, hand="right")
            with np.load(output_path, allow_pickle=True) as data:
                np.testing.assert_array_equal(data["orientation_valid"], [True, False])
                self.assertTrue(np.isfinite(data["right_hand"][1]).all())


if __name__ == "__main__":
    unittest.main()

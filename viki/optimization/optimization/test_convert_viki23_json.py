"""Tests for ViKi-dev skeleton JSON conversion."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from viki.optimization.optimization.convert_viki23_json import convert


class ConvertViki23JsonTests(unittest.TestCase):
    def test_wrist_only_conversion_ignores_arm_landmarks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "rec_test.json"
            output_path = root / "sample.npz"
            wrist = [1.0, 2.0, 3.0]
            elbow = [100.0, 100.0, 100.0]
            shoulder = [200.0, 200.0, 200.0]
            landmarks = [[0.0, 0.0, 0.0] for _ in range(23)]
            landmarks[0] = wrist
            landmarks[21] = elbow
            landmarks[22] = shoulder
            input_path.write_text(
                json.dumps(
                    [
                        {"ts": 1_000_000, "landmarks": landmarks},
                        {"ts": 1_100_000, "landmarks": landmarks},
                    ]
                ),
                encoding="utf-8",
            )

            summary = convert(input_path, output_path, hand="right", include_arm=False)
            with np.load(output_path, allow_pickle=True) as data:
                body = data["body"]

                self.assertEqual(body.shape, (2, 33, 3))
                expected_wrist = np.repeat([wrist], len(body), axis=0)
                np.testing.assert_allclose(body[:, 16, :], expected_wrist)
                np.testing.assert_allclose(body[:, 14, :], expected_wrist)
                np.testing.assert_allclose(body[:, 12, :], expected_wrist)
                self.assertEqual(data["working_hand"].item(), "right")
                self.assertFalse(bool(data["include_arm"].item()))

            self.assertEqual(summary["frames"], 2)
            self.assertEqual(summary["working_hand"], "right")

    def test_include_arm_copies_elbow_and_shoulder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "rec_test.json"
            output_path = root / "sample.npz"
            landmarks = [[0.0, 0.0, 0.0] for _ in range(23)]
            landmarks[0] = [1.0, 2.0, 3.0]
            landmarks[21] = [4.0, 5.0, 6.0]
            landmarks[22] = [7.0, 8.0, 9.0]
            input_path.write_text(
                json.dumps([{"ts": 1_000_000, "landmarks": landmarks}]),
                encoding="utf-8",
            )

            convert(input_path, output_path, hand="right", include_arm=True)
            with np.load(output_path, allow_pickle=True) as data:
                body = data["body"]
                np.testing.assert_allclose(body[:, 16, :], [[1.0, 2.0, 3.0]])
                np.testing.assert_allclose(body[:, 14, :], [[4.0, 5.0, 6.0]])
                np.testing.assert_allclose(body[:, 12, :], [[7.0, 8.0, 9.0]])
                self.assertTrue(bool(data["include_arm"].item()))


if __name__ == "__main__":
    unittest.main()

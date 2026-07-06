"""Retargeting tests that do not require PINK/Pinocchio."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from viki.optimization.optimization.retarget_rgb_only import (
    R_DEFAULT,
    build_parser,
    build_run_config,
    effective_orientation_cost,
    load_landmarks,
    normalize_robot,
    output_traj_path,
    should_apply_legacy_transform,
    transform_points,
)


class RetargetLogicTests(unittest.TestCase):
    def test_parser_accepts_wrist_position_command(self) -> None:
        args = build_parser().parse_args(
            [
                "--sample",
                "sample.npz",
                "--out",
                "out",
                "--target-mode",
                "wrist_position",
                "--robot",
                "ur10",
            ]
        )
        self.assertEqual(args.target_mode, "wrist_position")
        self.assertEqual(args.robot, "ur10")

    def test_wrist_position_forces_zero_orientation_cost(self) -> None:
        args = build_parser().parse_args(
            [
                "--sample",
                "sample.npz",
                "--out",
                "out",
                "--target-mode",
                "wrist_position",
                "--ik-orientation-cost",
                "0.5",
            ]
        )
        cfg = build_run_config(args, normalize_robot("ur10"), 1.0, 0.5, 20, 0)
        self.assertEqual(effective_orientation_cost(cfg), 0.0)

    def test_coordinate_frame_controls_legacy_transform(self) -> None:
        self.assertFalse(should_apply_legacy_transform("robot_base"))
        self.assertTrue(should_apply_legacy_transform("viki_world_or_camera"))

    def test_output_trajectory_path_uses_hdf5(self) -> None:
        robot = normalize_robot("ur10")
        sample = Path("sample.npz")

        self.assertEqual(
            output_traj_path(Path("out"), sample, robot).name, "out_traj.h5"
        )
        self.assertEqual(
            output_traj_path(Path("out.npz"), sample, robot).name, "out.h5"
        )
        self.assertEqual(
            output_traj_path(Path("out.hdf5"), sample, robot).name, "out.hdf5"
        )

    def test_robot_base_sample_skips_legacy_transform(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.npz"
            body = np.zeros((5, 33, 3), dtype=np.float64)
            hand = np.zeros((5, 21, 3), dtype=np.float64)
            body[:, :, 0] = np.arange(5)[:, None]
            hand[:, :, 1] = np.arange(5)[:, None]
            np.savez(
                path,
                body=body,
                right_hand=hand,
                fps=10.0,
                coordinate_frame="robot_base",
            )

            with patch(
                "skeleton_tests.optimization.retarget_rgb_only.smooth_savgol",
                side_effect=lambda points, **_: np.asarray(
                    points, dtype=np.float64
                ).copy(),
            ):
                loaded_body, loaded_hand, fps = load_landmarks(
                    path, "right", 3, 1, None
                )

            self.assertEqual(fps, 10.0)
            np.testing.assert_allclose(loaded_body, body)
            np.testing.assert_allclose(loaded_hand, hand)

    def test_legacy_sample_applies_default_transform(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.npz"
            body = np.ones((5, 33, 3), dtype=np.float64)
            hand = np.ones((5, 21, 3), dtype=np.float64) * 2.0
            np.savez(
                path,
                body=body,
                right_hand=hand,
                fps=10.0,
                coordinate_frame="viki_world_or_camera",
            )

            with patch(
                "skeleton_tests.optimization.retarget_rgb_only.smooth_savgol",
                side_effect=lambda points, **_: np.asarray(
                    points, dtype=np.float64
                ).copy(),
            ):
                loaded_body, loaded_hand, _ = load_landmarks(path, "right", 3, 1, None)

            np.testing.assert_allclose(loaded_body, transform_points(body, R_DEFAULT))
            np.testing.assert_allclose(loaded_hand, transform_points(hand, R_DEFAULT))


if __name__ == "__main__":
    unittest.main()

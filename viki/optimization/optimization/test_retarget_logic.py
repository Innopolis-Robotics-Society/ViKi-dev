"""Retargeting tests that do not require PINK/Pinocchio."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from viki.optimization.optimization.retarget_rgb_only import (
    R_DEFAULT,
    align_rotations_to_initial,
    build_direct_rotation_targets,
    build_targets,
    build_parser,
    build_run_config,
    effective_orientation_cost,
    fill_invalid_rotations,
    load_landmarks,
    load_orientation_valid,
    load_retarget_input,
    load_smoothed_targets,
    normalize_robot,
    output_traj_path,
    should_apply_legacy_transform,
    transform_points,
)


class FakePin:
    class SE3:
        def __init__(self, rotation, translation):
            self.rotation = np.asarray(rotation, dtype=np.float64)
            self.translation = np.asarray(translation, dtype=np.float64)


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

    def test_parser_defaults_to_wrist_position(self) -> None:
        args = build_parser().parse_args(["--sample", "sample.npz", "--out", "out"])
        self.assertEqual(args.target_mode, "wrist_position")

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

    def test_align_initial_orientation_flag_reaches_config(self) -> None:
        args = build_parser().parse_args(
            [
                "--sample",
                "sample.npz",
                "--out",
                "out",
                "--target-mode",
                "hand_se3",
                "--align-initial-orientation",
            ]
        )
        cfg = build_run_config(args, normalize_robot("ur10"), 1.0, 0.5, 20, 0)

        self.assertTrue(cfg.align_initial_orientation)

    def test_coordinate_frame_controls_legacy_transform(self) -> None:
        self.assertFalse(should_apply_legacy_transform("robot_base"))
        self.assertTrue(should_apply_legacy_transform("viki_world_or_camera"))

    def test_output_trajectory_path_uses_hdf5(self) -> None:
        robot = normalize_robot("ur10")
        sample = Path("sample.npz")

        self.assertEqual(output_traj_path(Path("out"), sample, robot).name, "out_traj.h5")
        self.assertEqual(output_traj_path(Path("out.npz"), sample, robot).name, "out.h5")
        self.assertEqual(output_traj_path(Path("out.hdf5"), sample, robot).name, "out.hdf5")

    def test_hand_se3_targets_use_palm_frame_orientation(self) -> None:
        body = np.zeros((1, 33, 3), dtype=np.float64)
        body[0, 16] = [1.0, 2.0, 3.0]
        hand = np.zeros((1, 21, 3), dtype=np.float64)
        hand[0, 0] = [1.0, 2.0, 3.0]
        hand[0, 1] = [1.0, 3.0, 3.0]
        hand[0, 9] = [2.0, 2.0, 3.0]

        targets, valid = build_targets(FakePin, body, hand, "right")

        self.assertTrue(bool(valid[0]))
        np.testing.assert_allclose(targets[0].translation, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(targets[0].rotation[:, 0], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(targets[0].rotation[:, 1], [0.0, 1.0, 0.0])
        np.testing.assert_allclose(targets[0].rotation[:, 2], [0.0, 0.0, 1.0])

    def test_invalid_rotations_fill_from_nearest_valid_frame(self) -> None:
        valid_rotation = np.eye(3)
        filled, valid = fill_invalid_rotations([None, valid_rotation, None])

        np.testing.assert_allclose(filled[0], valid_rotation)
        np.testing.assert_allclose(filled[1], valid_rotation)
        np.testing.assert_allclose(filled[2], valid_rotation)
        np.testing.assert_array_equal(valid, [False, True, False])

    def test_hand_se3_targets_respect_orientation_valid_hint(self) -> None:
        body = np.zeros((2, 33, 3), dtype=np.float64)
        body[:, 16] = [1.0, 2.0, 3.0]
        hand = np.zeros((2, 21, 3), dtype=np.float64)
        hand[:, 0] = [1.0, 2.0, 3.0]
        hand[:, 1] = [1.0, 3.0, 3.0]
        hand[:, 9] = [2.0, 2.0, 3.0]

        targets, valid = build_targets(FakePin, body, hand, "right", np.array([True, False]))

        np.testing.assert_array_equal(valid, [True, False])
        np.testing.assert_allclose(targets[1].rotation, targets[0].rotation)

    def test_align_rotations_to_initial_maps_first_frame_to_target(self) -> None:
        angle = np.pi / 2.0
        first = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        second = np.eye(3, dtype=np.float64)
        initial_target = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )

        aligned = align_rotations_to_initial(np.stack([first, second]), initial_target)

        np.testing.assert_allclose(aligned[0], initial_target, atol=1e-12)
        np.testing.assert_allclose(aligned[1], second @ first.T @ initial_target, atol=1e-12)

    def test_load_orientation_valid_reads_optional_sample_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.npz"
            np.savez(path, orientation_valid=np.array([True, False, True]))

            mask = load_orientation_valid(path, limit_frames=2)

            np.testing.assert_array_equal(mask, [True, False])

    def test_smoothed_targets_load_positions_rotations_and_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cln-test.npz"
            positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
            rotations = np.stack([np.eye(3), np.full((3, 3), np.nan)]).astype(np.float32)
            valid = np.array([True, True])
            timestamps = np.array([1_000_000, 1_100_000], dtype=np.int64)
            np.savez(path, positions=positions, rotations=rotations, valid=valid, timestamps=timestamps)

            loaded = load_smoothed_targets(path, "right", limit_frames=None)

            self.assertEqual(loaded.source_format, "smoothed_targets")
            self.assertIsNone(loaded.hand)
            self.assertEqual(loaded.body.shape, (2, 33, 3))
            np.testing.assert_allclose(loaded.body[:, 16, :], positions)
            np.testing.assert_allclose(loaded.target_rotations, rotations)
            np.testing.assert_array_equal(loaded.orientation_valid, [True, False])
            np.testing.assert_array_equal(loaded.timestamps_us, timestamps)
            self.assertAlmostEqual(loaded.fps, 10.0)

    def test_smoothed_targets_skip_landmark_smoothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cln-test.npz"
            np.savez(
                path,
                positions=np.ones((2, 3), dtype=np.float64),
                rotations=np.tile(np.eye(3), (2, 1, 1)),
                valid=np.array([True, True]),
                timestamps=np.array([0, 100_000], dtype=np.int64),
            )

            with patch("viki.optimization.optimization.retarget_rgb_only.smooth_savgol") as smooth:
                loaded = load_retarget_input(path, "right", 99, 3, None)

            smooth.assert_not_called()
            self.assertEqual(loaded.source_format, "smoothed_targets")

    def test_smoothed_targets_interpolate_missing_positions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cln-gap.npz"
            np.savez(
                path,
                positions=np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [np.nan, np.nan, np.nan],
                        [2.0, 4.0, 6.0],
                    ],
                    dtype=np.float64,
                ),
                rotations=np.tile(np.eye(3), (3, 1, 1)),
                valid=np.array([True, True, True]),
                timestamps=np.array([0, 100_000, 200_000], dtype=np.int64),
            )

            loaded = load_smoothed_targets(path, "right", None)

            np.testing.assert_allclose(loaded.body[:, 16, :], [[0, 0, 0], [1, 2, 3], [2, 4, 6]])
            np.testing.assert_array_equal(loaded.orientation_valid, [True, True, True])

    def test_smoothed_targets_reject_malformed_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cln-bad.npz"
            np.savez(
                path,
                positions=np.ones((2, 3), dtype=np.float64),
                valid=np.array([True, True]),
                timestamps=np.array([0, 100_000], dtype=np.int64),
            )

            with self.assertRaises(KeyError):
                load_smoothed_targets(path, "right", None)

    def test_direct_rotation_targets_use_positions_and_valid_mask(self) -> None:
        body = np.zeros((2, 33, 3), dtype=np.float64)
        body[:, 16, :] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        rotations = np.stack([np.eye(3), np.diag([1.0, -1.0, -1.0])])

        targets, valid = build_direct_rotation_targets(
            FakePin,
            body,
            rotations,
            "right",
            np.array([True, False]),
        )

        np.testing.assert_array_equal(valid, [True, False])
        np.testing.assert_allclose(targets[0].translation, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(targets[1].translation, [4.0, 5.0, 6.0])
        np.testing.assert_allclose(targets[0].rotation, np.eye(3))
        np.testing.assert_allclose(targets[1].rotation, np.eye(3))

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
                "viki.optimization.optimization.retarget_rgb_only.smooth_savgol",
                side_effect=lambda points, **_: np.asarray(points, dtype=np.float64).copy(),
            ):
                loaded_body, loaded_hand, fps = load_landmarks(path, "right", 3, 1, None)

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
                "viki.optimization.optimization.retarget_rgb_only.smooth_savgol",
                side_effect=lambda points, **_: np.asarray(points, dtype=np.float64).copy(),
            ):
                loaded_body, loaded_hand, _ = load_landmarks(path, "right", 3, 1, None)

            np.testing.assert_allclose(loaded_body, transform_points(body, R_DEFAULT))
            np.testing.assert_allclose(loaded_hand, transform_points(hand, R_DEFAULT))

    def test_zero_landmark_sg_window_skips_smoothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.npz"
            body = np.ones((5, 33, 3), dtype=np.float64)
            hand = np.ones((5, 21, 3), dtype=np.float64) * 2.0
            np.savez(
                path,
                body=body,
                right_hand=hand,
                fps=10.0,
                coordinate_frame="robot_base",
            )

            with patch("viki.optimization.optimization.retarget_rgb_only.smooth_savgol") as smooth:
                loaded_body, loaded_hand, _ = load_landmarks(path, "right", 0, 1, None)

            smooth.assert_not_called()
            np.testing.assert_allclose(loaded_body, body)
            np.testing.assert_allclose(loaded_hand, hand)


if __name__ == "__main__":
    unittest.main()

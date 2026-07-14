"""Tests for optimisation API endpoints."""

from __future__ import annotations

import json
import queue
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from viki.server.routes import optimization


class OptimizationRoutesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.samples = self.root / "data" / "skeleton_smoothed"
        self.output = self.root / "data" / "robot_out"
        self.samples.mkdir(parents=True)
        self.output.mkdir(parents=True)
        np.savez(
            self.samples / "cln_api.npz",
            positions=np.ones((2, 3), dtype=np.float64),
            rotations=np.tile(np.eye(3), (2, 1, 1)),
            valid=np.array([True, True]),
            timestamps=np.array([1_000_000, 1_100_000], dtype=np.int64),
        )
        app = FastAPI()
        app.include_router(optimization.router)
        self.client = TestClient(app)
        self.patches = [
            patch.object(optimization, "PROJECT_ROOT", self.root),
            patch.object(optimization, "SMOOTHED_INPUT_DIR", self.samples),
            patch.object(optimization, "OUTPUT_DIR", self.output),
            patch.object(optimization, "_jobs", {}),
            patch.object(optimization, "_job_queue", queue.Queue()),
            patch.object(optimization, "_worker_started", False),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self) -> None:
        for item in reversed(self.patches):
            item.stop()
        self.tmp.cleanup()

    def test_samples_listing(self) -> None:
        samples = self.client.get("/api/optimization/samples")
        self.assertEqual(samples.status_code, 200)
        self.assertEqual(samples.json()["samples"][0]["filename"], "cln_api.npz")

    def test_retarget_rejects_missing_sample(self) -> None:
        response = self.client.post(
            "/api/optimization/retarget",
            json={"sample": "missing.npz", "output_name": "out"},
        )
        self.assertEqual(response.status_code, 404)

    def test_retarget_enqueues_job(self) -> None:
        (self.samples / "sample.npz").write_bytes(b"fake")

        fake_job = optimization.OptimizationJob(
            job_id="test-job",
            status="queued",
            description="test",
            output_stem="real_wrist_ur10",
        )
        with patch.object(
            optimization,
            "_enqueue_job",
            return_value=fake_job,
        ) as enqueue:
            response = self.client.post(
                "/api/optimization/retarget",
                json={
                    "sample": "sample.npz",
                    "robot": "ur10",
                    "output_name": "real_wrist_ur10",
                    "target_mode": "hand_se3",
                    "trajectory_scale": 1.5,
                    "trajectory_scale_origin": "robot_base",
                },
            )

        self.assertEqual(response.status_code, 200)
        args, _ = enqueue.call_args
        self.assertEqual(len(args), 6)
        cfg = args[4]
        self.assertEqual(cfg.trajectory_scale, 1.5)
        self.assertEqual(cfg.trajectory_scale_origin, "robot_base")

    def test_board_base_defaults_are_robot_specific(self) -> None:
        ur10 = optimization._retarget_defaults("ur10", "hand_se3")
        iiwa = optimization._retarget_defaults("iiwa14", "hand_se3")

        self.assertEqual(ur10.trajectory_scale, 0.75)
        self.assertEqual(iiwa.trajectory_scale, 0.55)
        self.assertEqual(ur10.ik_orientation_cost, 0.6)
        self.assertEqual(iiwa.ik_orientation_cost, 0.3)
        self.assertTrue(ur10.align_initial_orientation)
        self.assertFalse(iiwa.align_initial_orientation)

    def test_job_worker_calls_retarget(self) -> None:
        fake_summary = {"traj_path": "/fake/traj.h5", "frames": 5}
        from unittest.mock import MagicMock

        mock_cfg = MagicMock()
        with (
            patch.object(
                optimization, "retarget", return_value=fake_summary
            ) as mock_retarget,
            patch.object(optimization, "evaluate_saved_traj", return_value={}),
        ):
            job = optimization._enqueue_job(
                description="test",
                output_stem="test",
                sample_path=Path("/fake/sample.npz"),
                output_path=Path("/fake/traj.h5"),
                cfg=mock_cfg,
                do_evaluate=False,
            )
            deadline = time.time() + 2.0
            while (
                time.time() < deadline
                and optimization._jobs[job.job_id].status != "succeeded"
            ):
                time.sleep(0.01)

        self.assertEqual(job.status, "succeeded")
        self.assertEqual(job.exit_code, 0)
        mock_retarget.assert_called_once()

    def test_output_listing_and_download_are_sanitized(self) -> None:
        output = self.output / "result.h5"
        output.write_bytes(b"fake hdf5")
        listed = self.client.get("/api/optimization/outputs")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json()["outputs"][0]["filename"], "result.h5")

        downloaded = self.client.get("/api/optimization/outputs/result.h5")
        self.assertEqual(downloaded.status_code, 200)

        escaped = self.client.get("/api/optimization/outputs/..%2Fsecret.json")
        self.assertIn(escaped.status_code, {400, 404})


if __name__ == "__main__":
    unittest.main()

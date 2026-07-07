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
        self.legacy_samples = self.root / "data" / "optimization_samples"
        self.output = self.root / "data" / "robot_out"
        self.opt_dir = self.root / "viki" / "optimization" / "optimization"
        self.samples.mkdir(parents=True)
        self.legacy_samples.mkdir(parents=True)
        self.output.mkdir(parents=True)
        self.opt_dir.mkdir(parents=True)
        np.savez(
            self.samples / "cln_api.npz",
            positions=np.ones((2, 3), dtype=np.float64),
            rotations=np.tile(np.eye(3), (2, 1, 1)),
            valid=np.array([True, True]),
            timestamps=np.array([1_000_000, 1_100_000], dtype=np.int64),
        )
        self.recording = self.root / "rec_api_smoothed.json"
        landmarks = [[0.0, 0.0, 0.0] for _ in range(23)]
        landmarks[0] = [1.0, 2.0, 3.0]
        landmarks[1] = [1.0, 3.0, 3.0]
        landmarks[9] = [2.0, 2.0, 3.0]
        landmarks[21] = [100.0, 100.0, 100.0]
        landmarks[22] = [200.0, 200.0, 200.0]
        self.recording.write_text(
            json.dumps([{"ts": 1_000_000, "landmarks": landmarks}]),
            encoding="utf-8",
        )
        app = FastAPI()
        app.include_router(optimization.router)
        self.client = TestClient(app)
        self.patches = [
            patch.object(optimization, "PROJECT_ROOT", self.root),
            patch.object(optimization, "SMOOTHED_INPUT_DIR", self.samples),
            patch.object(optimization, "LEGACY_SAMPLES_DIR", self.legacy_samples),
            patch.object(optimization, "OUTPUT_DIR", self.output),
            patch.object(
                optimization,
                "RECORDING_DIRS",
                (self.root, self.root / "data" / "skeleton_recs"),
            ),
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

    def test_recording_listing_and_conversion(self) -> None:
        listed = self.client.get("/api/optimization/recordings")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(
            listed.json()["recordings"][0]["filename"], self.recording.name
        )

        converted = self.client.post(
            "/api/optimization/convert",
            json={
                "recording": self.recording.name,
                "output_name": "sample",
                "hand": "right",
                "include_arm": False,
            },
        )
        self.assertEqual(converted.status_code, 200)
        self.assertEqual(converted.json()["frames"], 1)
        self.assertNotIn("include_arm", converted.json())
        self.assertEqual(converted.json()["orientation_valid_frames"], 1)
        self.assertTrue((self.legacy_samples / "sample.npz").exists())

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

        with patch.object(optimization, "_enqueue_job") as enqueue:
            response = self.client.post(
                "/api/optimization/retarget",
                json={
                    "sample": "sample.npz",
                    "robot": "ur10",
                    "output_name": "real_wrist_ur10",
                    "target_mode": "hand_se3",
                },
            )

        self.assertEqual(response.status_code, 200)
        _, kwargs = enqueue.call_args
        self.assertIn("description", kwargs)
        self.assertIn("output_stem", kwargs)
        self.assertIn("sample_path", kwargs)
        self.assertIn("cfg", kwargs)

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
        self.assertEqual(escaped.status_code, 400)


if __name__ == "__main__":
    unittest.main()

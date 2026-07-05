"""Tests for optimisation API endpoints."""

from __future__ import annotations

import json
import queue
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from viki.server.routes import optimization


class OptimizationRoutesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.samples = self.root / "skeleton_tests" / "samples"
        self.output = self.root / "skeleton_tests" / "output"
        self.opt_dir = Path("skeleton_tests") / "optimization"
        self.samples.mkdir(parents=True)
        self.output.mkdir(parents=True)
        self.recording = self.root / "rec_api_smoothed.json"
        landmarks = [[0.0, 0.0, 0.0] for _ in range(23)]
        landmarks[0] = [1.0, 2.0, 3.0]
        self.recording.write_text(
            json.dumps([{"ts": 1_000_000, "landmarks": landmarks}]),
            encoding="utf-8",
        )
        app = FastAPI()
        app.include_router(optimization.router)
        self.client = TestClient(app)
        self.patches = [
            patch.object(optimization, "PROJECT_ROOT", self.root),
            patch.object(optimization, "SAMPLES_DIR", self.samples),
            patch.object(optimization, "OUTPUT_DIR", self.output),
            patch.object(optimization, "OPTIMIZATION_DIR", self.opt_dir),
            patch.object(optimization, "RECORDING_DIRS", (self.root, self.root / "data" / "skeleton_recs")),
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
        self.assertEqual(listed.json()["recordings"][0]["filename"], self.recording.name)

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
        self.assertTrue((self.samples / "sample.npz").exists())

        samples = self.client.get("/api/optimization/samples")
        self.assertEqual(samples.status_code, 200)
        self.assertEqual(samples.json()["samples"][0]["filename"], "sample.npz")

    def test_retarget_rejects_missing_sample_before_conda_check(self) -> None:
        response = self.client.post(
            "/api/optimization/retarget",
            json={"sample": "missing.npz", "output_name": "out"},
        )
        self.assertEqual(response.status_code, 404)

    def test_retarget_reports_unavailable_conda(self) -> None:
        (self.samples / "sample.npz").write_bytes(b"fake")
        with patch.object(optimization, "_resolve_conda_exe", side_effect=optimization.HTTPException(503, "no conda")):
            response = self.client.post(
                "/api/optimization/retarget",
                json={"sample": "sample.npz", "output_name": "out"},
            )
        self.assertEqual(response.status_code, 503)

    def test_retarget_enqueues_job_with_mocked_conda(self) -> None:
        (self.samples / "sample.npz").write_bytes(b"fake")
        fake_job = optimization.OptimizationJob(
            job_id="job1",
            status="queued",
            command=["conda", "run"],
            output_stem="real_wrist_ur10",
        )
        with (
            patch.object(optimization, "_resolve_conda_exe", return_value="conda"),
            patch.object(optimization, "_enqueue_job", return_value=fake_job) as enqueue,
        ):
            response = self.client.post(
                "/api/optimization/retarget",
                json={
                    "sample": "sample.npz",
                    "robot": "ur10",
                    "output_name": "real_wrist_ur10",
                    "target_mode": "wrist_position",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["job_id"], "job1")
        command = enqueue.call_args.args[0]
        self.assertIn("--target-mode", command)
        self.assertIn("wrist_position", command)

    def test_job_worker_records_subprocess_success(self) -> None:
        completed = subprocess.CompletedProcess(["conda"], 0, stdout="ok", stderr="")
        with patch.object(optimization.subprocess, "run", return_value=completed):
            job = optimization._enqueue_job(["conda", "run"], "worker_out")
            deadline = time.time() + 2.0
            while time.time() < deadline and optimization._jobs[job.job_id].status != "succeeded":
                time.sleep(0.01)

        response = self.client.get(f"/api/optimization/jobs/{job.job_id}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "succeeded")
        self.assertEqual(data["exit_code"], 0)
        self.assertEqual(data["stdout_tail"], "ok")

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

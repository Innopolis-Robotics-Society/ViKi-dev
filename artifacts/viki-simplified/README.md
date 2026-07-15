# ViKi — Simplified Skeleton Subsystem (UML source)

This directory contains a **simplified, dependency-free** mirror of the ViKi
skeleton pipeline, used only to generate a UML class diagram with `pyreverse`.
It preserves the main manager classes, workers, and the data flow — but omits
MediaPipe / OpenCV / numpy specifics.

## Main classes

| Class | Role |
|---|---|
| `CameraBackend` (ABC) | Device capture contract (RealSense / Kinect) |
| `Frame` | One color+depth sample |
| `CameraManager` | Owns backends + `_CameraWorker`s, serves latest frame |
| `_CameraWorker` | Daemon thread pulling frames into a ring buffer |
| `SyncedFrameGroup` | Frames from all cameras aligned to one host timestamp |
| `MultiCameraSync` | Builds `SyncedFrameGroup` from `CameraManager` |
| `CalibrationExtrinsics` | Per-camera pose → `transform_matrix` (camera→world) |
| `CalibrationManager` | Stores intrinsics/extrinsics; loads from disk |
| `PreparedFrame` | Color+depth (metres) ready for inference |
| `HandDetection` | Raw MediaPipe output (pixel space) |
| `Landmarks3D` | 3D landmarks in camera coordinates |
| `SkeletonFrame` | Fused world-frame skeleton + end-effector pose |
| `PipelineResult` | `fused_frame` + `detections` + `debug_depth_marks` |
| `SkeletonPipeline` | `SyncedFrameGroup` → `PipelineResult` |
| `SkeletonWorker` | Background thread running the pipeline, caching result |
| `AppState` | Wiring of all subsystems (FastAPI `app.state`) |

## Main data flow

```
CameraBackend.get_frame()
        │  Frame
        ▼
_CameraWorker (ring buffer)  ──latest()──►  CameraManager.latest_frame()
                                                    │
                                            MultiCameraSync.get_synced_frame()
                                                    │  SyncedFrameGroup
                                                    ▼
                                            SkeletonPipeline.process()
                                                    │
              ┌─────────────────────────────────────┼─────────────────────────────┐
              │ prepare  →  PreparedFrame            │                            │
              │ detect   →  HandDetection            │                            │
              │ lift     →  Landmarks3D (per camera) │                            │
              │ fuse     →  SkeletonFrame (world)    │                            │
              │ debug    →  debug_depth_marks        │                            │
              └─────────────────────────────────────┼─────────────────────────────┘
                                                    │  PipelineResult
                                                    ▼
                                          SkeletonWorker (cached)
                                                    │
                                            WebSocket /api/skeleton/stream
                                                    │  JSON (landmarks, detections,
                                                    │          debug_depth_marks)
                                                    ▼
                                              Frontend 3D panel
```

## Generate the UML

From this directory:

```bash
pyreverse -o png -p vikisim vikisim
# produces classes_vikisim.png and packages_vikisim.png
```

(Requires Graphviz `dot` for raster output; otherwise use `-o dot` to emit
`.dot` files.)

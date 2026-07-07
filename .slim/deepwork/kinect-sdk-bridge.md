# Kinect SDK Bridge Migration

## Goal
Replace the `ctypes`-based Kinect implementation in `viki/capture/kinect.py` with a native Pybind11 C++ extension to improve stability, performance, and alignment correctness.

## Current State
- `viki/capture/kinect.py` uses `viki_sdk` C++ extension.
- Depth alignment is handled by the official SDK via `KinectTransformation`.
- Build process is integrated into Docker.
- Memory safety and performance optimizations (intrinsics caching, 1D color buffers) implemented.

## Target Architecture
- **C++ Module (`viki_sdk`)**: A Pybind11 extension that wraps `k4a::device`, `k4a::capture`, and `k4a::transformation`.
- **Data Flow**: Zero-copy NumPy arrays for images.
- **Python Backend**: `KinectBackend` delegates all SDK calls to `viki_sdk`.

## Implementation Plan

### Phase 1: Build Infrastructure
- [x] Create `viki_sdk/` directory with C++ source and `CMakeLists.txt`.
- [x] Update `Dockerfile` to include build dependencies (`cmake`, `pybind11`, `libk4a-dev`).
- [x] Implement a multi-stage build to keep the runtime image slim.
- [x] Verify that the module can be compiled and imported in Python.

### Phase 2: Core SDK Bridge (Lifecycle & Capture)
- [x] Implement `KinectDevice` class in C++:
    - `open()`, `close()`, `start()`, `stop()`.
    - `get_serial()`.
- [x] Implement `KinectCapture` class in C++:
    - `get_color_image()` $\to$ NumPy array.
    - `get_depth_image()` $\to$ NumPy array.
    - `get_timestamp()`.
- [x] Integrate into `KinectBackend.start()` and `KinectBackend.get_frame()`.

### Phase 3: Alignment & Calibration
- [x] Implement `KinectTransformation` class in C++:
    - `align_depth_to_color()` $\to$ Aligned NumPy array.
    - `project_3d_to_2d()`, `transform_3d_to_3d()`.
- [x] Update `KinectBackend._transform_depth()` to use the new bridge.
- [x] Remove `viki/capture/aligner.py` and `data/kinect_calibration.json`.

### Phase 4: Validation & Cleanup
- [x] Verify frame rates and alignment quality.
- [x] Remove all `ctypes` definitions from `viki/capture/kinect.py`.
- [x] Final code review by `@oracle`.

## Status
- [x] Phase 1: Build Infrastructure
- [x] Phase 2: Core SDK Bridge
- [x] Phase 3: Alignment & Calibration
- [x] Phase 4: Validation & Cleanup

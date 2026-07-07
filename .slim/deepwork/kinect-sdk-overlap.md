# Kinect SDK Overlap Investigation

## Goal
Investigate if the project implements functionality that already exists in the Kinect K4A C++ SDK and identify where these redundancies occur.

## Current Understanding
The project uses `ctypes` to interface with `libk4a.so` directly in `viki/capture/kinect.py` instead of using a wrapper like `pyk4a`. This suggests a manual implementation of SDK calls.

## Research Context
- **Project Implementation**: 
    - Uses `ctypes` for device discovery, lifecycle, frame capture, and calibration.
    - Implements a custom `DepthAligner` (`viki/capture/aligner.py`) using NumPy for depth-to-color projection to avoid SDK "jitter".
    - Uses `MultiCameraSync` for host-side frame alignment.
- **K4A SDK Capabilities**: 
    - Provides high-level C++ classes (`k4a::device`, `k4a::capture`, `k4a::transformation`) that wrap the C API.
    - Includes a built-in `depth_image_to_color_camera` function for alignment.
    - Supports hardware-level synchronization (Master/Subordinate).

## Analysis & Verdict
- **C API Wrapper**: Justified. Avoids Docker build complexity.
- **Coordinate Transformations**: Uses SDK C API; no redundancy.
- **Depth Alignment**: 
    - **Redundant**: `DepthAligner.align()` is a manual re-implementation of the SDK's alignment function.
    - **Status**: Currently unused. `KinectBackend` still relies on the SDK's built-in alignment.
    - **Justification**: The intent is to eliminate jitter, but the implementation is incomplete/unused.
- **Multi-camera Sync**: Justified. Complementary to hardware sync; enables heterogeneous camera support.

## Plan
1. **Codebase Recon**: Map out all Kinect-related functionality in the project. [Completed]
2. **SDK Research**: Identify equivalent features in the K4A C++ SDK. [Completed]
3. **Gap Analysis**: Compare project implementation vs SDK capabilities to find redundancies. [Completed]
4. **Final Report**: Document findings and recommendations. [Completed]

## Status
- [x] Phase 1: Codebase Recon
- [x] Phase 2: SDK Research
- [x] Phase 3: Gap Analysis
- [x] Phase 4: Final Report

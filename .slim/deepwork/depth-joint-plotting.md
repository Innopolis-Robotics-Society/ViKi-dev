# Depth Joint Plotting Improvements

## Goal
Modify the depth base plotting module to:
1. Plot all 3 joints (Shoulder, Elbow, Wrist) every second on a single image (3x3 grid).
2. Ensure depth frames are synced with color frames (Temporal sync confirmed via SDK; Spatial sync via projection).
3. Use median instead of mean for depth estimation.
4. **Offload plotting to a background thread to prevent pipeline freezes.**

## Current Understanding
- **Module**: `viki/skeleton/viz.py` and `viki/skeleton/geometry.py`.
- **Estimation**: Currently uses a "robust median" (median of values within 10% of median).
- **Visualization**: Now plots 3 joints in a 3x3 grid every second.
- **Performance Issue**: Synchronous call to `plt.savefig` in `lift_to_3d` causes a ~5s freeze every second.
- **Sync**: Temporal sync is handled by `k4a_device_get_capture`. Spatial sync is handled by `project_color_to_depth`.

## Plan (Oracle Reviewed)
- [x] Discovery: Locate the plotting and estimation module.
- [x] Analysis: Understand how depth is estimated and how frames are synced.
- [x] Design: Draft a plan and get `@oracle` review.
- [x] Phase 1: Implementation
    - [x] Swap `np.mean` for `np.median` in `viki/skeleton/geometry.py`.
    - [x] Update `viki/skeleton/viz.py:visualize_depth_subtraction` to a 3x3 grid layout.
    - [x] Update `viki/skeleton/geometry.py:lift_to_3d` to collect data for `LM.ARM_CHAIN` and plot once per second.
- [x] Phase 2: Performance Optimization
    - [x] Implement asynchronous plotting using a `ThreadPoolExecutor` to prevent pipeline freezes.
- [x] Phase 3: Validation
    - [x] Verify the 3x3 plots are generated correctly.
    - [x] Confirm depth estimation stability with median.
    - [x] Verify temporal sync.
    - [x] Verify that plotting no longer blocks the estimation pipeline.
- [x] Final Review: Get `@oracle` review of the implementation.

## Implementation Phases
- **Phase 1**: Logic and Viz updates. (Completed)
- **Phase 2**: Async Plotting. (In Progress)
- **Phase 3**: Validation and Review.


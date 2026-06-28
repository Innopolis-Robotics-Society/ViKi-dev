import numpy as np
from viki.skeleton.geometry import lift_to_3d
from viki.skeleton.models import HandDetection, PreparedFrame, LM
from unittest.mock import MagicMock

def test_robust_mean():
    print("Testing robust mean...")
    # Mock data
    K = np.eye(3)
    depth_m = np.full((720, 1280), 1.0, dtype=np.float32)
    # Add outliers in the ROI (5x5 around 100, 100)
    depth_m[97:103, 97:103] = 1.0
    depth_m[100, 100] = 5.0 # Big outlier
    depth_m[101, 101] = 0.1 # Small outlier

    frame = PreparedFrame(
        rgb=np.zeros((720, 1280, 3), dtype=np.uint8),
        depth_m=depth_m,
        K=K,
        device_id="cam0",
        timestamp_us=0,
        base_depth_m=None
    )
    
    det = HandDetection(
        points={LM(i): np.array([100.0, 100.0]) for i in range(LM.N)},
        lm_z_rel=np.ones(LM.N, dtype=np.float32),
        confidence=1.0,
        device_id="cam0",
        timestamp_us=0
    )
    
    backend = MagicMock()
    
    import viki.config
    viki.config.SKELETON_ENABLE_DEPTH_VALIDATION = True
    viki.config.SKELETON_DEPTH_SAMP_RADIUS = 5

    res = lift_to_3d(det, frame, backend)
    z_val = res.points[LM(0)][2]
    print(f"Z value with outliers: {z_val:.3f} (Expected ~1.0)")
    assert abs(z_val - 1.0) < 0.1

def test_convergence_logic():
    print("\nTesting convergence logic...")
    K = np.eye(3)
    depth_m = np.full((720, 1280), 1.0, dtype=np.float32)
    frame = PreparedFrame(
        rgb=np.zeros((720, 1280, 3), dtype=np.uint8),
        depth_m=depth_m,
        K=K,
        device_id="cam0",
        timestamp_us=0,
        base_depth_m=None
    )
    backend = MagicMock()
    
    # Scenario 1: Close agreement (Z_proj=1.0, Z_est=1.05)
    # wrist_z_rel = 1.0, depth[100,100]=1.0 => mp_z_scale = 1.0
    # LM(0).z_rel = 1.05 => Z_est = 1.05
    z_rel_close = np.ones(LM.N, dtype=np.float32)
    z_rel_close[0] = 1.05 
    
    det_close = HandDetection(
        points={LM(i): np.array([100.0, 100.0]) for i in range(LM.N)},
        lm_z_rel=z_rel_close,
        confidence=1.0,
        device_id="cam0",
        timestamp_us=0
    )
    
    res_close = lift_to_3d(det_close, frame, backend)
    z_close = res_close.points[LM(0)][2]
    print(f"Close agreement Z: {z_close:.3f}")

    # Scenario 2: Large disagreement (Z_proj=1.0, Z_est=2.0)
    z_rel_far = np.ones(LM.N, dtype=np.float32)
    z_rel_far[0] = 2.0
    
    det_far = HandDetection(
        points={LM(i): np.array([100.0, 100.0]) for i in range(LM.N)},
        lm_z_rel=z_rel_far,
        confidence=1.0,
        device_id="cam0",
        timestamp_us=0
    )
    res_far = lift_to_3d(det_far, frame, backend)
    z_far = res_far.points[LM(0)][2]
    print(f"Large disagreement Z: {z_far:.3f} (Expected to favor Z_est ~2.0)")
    assert z_far > 1.5

if __name__ == "__main__":
    try:
        test_robust_mean()
        test_convergence_logic()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()

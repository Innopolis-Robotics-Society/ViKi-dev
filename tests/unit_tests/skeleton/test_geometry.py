import pytest
import numpy as np
from viki.skeleton.geometry import weighted_median, _pixel_to_3d

def test_weighted_median_basic():
    values = np.array([1.0, 2.0, 3.0])
    weights = np.array([1.0, 1.0, 1.0])
    assert weighted_median(values, weights) == 2.0

def test_weighted_median_skewed():
    values = np.array([1.0, 2.0, 3.0])
    weights = np.array([1.0, 10.0, 1.0])
    # Weight is concentrated at 2.0
    assert weighted_median(values, weights) == 2.0

def test_weighted_median_edge():
    values = np.array([1.0, 2.0, 3.0])
    weights = np.array([10.0, 1.0, 1.0])
    # Weight is concentrated at 1.0
    assert weighted_median(values, weights) == 1.0

def test_weighted_median_empty():
    assert np.isnan(weighted_median(np.array([]), np.array([])))

def test_weighted_median_zero_weights():
    values = np.array([1.0, 2.0, 3.0])
    weights = np.array([0.0, 0.0, 0.0])
    # Fallback to simple median
    assert weighted_median(values, weights) == 2.0

def test_pixel_to_3d():
    # Mock intrinsics
    fx, fy = 500.0, 500.0
    cx, cy = 320.0, 240.0
    Z = 1.0 # 1 meter
    u, v = 320.0, 240.0 # Center pixel
    
    res = _pixel_to_3d(u, v, Z, fx, fy, cx, cy)
    np.testing.assert_allclose(res, [0.0, 0.0, 1.0])
    
    # Test offset pixel
    u, v = 820.0, 740.0 # (820-320)*1/500 = 1.0
    res = _pixel_to_3d(u, v, Z, fx, fy, cx, cy)
    np.testing.assert_allclose(res, [1.0, 1.0, 1.0])

import json
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

class DepthAligner:
    """
    Performs a fixed-matrix projection of depth images into the color camera space.
    Avoids SDK internal estimation to eliminate jitter.
    """

    def __init__(
        self, 
        device_id: str, 
        depth_res: tuple[int, int], 
        color_res: tuple[int, int], 
        calibration_file: str = "data/kinect_calibration.json"
    ) -> None:
        self.device_id = device_id
        self.depth_w, self.depth_h = depth_res
        self.color_w, self.color_h = color_res

        with open(calibration_file, "r") as f:
            calibs = json.load(f)
        
        config = next((c for c in calibs if c["device_id"] == device_id), None)
        if config is None:
            raise ValueError(f"No calibration found for device {device_id} in {calibration_file}")

        # Intrinsics
        c_int = config["color_intrinsics"]
        d_int = config["depth_intrinsics"]
        
        self.fx_c, self.fy_c, self.cx_c, self.cy_c = c_int["fx"], c_int["fy"], c_int["cx"], c_int["cy"]
        self.fx_d, self.fy_d, self.cx_d, self.cy_d = d_int["fx"], d_int["fy"], d_int["cx"], d_int["cy"]

        # Extrinsics
        self.R = np.array(config["extrinsics"]["rotation"], dtype=np.float32)
        self.T = np.array(config["extrinsics"]["translation"], dtype=np.float32)

        # Precompute depth image coordinate grids
        u_d, v_d = np.meshgrid(np.arange(self.depth_w), np.arange(self.depth_h))
        self._u_d = u_d.astype(np.float32)
        self._v_d = v_d.astype(np.float32)

    def align(self, depth: np.ndarray) -> np.ndarray:
        """
        Project depth frame into color camera space.
        Input: depth (self.depth_h x self.depth_w) uint16 (mm)
        Output: aligned_depth (self.color_h x self.color_w) uint16 (mm)
        """
        z_d = depth.astype(np.float32)
        
        # 1. Unproject Depth (2D -> 3D in depth camera space)
        # X = (u - cx) * Z / fx
        x_d = (self._u_d - self.cx_d) * z_d / self.fx_d
        y_d = (self._v_d - self.cy_d) * z_d / self.fy_d
        
        # Stack into (3, H*W) for vectorized transformation
        # z_d is already the Z coordinate
        pts_d = np.stack([x_d.ravel(), y_d.ravel(), z_d.ravel()])

        # 2. Transform to Color Camera Space (3D -> 3D)
        # P_c = R * P_d + T
        pts_c = self.R @ pts_d + self.T.reshape(3, 1)
        
        x_c, y_c, z_c = pts_c[0], pts_c[1], pts_c[2]

        # 3. Project to Color Pixels (3D -> 2D)
        # u = (X * fx / Z) + cx
        # Avoid division by zero
        mask = z_c > 0
        u_c = np.full_like(x_c, -1.0)
        v_c = np.full_like(y_c, -1.0)
        
        u_c[mask] = (x_c[mask] * self.fx_c / z_c[mask]) + self.cx_c
        v_c[mask] = (y_c[mask] * self.fy_c / z_c[mask]) + self.cy_c

        # 4. Map to Aligned Depth Image
        aligned_depth = np.zeros((self.color_h, self.color_w), dtype=np.uint16)
        
        # Filter points that fall outside the color image bounds
        valid_mask = (
            mask & 
            (u_c >= 0) & (u_c < self.color_w) & 
            (v_c >= 0) & (v_c < self.color_h)
        )
        
        # Indices in the color image
        u_idx = np.round(u_c[valid_mask]).astype(np.int32)
        v_idx = np.round(v_c[valid_mask]).astype(np.int32)
        
        # Depth values to map
        z_vals = z_d.ravel()[valid_mask].astype(np.uint16)
        
        # Populate aligned image. If multiple depth pixels map to same color pixel, last one wins.
        aligned_depth[v_idx, u_idx] = z_vals

        return aligned_depth

    def project_color_to_depth(self, u_c: float, v_c: float, z_c: float) -> tuple[float, float] | None:
        """
        Inverse projection: Color Pixel + Depth -> Depth Pixel.
        """
        # 1. Color Pixel -> 3D Point in Color Space
        x_c = (u_c - self.cx_c) * z_c / self.fx_c
        y_c = (v_c - self.cy_c) * z_c / self.fy_c
        p_c = np.array([x_c, y_c, z_c], dtype=np.float32)

        # 2. Color Space -> Depth Space
        # P_d = R^T * (P_c - T)
        p_d = self.R.T @ (p_c - self.T)
        
        x_d, y_d, z_d = p_d
        
        if z_d <= 0:
            return None

        # 3. Depth 3D Point -> Depth Pixel
        u_d = (x_d * self.fx_d / z_d) + self.cx_d
        v_d = (y_d * self.fy_d / z_d) + self.cy_d
        
        if not (0 <= u_d < self.depth_w and 0 <= v_d < self.depth_h):
            return None
            
        return float(u_d), float(v_d)


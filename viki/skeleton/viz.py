"""
viki.skeleton.viz
-----------------
Visualisation helpers for debugging depth projection and skeleton processing.

These functions generate plots (saved to disk) for inspecting how colour
pixels map to depth and how background subtraction works.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime

import cv2
import numpy as np
import os
from typing import Optional

_DEBUG_DIR = "data/debug"

def visualize_depth_subtraction(
    base_depth: Optional[np.ndarray],
    current_depth: np.ndarray,
    landmark_data: list[dict],
):
    """
    Visualizes the depth subtraction process for multiple landmarks in a grid.

    Saves a plot to `data/debug/depth_sub_multi_{timestamp}.png`.

    Parameters
    ----------
    base_depth : Optional[np.ndarray]
        Background depth map (metres), or None.
    current_depth : np.ndarray
        Current depth map (metres).
    landmark_data : list[dict]
        List of dicts containing:
            - name : str (landmark name)
            - u, v : float (colour pixel coords)
            - ud, vd : float (depth pixel coords)
            - r : int (sample radius)
            - v_start, v_end, u_start, u_end : int (ROI bounds)
            - diff_roi : np.ndarray (subtracted ROI image)
            - z_proj : float (projected Z value)
            - median_pixel : tuple[int, int] or None (pixel of median depth)
    """
    num_lms = len(landmark_data)
    if num_lms == 0:
        return

    fig, axes = plt.subplots(num_lms, 3, figsize=(18, 6 * num_lms))
    
    # Handle case where num_lms == 1 (axes becomes 1D)
    if num_lms == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, data in enumerate(landmark_data):
        name = data["name"]
        u, v = data["u"], data["v"]
        ud, vd = data["ud"], data["vd"]
        r = data["r"]
        v_start, v_end = data["v_start"], data["v_end"]
        u_start, u_end = data["u_start"], data["u_end"]
        diff_roi = data["diff_roi"]
        z_proj = data["z_proj"]

        # 1. Base Depth
        ax1 = axes[row, 0]
        if base_depth is not None:
            ax1.imshow(np.nan_to_num(base_depth, nan=0.0), cmap='jet')
            ax1.set_title(f"Base Depth\n{name}")
        else:
            ax1.text(0.5, 0.5, "No Base Depth", ha='center')
            ax1.set_title(f"Base Depth (None)\n{name}")

        # 2. Current Depth + ROI Highlight
        ax2 = axes[row, 1]
        ax2.imshow(np.nan_to_num(current_depth, nan=0.0), cmap='jet')
        rect = patches.Rectangle((u_start - 0.5, v_start - 0.5), 
                                 u_end - u_start, v_end - v_start, 
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax2.add_patch(rect)
        ax2.plot(u, v, 'ro', markersize=2) # Original color projection
        ax2.set_title(f"Current Depth (ROI)\nZ_proj={z_proj:.3f}m")
        ax2.legend()


        # 3. Subtracted ROI (Zoomed)
        ax3 = axes[row, 2]
        ax3.imshow(np.nan_to_num(diff_roi, nan=0.0), cmap='magma')
        
        median_pixel = data.get("median_pixel")
        if median_pixel:
            # median_pixel is (v_rel, u_rel)
            v_rel, u_rel = median_pixel
            ax3.plot(u_rel, v_rel, 'ro', markersize=5, markeredgecolor='w', label='Median Pixel')
            ax3.legend()

        ax3.set_title(f"Diff ROI (Subtracted)\nShape: {diff_roi.shape}")


    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join(_DEBUG_DIR, f"depth_sub_multi_{timestamp}.png")
    os.makedirs(_DEBUG_DIR, exist_ok=True)
    plt.savefig(filename)
    plt.close()



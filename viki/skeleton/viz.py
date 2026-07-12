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
from typing import Tuple, Optional
from viki.capture.kinect import KinectBackend
from viki.skeleton.geometry import color_to_depth_pixel

def visualize_color_depth_mapping(
    color_img: np.ndarray, 
    depth_img: np.ndarray, 
    u: float, 
    v: float,
    K: np.ndarray,
    backend: KinectBackend
):
    """
    Visualize the mapping from a colour pixel to its corresponding depth area.

    Saves a plot to `data/debug/skeleton_mapping_{timestamp}.png`.

    Parameters
    ----------
    color_img : np.ndarray
        BGR or RGB image.
    depth_img : np.ndarray
        Depth map (H, W) in metres.
    u, v : float
        Colour pixel coordinates.
    K : np.ndarray
        Color intrinsic matrix (3x3).
    backend : KinectBackend
        Kinect backend for SDK projection.
    """
    # Prepare the plot first so we can save it even if reprojection fails
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Color Image
    axes[0].imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB) if len(color_img.shape) == 3 and color_img.shape[2] == 3 else color_img)
    axes[0].plot(u, v, 'ro', markersize=5, label='Sampled Pixel')
    axes[0].set_title("Color Image")
    axes[0].legend()
    
    # Depth Image
    depth_viz = np.nan_to_num(depth_img, nan=0.0)
    axes[1].imshow(depth_viz, cmap='jet')
    axes[1].set_title("Depth Image (Reprojection Failed)")

    # We need a Z value to project. Sample it from the depth map (roughly)
    z_est = 1.0 
    res = color_to_depth_pixel(u, v, z_est, K, backend, depth_img)
    
    if res is None:
        print("Reprojection failed")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        plt.tight_layout()
        plt.savefig(f"data/debug/skeleton_mapping_{timestamp}.png")
        plt.close()
        return
        
    ud, vd, _ = res
    ui, vi = int(round(ud)), int(round(vd))
    h, w = depth_img.shape[:2]
    
    # Ensure coordinates are within bounds
    if not (0 <= vi < h and 0 <= ui < w):
        print(f"Projected point ({ui}, {vi}) outside image boundaries ({w}x{h})")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        plt.tight_layout()
        plt.savefig(f"data/debug/skeleton_mapping_{timestamp}.png")
        plt.close()
        return
    
    # To make the visualization better, let's try to find a Z that actually 
    # maps to something visible in the depth map.
    for _ in range(3):
        v_start, v_end = max(0, vi - 1), min(h, vi + 2)
        u_start, u_end = max(0, ui - 1), min(w, ui + 2)
        window = depth_img[v_start:v_end, u_start:u_end]
        # Treat 0 as nan
        window_fixed = np.where(window == 0, np.nan, window)
        valid = window_fixed[~np.isnan(window_fixed)]
        if valid.size > 0:
             z_est = np.median(valid)
             res = color_to_depth_pixel(u, v, z_est, K, backend, depth_img)
             if res:
                 ud, vd, _ = res
                 ui, vi = int(round(ud)), int(round(vd))
        else:
            break
    
    # Update depth plot with the refined point
    v_start, v_end = max(0, vi - 1), min(h, vi + 2)
    u_start, u_end = max(0, ui - 1), min(w, ui + 2)
    rect = patches.Rectangle((u_start - 0.5, v_start - 0.5), 3, 3, 
                           linewidth=2, edgecolor='r', facecolor='none')
    axes[1].add_patch(rect)
    axes[1].plot(ui, vi, 'ro', markersize=3)
    axes[1].set_title(f"Depth Image (Sampled Area at Z={z_est:.2f}m)")

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    plt.savefig(f"data/debug/skeleton_mapping_{timestamp}.png")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Color Image
    axes[0].imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB) if len(color_img.shape) == 3 and color_img.shape[2] == 3 else color_img)
    axes[0].plot(u, v, 'ro', markersize=5, label='Sampled Pixel')
    axes[0].set_title("Color Image")
    axes[0].legend()
    
    # Depth Image
    depth_viz = np.nan_to_num(depth_img, nan=0.0)
    axes[1].imshow(depth_viz, cmap='jet')
    
    v_start, v_end = max(0, vi - 1), min(h, vi + 2)
    u_start, u_end = max(0, ui - 1), min(w, ui + 2)
    
    rect = patches.Rectangle((u_start - 0.5, v_start - 0.5), 3, 3, 
                           linewidth=2, edgecolor='r', facecolor='none')
    axes[1].add_patch(rect)
    axes[1].plot(ui, vi, 'ro', markersize=3)
    
    axes[1].set_title(f"Depth Image (Sampled Area at Z={z_est:.2f}m)")
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    plt.savefig(f"data/debug/skeleton_mapping_{timestamp}.png")
    plt.close()


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
    filename = f"data/debug/depth_sub_multi_{timestamp}.png"
    plt.savefig(filename)
    plt.close()



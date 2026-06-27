import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime

import cv2
import numpy as np
from typing import Tuple

def visualize_color_depth_mapping(
    color_img: np.ndarray, 
    depth_img: np.ndarray, 
    u: float, 
    v: float
):
    """
    Visualizes the mapping from a color pixel to its corresponding depth area.
    
    Args:
        color_img: BGR or RGB image
        depth_img: Depth map (H, W)
        u: X coordinate of the sampled pixel
        v: Y coordinate of the sampled pixel
    """
    ui, vi = int(round(u)), int(round(v))
    h, w = depth_img.shape[:2]
    
    # Ensure coordinates are within bounds
    if not (0 <= vi < h and 0 <= ui < w):
        print("Point outside image boundaries")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Color Image
    axes[0].imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB) if len(color_img.shape) == 3 and color_img.shape[2] == 3 else color_img)
    axes[0].plot(u, v, 'ro', markersize=5, label='Sampled Pixel')
    axes[0].set_title("Color Image")
    axes[0].legend()
    
    # Depth Image
    # Normalize depth for visualization
    depth_viz = np.nan_to_num(depth_img, nan=0.0)
    axes[1].imshow(depth_viz, cmap='jet')
    
    # Highlight 3x3 window
    v_start, v_end = max(0, vi - 1), min(h, vi + 2)
    u_start, u_end = max(0, ui - 1), min(w, ui + 2)
    
    # Draw a rectangle around the sampled area
    rect = patches.Rectangle((u_start - 0.5, v_start - 0.5), 3, 3, 
                       linewidth=2, edgecolor='r', facecolor='none')
    axes[1].add_patch(rect)
    axes[1].plot(ui, vi, 'ro', markersize=3)
    
    axes[1].set_title("Depth Image (Sampled Area)")
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    plt.savefig(f"data/debug/skeleton_mapping_{timestamp}.png")
    plt.close()


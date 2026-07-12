import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import json
import os

# CONFIGURATION
EXTRINSICS_FILE = "data/extrinsics_calibration.json"
SQUARE_SIZE_MULTIPLIER = 1.0  # Adjust this if distances look scaled wrong (e.g. 2.0 if 2.5cm -> 5cm)

def main():
    print(f"Loading extrinsics from {EXTRINSICS_FILE}...")
    
    if not os.path.exists(EXTRINSICS_FILE):
        print(f"Error: File {EXTRINSICS_FILE} not found.")
        return

    try:
        with open(EXTRINSICS_FILE, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return

    if not isinstance(data, list):
        print("Error: Expected a list of calibration entries.")
        return

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 1. World Origin (Board)
    ax.scatter(0, 0, 0, c='red', marker='s', s=100, label='World Origin (Board)')
    
    # Draw Board Plane (approx 20cm x 20cm)
    plane_range = 0.2 * SQUARE_SIZE_MULTIPLIER
    px = np.linspace(-plane_range, plane_range, 10)
    py = np.linspace(-plane_range, plane_range, 10)
    px, py = np.meshgrid(px, py)
    pz = np.zeros_like(px)
    ax.plot_surface(px, py, pz, alpha=0.2, color='gray')

    # 2. Reference point 1m up
    ax.scatter(0, 0, 1.0 * SQUARE_SIZE_MULTIPLIER, c='green', marker='^', s=100, label='1m above origin')

    # 3. Cameras
    for entry in data:
        dev_id = entry.get("device_id", "unknown")
        rvec = np.array(entry.get("rvec", [0,0,0]), dtype=np.float32).flatten()
        tvec = np.array(entry.get("tvec", [0,0,0]), dtype=np.float32).flatten()

        # Apply scaling multiplier to translation
        tvec = tvec * SQUARE_SIZE_MULTIPLIER

        # Camera position in world coordinates: C = -R^T * t
        R, _ = cv2.Rodrigues(rvec)
        cam_pos = -R.T @ tvec
        
        ax.scatter(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]), s=100, label=f'Camera {dev_id}')
        ax.text(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]), f' {dev_id}')

        # Gaze direction: Camera looks along its Z axis [0, 0, 1]
        # In world coordinates, this is R^T * [0, 0, 1]^T
        gaze_dir = R.T @ np.array([0, 0, 1], dtype=np.float32)
        line_len = 0.5 * SQUARE_SIZE_MULTIPLIER
        ax.quiver(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]), 
                  float(gaze_dir[0]), float(gaze_dir[1]), float(gaze_dir[2]), 
                  length=line_len, color='blue', arrow_length_ratio=0.1)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Extrinsics Check (Scale: {SQUARE_SIZE_MULTIPLIER}x)')
    ax.legend()
    
    # Equal scaling
    max_range = 2.0 * SQUARE_SIZE_MULTIPLIER
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    
    plt.grid(True)
    plt.savefig("scripts/extrinsics_viz.png")
    print("Plot saved to scripts/extrinsics_viz.png")

if __name__ == "__main__":
    main()


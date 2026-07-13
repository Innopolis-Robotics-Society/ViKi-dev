import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import json
import os
import argparse
from pathlib import Path

# Import retargeting config
try:
    from viki.config import (
        RETARGET_BASE_ROTATION, 
        RETARGET_BASE_TRANSLATION, 
        RETARGET_WRIST_SCALE, 
        RETARGET_RECENTER_TO_NEUTRAL,
        RETARGET_DEFAULT_ROBOT,
        MODELS_DIR
    )
except ImportError:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from viki.config import (
        RETARGET_BASE_ROTATION, 
        RETARGET_BASE_TRANSLATION, 
        RETARGET_WRIST_SCALE, 
        RETARGET_RECENTER_TO_NEUTRAL,
        RETARGET_DEFAULT_ROBOT,
        MODELS_DIR
    )


# Try to import robotics for neutral position
try:
    import pinocchio as pin
    from robot_descriptions.loaders.pinocchio import load_robot_description
    HAS_ROBOTICS = True
except ImportError:
    HAS_ROBOTICS = False

# CONFIGURATION
EXTRINSICS_FILE = "data/extrinsics_calibration.json"
SQUARE_SIZE_MULTIPLIER = 1.0 

def get_robot_world_pos(p_robot):
    """Transform robot frame point back to board world frame."""
    R = np.array(RETARGET_BASE_ROTATION, dtype=np.float64)
    T = np.array(RETARGET_BASE_TRANSLATION, dtype=np.float64)
    return R.T @ (p_robot - T)

def get_neutral_ee_pos(robot_alias):
    if not HAS_ROBOTICS:
        return np.array([0.4, 0.0, 0.4]) # Fallback guess
    
    # Map alias to local URDF search path relative to MODELS_DIR
    robot_urdf_search_map = {
        "ur10": "robot_descriptions/xacrodoc/ur10_official_description",
        "iiwa14": "robot_descriptions/drake/manipulation/models/iiwa_description/urdf/iiwa14_primitive_collision.urdf"
    }
    
    search_path = robot_urdf_search_map.get(robot_alias)
    if not search_path:
        print(f"Warning: No local URDF path mapping found for robot alias: {robot_alias}")
        return np.array([0.4, 0.0, 0.4])

    # Handle both direct file paths and directories (for hashed filenames)
    if os.path.isdir(os.path.join(MODELS_DIR, search_path)):
        full_search_dir = os.path.join(MODELS_DIR, search_path)
        urdf_files = [f for f in os.listdir(full_search_dir) if f.endswith(".urdf")]
        if not urdf_files:
            print(f"Warning: No URDF files found in {full_search_dir}. Falling back.")
            return np.array([0.4, 0.0, 0.4])
        urdf_path = os.path.join(full_search_dir, urdf_files[0])
    else:
        urdf_path = os.path.join(MODELS_DIR, search_path)

    if not os.path.exists(urdf_path):
        print(f"Warning: URDF file not found at {urdf_path}. Falling back to default EE position.")
        return np.array([0.4, 0.0, 0.4])

    try:
        model = pin.buildModelFromUrdf(urdf_path)
        data = pin.Data(model)
        ee_frame = "tool0" if "ur" in urdf_path else "iiwa_link_ee"
        frame_id = model.getFrameId(ee_frame)
        
        # Use a more compact home pose for UR robots to avoid extending to workspace edge
        if "ur" in urdf_path:
            q0 = np.array([0, -np.pi/2, 0, -np.pi/2, 0, 0])
        else:
            q0 = pin.neutral(model)
            
        pin.forwardKinematics(model, data, q0)
        pin.updateFramePlacements(model, data)
        return np.asarray(data.oMf[frame_id].translation, dtype=np.float64)
    except Exception as e:
        print(f"Warning: Could not load neutral EE for {robot_alias} from {urdf_path}: {e}")
        return np.array([0.4, 0.0, 0.4])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=str, help="Path to skeleton .npz recording")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to visualize")
    parser.add_argument("--start-frame", type=int, default=None, help="Start frame for trajectory plot")
    parser.add_argument("--end-frame", type=int, default=None, help="End frame for trajectory plot")
    parser.add_argument("--robot", type=str, default=RETARGET_DEFAULT_ROBOT, help="Robot alias")
    args = parser.parse_args()

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

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 1. World Origin (Board)
    ax.scatter(0, 0, 0, c='red', marker='s', s=100, label='World Origin (Board)')
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
        tvec = tvec * SQUARE_SIZE_MULTIPLIER
        R, _ = cv2.Rodrigues(rvec)
        cam_pos = -R.T @ tvec
        ax.scatter(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]), s=100, label=f'Camera {dev_id}')
        ax.text(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]), f' {dev_id}')
        gaze_dir = R.T @ np.array([0, 0, 1], dtype=np.float32)
        line_len = 0.5 * SQUARE_SIZE_MULTIPLIER
        ax.quiver(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]), 
                  float(gaze_dir[0]), float(gaze_dir[1]), float(gaze_dir[2]), 
                  length=line_len, color='blue', arrow_length_ratio=0.1)

    # 4. Robot and Landmarks
    if args.sample:
        sample_path = Path(args.sample)
        if sample_path.exists():
            with np.load(sample_path, allow_pickle=True) as data:
                # Determine frame range
                start_f = args.start_frame if args.start_frame is not None else args.frame
                end_f = args.end_frame if args.end_frame is not None else args.frame + 1
                
                # Support both legacy (body) and smoothed (positions) formats
                if "positions" in data.files:
                    # Smoothed target format: positions is (T, 3) wrist positions
                    wrist_positions = np.asarray(data["positions"], dtype=np.float64)
                    frame_count = len(wrist_positions)
                else:
                    # Legacy format: body has (T, 33, 3), wrist at index 16
                    points = data["points"] if "points" in data.files else data["body"]
                    wrist_idx = 16 if "right_hand" in data.files else 15
                    wrist_positions = np.asarray(points[:, wrist_idx, :], dtype=np.float64)
                    frame_count = len(points)
                
                # Clip range to available frames
                start_f = max(0, min(start_f, frame_count - 1))
                end_f = max(start_f + 1, min(end_f, frame_count))
                
                R_base = np.array(RETARGET_BASE_ROTATION, dtype=np.float64)
                T_base = np.array(RETARGET_BASE_TRANSLATION, dtype=np.float64)
                
                WRIST_0 = wrist_positions[0]
                
                # Visualization points in Board World
                robot_base_board = get_robot_world_pos(np.zeros(3))
                ax.scatter(robot_base_board[0], robot_base_board[1], robot_base_board[2], c='black', marker='s', s=200, label='Robot Base (Board)')
                
                # Neutral EE marker
                p_neutral_robot = get_neutral_ee_pos(args.robot)
                p_neutral_board = get_robot_world_pos(p_neutral_robot)
                ax.scatter(p_neutral_board[0], p_neutral_board[1], p_neutral_board[2], c='orange', marker='*', s=150, label='Robot Neutral EE')

                # Workspace Sphere (Approximate)
                reach = 1.3 if "ur10" in args.robot else 0.8
                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                xs = reach * np.cos(u) * np.sin(v) + robot_base_board[0]
                ys = reach * np.sin(u) * np.sin(v) + robot_base_board[1]
                zs = reach * np.cos(v) + robot_base_board[2]
                ax.plot_wireframe(xs, ys, zs, color='gray', alpha=0.1, linewidth=0.5)

                # Plot trajectory segment
                p_board_traj = wrist_positions[start_f:end_f]
                
                # Compute robot path for this segment
                p_robot_traj = []
                for p_board in p_board_traj:
                    p_robot_raw = R_base @ p_board + T_base
                    p_robot = p_robot_raw.copy()
                    if RETARGET_RECENTER_TO_NEUTRAL:
                        p_robot_0 = R_base @ WRIST_0 + T_base
                        p_robot = p_robot + (p_neutral_robot - p_robot_0)
                    
                    if RETARGET_WRIST_SCALE != 1.0:
                        p_robot_start = R_base @ WRIST_0 + T_base
                        if RETARGET_RECENTER_TO_NEUTRAL:
                            p_robot_start = p_neutral_robot
                        p_robot = p_robot_start + (p_robot - p_robot_start) * RETARGET_WRIST_SCALE
                    
                    p_robot_traj.append(get_robot_world_pos(p_robot))
                
                p_robot_traj = np.array(p_robot_traj)
                
                ax.plot(p_board_traj[:, 0], p_board_traj[:, 1], p_board_traj[:, 2], c='magenta', label='Human Wrist Path')
                ax.plot(p_robot_traj[:, 0], p_robot_traj[:, 1], p_robot_traj[:, 2], c='cyan', label='Robot EE Path')
                
                # Highlight current frame
                frame_idx = min(args.frame, len(wrist_positions) - 1)
                p_board = wrist_positions[frame_idx]
                # For current frame robot EE, we re-calculate scaled/recentered pos
                p_robot_curr = R_base @ p_board + T_base
                if RETARGET_RECENTER_TO_NEUTRAL:
                    p_robot_0 = R_base @ WRIST_0 + T_base
                    p_robot_curr = p_robot_curr + (p_neutral_robot - p_robot_0)
                if RETARGET_WRIST_SCALE != 1.0:
                    p_robot_start = R_base @ WRIST_0 + T_base
                    if RETARGET_RECENTER_TO_NEUTRAL:
                        p_robot_start = p_neutral_robot
                    p_robot_curr = p_robot_start + (p_robot_curr - p_robot_start) * RETARGET_WRIST_SCALE
                
                robot_ee_board = get_robot_world_pos(p_robot_curr)
                
                ax.scatter(p_board[0], p_board[1], p_board[2], c='magenta', marker='o', s=100, label='Human Wrist (Current)')
                ax.scatter(robot_ee_board[0], robot_ee_board[1], robot_ee_board[2], c='cyan', marker='X', s=100, label='Robot EE (Current)')
                ax.plot([robot_base_board[0], robot_ee_board[0]], 
                        [robot_base_board[1], robot_ee_board[1]], 
                        [robot_base_board[2], robot_ee_board[2]], 
                        color='black', linestyle='--', alpha=0.6)

        else:
            print(f"Error: Sample file {args.sample} not found.")

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Extrinsics & Retargeting Check (Scale: {SQUARE_SIZE_MULTIPLIER}x)')
    ax.legend()
    
    max_range = 2.0 * SQUARE_SIZE_MULTIPLIER
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    
    plt.grid(True)
    plt.savefig("scripts/extrinsics_viz.png")
    print("Plot saved to scripts/extrinsics_viz.png")

if __name__ == "__main__":
    main()

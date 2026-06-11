import cv2
import numpy as np
import time
import json
from viki.capture.manager import CameraManager

# --- Configuration ---
CHESSBOARD_SIZE = (9, 7)  # Internal corners
SQUARE_SIZE = 0.025       # Meters (adjust to your board)
MIN_SAMPLES = 20
MAX_SAMPLES = 50
RESULTS_FILE = "calibration_results.npz"

def calibrate():
    manager = CameraManager()
    devices_info = manager.list_devices()
    
    # Combine all available devices to start
    all_devices = devices_info.get("realsense", []) + devices_info.get("kinect", [])
    if not all_devices:
        print("No cameras found.")
        return

    print(f"Starting cameras: {all_devices}")
    for dev_id in all_devices:
        # Use 1280x720 for Kinects as they don't support 640x480
        res = (1280, 720) if "kinect" in dev_id else (640, 480)
        manager.start(dev_id, color_width=res[0], color_height=res[1])

    # Object points: (0,0,0), (1,0,0), ..., (8,5,0)
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    images_per_cam = {dev_id: [] for dev_id in all_devices}
    corners_per_cam = {dev_id: [] for dev_id in all_devices}

    print("\n--- Calibration Capture Mode ---")
    print("Press 'c' to capture a frame")
    print("Press 'q' to start calibration")
    print("Press 'esc' to abort")

    try:
        while True:
            frames = {}
            for dev_id in all_devices:
                frame = manager.latest_frame(dev_id)
                if frame is not None:
                    frames[dev_id] = frame.color
            
            if not frames:
                continue

            # Create a combined view for display
            combined = None
            for dev_id in all_devices:
                img = frames.get(dev_id)
                if img is None: continue
                
                # Resize for display if too large
                h, w = img.shape[:2]
                display_img = cv2.resize(img, (w // 2, h // 2))
                
                if combined is None:
                    combined = display_img
                else:
                    combined = np.hstack([combined, display_img])

            if combined is not None:
                cv2.imshow("Calibration Capture", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                # Capture frames from all cameras
                valid_capture = True
                current_capture = {}
                for dev_id in all_devices:
                    frame = manager.latest_frame(dev_id)
                    if frame is None:
                        valid_capture = False
                        break
                    current_capture[dev_id] = frame.color
                
                if valid_capture:
                    for dev_id in all_devices:
                        images_per_cam[dev_id].append(current_capture[dev_id])
                    print(f"Captured sample {len(images_per_cam[all_devices[0]])}/{MAX_SAMPLES}")
                else:
                    print("Failed to capture from all cameras.")

            elif key == ord('q'):
                break
            elif key == 27:
                print("Aborted.")
                manager.stop_all()
                cv2.destroyAllWindows()
                return

    finally:
        cv2.destroyAllWindows()

    if len(images_per_cam[all_devices[0]]) < MIN_SAMPLES:
        print(f"Not enough samples captured. Need at least {MIN_SAMPLES}.")
        manager.stop_all()
        return

    print("\nProcessing corners...")
    
    # Only keep samples where corners were found in ALL cameras
    valid_indices = []
    num_samples = len(images_per_cam[all_devices[0]])
    
    for i in range(num_samples):
        all_found = True
        sample_corners = {}
        for dev_id in all_devices:
            img = images_per_cam[dev_id][i]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
            if ret:
                sample_corners[dev_id] = corners
            else:
                all_found = False
                break
        
        if all_found:
            valid_indices.append(i)
            for dev_id in all_devices:
                corners_per_cam[dev_id].append(sample_corners[dev_id])

    print(f"Valid samples found: {len(valid_indices)} / {num_samples}")

    if len(valid_indices) < MIN_SAMPLES:
        print(f"Not enough valid samples. Need at least {MIN_SAMPLES}.")
        manager.stop_all()
        return

    # 1. Calibrate each camera individually
    intrinsics = {}
    dist_coeffs = {}
    
    for dev_id in all_devices:
        print(f"Calibrating {dev_id}...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objp, corners_per_cam[dev_id], (images_per_cam[dev_id][0].shape[1], images_per_cam[dev_id][0].shape[0]), None, None
        )
        if not ret:
            print(f"Calibration failed for {dev_id}")
            manager.stop_all()
            return
        intrinsics[dev_id] = mtx
        dist_coeffs[dev_id] = dist

    # 2. Stereo calibration for the first two cameras (if available)
    if len(all_devices) >= 2:
        cam0, cam1 = all_devices[0], all_devices[1]
        print(f"Performing stereo calibration between {cam0} and {cam1}...")
        
        # We need object points and image points for both cameras
        obj_points = [objp] * len(valid_indices)
        img_points0 = corners_per_cam[cam0]
        img_points1 = corners_per_cam[cam1]
        
        # Use previously found intrinsics as initial guess
        flags = cv2.CALIB_FIX_INTRINSIC
        
        ret, mtx0, dist0, mtx1, dist1, R, T, E, F = cv2.stereoCalibrate(
            obj_points, img_points0, img_points1,
            intrinsics[cam0], dist_coeffs[cam0],
            intrinsics[cam1], dist_coeffs[cam1],
            (images_per_cam[cam0][0].shape[1], images_per_cam[cam0][0].shape[0]),
            flags=flags
        )
        
        if ret:
            print("Stereo calibration successful!")
            # Store extrinsics
            extrinsic_data = {
                "pair": (cam0, cam1),
                "R": R,
                "T": T
            }
        else:
            print("Stereo calibration failed.")
            extrinsic_data = None
    else:
        extrinsic_data = None

    # Save results
    np.savez(RESULTS_FILE, 
             intrinsics=intrinsics, 
             dist_coeffs=dist_coeffs, 
             extrinsic_data=extrinsic_data)
    
    print(f"Results saved to {RESULTS_FILE}")
    manager.stop_all()

if __name__ == "__main__":
    calibrate()

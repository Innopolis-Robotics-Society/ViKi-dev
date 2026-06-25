import argparse
import time
import sys
from viki.capture.manager import CameraManager
from viki.capture.recorder import RGBDRecorder

def main():
    parser = argparse.ArgumentParser(description="Record synchronized RGB-D data from ViKi cameras.")
    parser.add_argument("--duration", type=float, default=10.0, help="Recording duration in seconds")
    parser.add_argument("--fps", type=int, default=15, help="Sync FPS for recording")
    parser.add_argument("--devices", nargs="+", help="List of device IDs to record (e.g., 'serial1 serial2'). If omitted, all detected are used.")
    parser.add_argument("--output", type=str, default="data/videos", help="Base directory for recordings")
    
    args = parser.parse_args()

    manager = CameraManager()
    
    # Device discovery
    devices_info = manager.list_devices()
    print("Detected devices:", devices_info)

    if args.devices:
        target_devices = args.devices
    else:
        # Use all detected RealSense and Kinect cameras by default
        target_devices = devices_info.get("realsense", []) + devices_info.get("kinect", [])
        if not target_devices:
            print("No cameras detected. Please specify --devices or check connections.")
            sys.exit(1)

    print(f"Starting cameras: {target_devices}")
    try:
        for dev_id in target_devices:
            manager.start(dev_id)
            print(f"Started {dev_id}")
        
        # Give cameras a moment to warm up
        time.sleep(2)

        recorder = RGBDRecorder(manager, output_base_dir=args.output)
        recorder.record(duration_s=args.duration, sync_fps=args.fps)

    except KeyboardInterrupt:
        print("\nRecording interrupted by user.")
    except Exception as e:
        print(f"Error during recording: {e}")
    finally:
        print("Stopping all cameras...")
        manager.stop_all()
        print("Done.")

if __name__ == "__main__":
    main()

// Server configuration panel: raw JSON editor + load/save/reset/restart and a
// collapsible help block. Ports toggleConfig/loadConfig/saveConfig/resetConfig/
// restartServer/toggleConfigHelp.

import { useStore } from "../../shared/store/store";
import { Panel } from "../../shared/ui/Panel";
import styles from "./config.module.css";

const HELP_ITEMS: Array<[string, string]> = [
  ["INTRINSICS_FILENAME", "Path to intrinsics calibration"],
  ["EXTRINSICS_FILENAME", "Path to extrinsics calibration"],
  ["DEFAULT_FPS", "Default frames per second for camera start"],
  ["DEFAULT_COLOR_WIDTH", "Default color image width"],
  ["DEFAULT_COLOR_HEIGHT", "Default color image height"],
  ["DEFAULT_DEPTH_MODE", "Default depth mode (NFOV_UNBINNED recommended)"],
  ["JPEG_QUALITY", "Quality of JPEG encoding (0-100)"],
  ["STREAM_IDLE_SLEEP", "Sleep duration for MJPEG stream generators (default value works well)"],
  ["PLACEHOLDER_SIZE", "Dimensions of the placeholder image when camera not started"],
  ["DEFAULT_SYNCHRONIZED_IMAGES_ONLY", "Only use synchronized frames (not tested with true)"],
  ["FRAME_BUFFER_SIZE", "Number of frames kept in ring buffer per camera"],
  ["RECORD_DEPTH", "Save depth frames as .npy files alongside the rgb-d videos (heavy on the disk)"],
  ["DEPTH_PROJECTION_DEBUG", "Enable debug plotting for depth projection (slows down live skeleton estimation)"],
  ["SKELETON_DEPTH_SAMP_RADIUS", "Radius for depth sampling during fusion (default worked well in testing)"],
  ["SKELETON_DEPTH_BASE_DIR", "Directory for background depth frames"],
  ["SKELETON_ENABLE_DEPTH_VALIDATION", "Toggle depth-based outlier rejection (disabling WILL lead to great instability)"],
  ["SKELETON_DEPTH_SUBTRACT_THRESHOLD", "Min depth diff for object detection (untested outside default value)"],
  ["HAND_TO_DETECT", "Hand side to estimate ('left' or 'right')"],
  ["CALIB_MODE", "Calibration mode ('manual' or 'auto') (auto is deprecated)"],
  ["CALIB_BOARD_TYPE", "Board type ('chess' or 'aruco')"],
  ["BONE_LENGTHS", "Manual bone length overrides {(Parent, Child): length_m} (default uses a moving mean, use this if you're sure you know what you're doing)"],
  ["BONE_TOLERANCE", "Tolerance for kinematic constraints (0.0-1.0) (default works well with mean for lenghts)"],
  ["CALIB_CHESS_BOARD_SIZE", "Default chessboard dimensions [cols, rows]"],
  ["CALIB_CHESS_SQUARE_SIZE", "Default chessboard square size in meters"],
  ["CALIB_ARUCO_BOARD_SIZE", "Default aruco board dimensions [cols, rows]"],
  ["CALIB_ARUCO_SQUARE_SIZE", "Default aruco square size in meters"],
  ["CALIB_ARUCO_MARKER_SIZE", "Default aruco marker size in meters"],
];

export function ConfigPanel() {
  const open = useStore((s) => s.configOpen);
  const helpOpen = useStore((s) => s.helpOpen);
  const configText = useStore((s) => s.configText);
  const setConfigText = useStore((s) => s.setConfigText);
  const toggleConfig = useStore((s) => s.toggleConfig);
  const toggleConfigHelp = useStore((s) => s.toggleConfigHelp);
  const loadConfig = useStore((s) => s.loadConfig);
  const saveConfig = useStore((s) => s.saveConfig);
  const resetConfig = useStore((s) => s.resetConfig);
  const restartServer = useStore((s) => s.restartServer);

  if (!open) return null;

  return (
    <Panel title="Server Configuration" onClose={toggleConfig}>
      <div className={styles.body}>
        <textarea
          className={styles.editor}
          spellCheck={false}
          value={configText}
          onChange={(e) => setConfigText(e.target.value)}
        />
        {helpOpen && (
          <div className={styles.help}>
            {HELP_ITEMS.map(([key, desc]) => (
              <div key={key}>
                <b>{key}</b>: {desc}
              </div>
            ))}
            <div>
              <b>CALIB_ARUCO_DICT</b>: OpenCV Aruco dictionary ID (our main board has 40
              markers total, each 5x5, so DICT_5X5_50 is good)
              <br />
              =============================================
              <br />
            </div>
            <div>Load: pulls the user configuration from backend</div>
            <div>Reset to Defaults: loads the default parameters for the system</div>
            <div>Save: save the current configuratiuon to the server</div>
            <div>Restart Server: restarts docker to apply all the configuration changes</div>
          </div>
        )}
        <div className={styles.footer}>
          <span className={styles.hint}>Restart is required for changes to take effect</span>
          <button onClick={toggleConfigHelp}>❔ Help</button>
          <button onClick={() => void loadConfig()}>Load</button>
          <button onClick={() => void resetConfig()}>Reset to Defaults</button>
          <button className="primary" onClick={() => void saveConfig()}>
            Save
          </button>
          <button className="danger" onClick={() => void restartServer()}>
            Restart Server
          </button>
        </div>
      </div>
    </Panel>
  );
}

// 3D skeleton estimation panel: status, per-camera detection state, view-mode
// toggle, camera selector, start/stop estimation + recording, the 3D canvas and
// the per-joint table. Ports toggleSkeleton/updateSkeletonUI/updateSkelStatus.

import { useStore } from "../../shared/store/store";
import { Panel } from "../../shared/ui/Panel";
import { Skeleton3DCanvas } from "./Skeleton3DCanvas";
import { SKEL_NAMES } from "./skeleton.constants";
import type { Point3D } from "./skeleton.types";
import styles from "./skeleton.module.css";

const VIEW_LABELS: Record<string, string> = {
  projections: "Projections",
  isometric: "Isometric",
  camera: "Camera",
};

function isValidPoint(p: Point3D | null): p is Point3D {
  return !!p && p[0] !== null && !isNaN(p[0]);
}

export function SkeletonPanel() {
  const open = useStore((s) => s.skelOpen);
  const estimating = useStore((s) => s.estimating);
  const recording = useStore((s) => s.recording);
  const viewMode = useStore((s) => s.viewMode);
  const vizCam = useStore((s) => s.vizCam);
  const devices = useStore((s) => s.devices);
  const detectionStatus = useStore((s) => s.detectionStatus);
  const landmarks = useStore((s) => s.landmarks);
  const toggleSkeleton = useStore((s) => s.toggleSkeleton);
  const cycleViewMode = useStore((s) => s.cycleViewMode);
  const setVizCam = useStore((s) => s.setVizCam);
  const toggleEstimation = useStore((s) => s.toggleEstimation);
  const toggleRecording = useStore((s) => s.toggleRecording);

  if (!open) return null;

  const rows = Object.entries(landmarks)
    .filter(([, p]) => isValidPoint(p))
    .map(([i, p]) => {
      const point = p as Point3D;
      return {
        index: i,
        name: SKEL_NAMES[Number(i)] ?? i,
        x: point[0].toFixed(3),
        y: point[1].toFixed(3),
        z: point[2].toFixed(3),
      };
    });

  return (
    <Panel title="3D Skeleton Estimation" onClose={toggleSkeleton}>
      <div className={styles.body}>
        <Skeleton3DCanvas />

        <div className={styles.controls}>
          <div>
            Status: {estimating ? "Running" : "Idle"}
            {recording ? " (RECORDING)" : ""}
          </div>

          <div className={styles.detections}>
            {devices.map((d) => {
              const detected = detectionStatus[d.id];
              return (
                <div key={d.id} className={detected ? styles.detected : styles.notDetected}>
                  {d.id}: {detected ? "Detected" : "Not Detected"}
                </div>
              );
            })}
          </div>

          <div className={styles.viewRow}>
            <button className={styles.viewBtn} onClick={cycleViewMode}>
              View: {VIEW_LABELS[viewMode]}
            </button>
          </div>

          <div className={styles.fieldCol}>
            <label htmlFor="skel-viz-cam">Visualize Camera</label>
            <select
              id="skel-viz-cam"
              value={vizCam ?? ""}
              onChange={(e) => setVizCam(e.target.value)}
            >
              {devices.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.id}
                </option>
              ))}
            </select>
          </div>

          <button
            className="primary"
            disabled={estimating}
            onClick={() => void toggleEstimation(true)}
          >
            ▶ Start Estimation
          </button>
          <button
            className="danger"
            disabled={!estimating}
            onClick={() => void toggleEstimation(false)}
          >
            ■ Stop Estimation
          </button>

          <button disabled={!estimating} onClick={() => void toggleRecording(true)}>
            🔴 Record
          </button>
          <button
            className="danger"
            disabled={!recording}
            onClick={() => void toggleRecording(false)}
          >
            ⏹ Stop Recording
          </button>

          <div className={styles.tableContainer}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>Joint</th>
                  <th>X</th>
                  <th>Y</th>
                  <th>Z</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.index}>
                    <td>{r.name}</td>
                    <td>{r.x}</td>
                    <td>{r.y}</td>
                    <td>{r.z}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </Panel>
  );
}

// Camera calibration panel: live per-device preview streams with sample counts
// plus the board params and calibration actions. Ports toggleCalibration,
// populateCalibrationStreams, updateCalibStatus and the session lifecycle.

import { useEffect } from "react";
import { useStore } from "../../shared/store/store";
import { usePolling } from "../../shared/hooks/usePolling";
import { Panel } from "../../shared/ui/Panel";
import { camerasApi } from "../cameras/cameras.api";
import { BoardParams } from "./BoardParams";
import { ExtrinsicsControls } from "./ExtrinsicsControls";
import { IntrinsicsCard } from "./IntrinsicsCard";
import styles from "./calibration.module.css";

export function CalibrationPanel() {
  const open = useStore((s) => s.calibOpen);
  const devices = useStore((s) => s.devices);
  const sampleCounts = useStore((s) => s.sampleCounts);
  const toggleCalibration = useStore((s) => s.toggleCalibration);
  const startSession = useStore((s) => s.startSession);
  const resetCalibration = useStore((s) => s.resetCalibration);
  const updateStatus = useStore((s) => s.updateStatus);
  const captureSample = useStore((s) => s.captureSample);
  const clearSamples = useStore((s) => s.clearSamples);

  // Start a (non-resetting) session on open; reset on close (ports toggleCalibration).
  useEffect(() => {
    if (!open) return;
    void startSession(false);
    return () => {
      void resetCalibration();
    };
  }, [open, startSession, resetCalibration]);

  // Poll per-device sample counts every second while open.
  usePolling(updateStatus, 1000, open);

  if (!open) return null;

  return (
    <Panel title="Camera Calibration" onClose={toggleCalibration}>
      <div className={styles.body}>
        <div className={styles.streams}>
          {devices.length === 0 ? (
            <div className={styles.empty}>No cameras detected</div>
          ) : (
            devices.map((d) => (
              <div key={d.id} className={styles.streamWrapper}>
                <div className={styles.imgContainer}>
                  <img src={camerasApi.colorStreamUrl(d.id, false)} alt={d.id} />
                  <span className={styles.streamLabel}>
                    {d.id}{" "}
                    <span className={styles.count}>
                      {sampleCounts[d.id] ?? "(0 samples)"}
                    </span>
                  </span>
                </div>
              </div>
            ))
          )}
        </div>

        <div className={styles.controls}>
          <BoardParams />
          <button className="primary" onClick={() => void captureSample()}>
            Capture Sample
          </button>
          <div className={styles.divider} />
          <IntrinsicsCard />
          <ExtrinsicsControls />
          <div className={styles.divider} />
          <button className="danger" onClick={() => void clearSamples()}>
            Clear Samples
          </button>
          <div className={styles.note}>
            Extrinsic calibration requires 1 sample per camera
          </div>
        </div>
      </div>
    </Panel>
  );
}

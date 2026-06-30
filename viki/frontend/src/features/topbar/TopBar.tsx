// Header + toolbar: server-alive dot, scan / start all / stop all, calibration
// and skeleton panel toggles, RGB-D record, and the config toggle. Ports the
// <header> and .toolbar markup plus startAll/stopAll/scanDevices and
// startRGBDRecording's button feedback.

import { useEffect, useRef, useState } from "react";
import { useStore } from "../../shared/store/store";
import styles from "./topbar.module.css";

export function TopBar() {
  const serverAlive = useStore((s) => s.serverAlive);
  const scanDevices = useStore((s) => s.scanDevices);
  const startAll = useStore((s) => s.startAll);
  const stopAll = useStore((s) => s.stopAll);
  const recordRgbd = useStore((s) => s.recordRgbd);
  const toggleConfig = useStore((s) => s.toggleConfig);
  const toggleCalibration = useStore((s) => s.toggleCalibration);
  const toggleSkeleton = useStore((s) => s.toggleSkeleton);

  const [recording, setRecording] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout>>();
  useEffect(() => () => clearTimeout(timer.current), []);

  const onRecord = async () => {
    const duration = await recordRgbd();
    if (duration == null) return;
    setRecording(true);
    timer.current = setTimeout(() => {
      setRecording(false);
      useStore.setState({ isRecordingRgbd: false });
      useStore.getState().log("Recording duration elapsed", "ok");
    }, duration * 1000);
  };

  return (
    <>
      <header className={styles.header}>
        <div className={`dot ${serverAlive ? "alive" : ""}`} />
        <h1 className={styles.title}>ViKi</h1>
        <span className={styles.subtitle}>capture server</span>
      </header>

      <div className={styles.toolbar}>
        <button className="primary" onClick={() => void scanDevices()}>
          ⟳ Scan devices
        </button>
        <button onClick={() => void startAll()}>▶ Start all</button>
        <button className="danger" onClick={() => void stopAll()}>
          ■ Stop all
        </button>
        <div className={styles.sep} />
        <button onClick={toggleCalibration}>⚙ Calibration</button>
        <button onClick={toggleSkeleton}>🦴 Skeleton</button>
        <div className={styles.sep} />
        <button
          className={recording ? "danger" : "primary"}
          disabled={recording}
          onClick={() => void onRecord()}
        >
          {recording ? "Recording..." : "🔴 Record RGB-D"}
        </button>
        <button className={styles.configBtn} onClick={toggleConfig}>
          ⚙ Config
        </button>
      </div>
    </>
  );
}

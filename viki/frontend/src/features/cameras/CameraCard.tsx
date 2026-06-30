// One camera: header, start/stop + resolution/fps/depth controls, colour and
// depth MJPEG streams with a skeleton overlay canvas, and live info footer.
// Ports buildCard/buildControls/getCardConfig/updateInfo/updateFpsForDepthMode/
// setRunning and the per-camera 2D skeleton overlay from startSkelStream.

import { useEffect, useRef } from "react";
import { useStore } from "../../shared/store/store";
import { usePolling } from "../../shared/hooks/usePolling";
import { useMjpegStream } from "./useMjpegStream";
import { camerasApi } from "./cameras.api";
import type { Device } from "./cameras.types";
import { drawOverlay } from "../skeleton/overlay";
import styles from "./cameras.module.css";

const OVERLAY_INDICES = [0, 21, 22];

export function CameraCard({ device }: { device: Device }) {
  const { id, type } = device;
  const running = useStore((s) => s.running[id] ?? false);
  const info = useStore((s) => s.info[id]);
  const cfg = useStore((s) => s.cardConfig[id]);
  const typeCfg = useStore((s) =>
    type === "kinect" ? s.cameraConfig.kinect : s.cameraConfig.realsense,
  );
  const setCardConfig = useStore((s) => s.setCardConfig);
  const startCamera = useStore((s) => s.startCamera);
  const stopCamera = useStore((s) => s.stopCamera);
  const fetchInfo = useStore((s) => s.fetchInfo);
  const detection = useStore((s) => s.detections[id]);

  const colorRef = useRef<HTMLImageElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);

  const colorSrc = useMjpegStream(running, () => camerasApi.colorStreamUrl(id));
  const depthSrc = useMjpegStream(running, () => camerasApi.depthStreamUrl(id));

  // Poll info every 2s while running (old attachStreams setInterval).
  usePolling(() => fetchInfo(id), 2000, running);

  // Draw / clear the 2D skeleton overlay when this camera's detection changes.
  useEffect(() => {
    const canvas = overlayRef.current;
    const img = colorRef.current;
    if (!canvas) return;
    if (!running) {
      canvas.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
    drawOverlay(canvas, img, detection, OVERLAY_INDICES);
  }, [detection, running]);

  if (!cfg) return null;

  const resValue = `${cfg.color_width}x${cfg.color_height}`;
  const maxFps = typeCfg.depthModeMaxFps?.[cfg.depth_mode] ?? Infinity;

  const colorAspect = info?.color_shape
    ? `${info.color_shape[1]}/${info.color_shape[0]}`
    : undefined;
  const depthAspect = info?.depth_shape
    ? `${info.depth_shape[1]}/${info.depth_shape[0]}`
    : undefined;

  const ci = info?.color_intrinsics;

  return (
    <div className={`${styles.card} ${running ? styles.running : ""}`}>
      <div className={styles.cardHeader}>
        <div className={`dot ${running ? "alive" : ""}`} />
        <span className={styles.name}>{id}</span>
        <span className={`${styles.tag} ${styles[type]}`}>{type}</span>
        {running && <span className={`${styles.tag} ${styles.liveTag}`}>LIVE</span>}
      </div>

      <div className={styles.cardControls}>
        <button disabled={running} onClick={() => void startCamera(id)}>
          ▶ Start
        </button>
        <button className="danger" disabled={!running} onClick={() => void stopCamera(id)}>
          ■ Stop
        </button>
        <div className={styles.sep} />
        <label>res</label>
        <select
          value={resValue}
          onChange={(e) => {
            const [w, h] = e.target.value.split("x").map(Number);
            setCardConfig(id, { color_width: w, color_height: h });
          }}
        >
          {typeCfg.resolutions.map((r) => (
            <option key={r} value={r}>
              {r.replace("x", " × ")}
            </option>
          ))}
        </select>
        <label>fps</label>
        <select
          value={cfg.fps}
          onChange={(e) => setCardConfig(id, { fps: Number(e.target.value) })}
        >
          {typeCfg.fps.map((f) => (
            <option key={f} value={f} disabled={f > maxFps}>
              {f} fps
            </option>
          ))}
        </select>
        {typeCfg.depthModes && (
          <>
            <div className={styles.sep} />
            <label>depth</label>
            <select
              value={cfg.depth_mode}
              onChange={(e) => {
                const depth = e.target.value;
                const cap = typeCfg.depthModeMaxFps?.[depth] ?? Infinity;
                setCardConfig(id, {
                  depth_mode: depth,
                  ...(cfg.fps > cap ? { fps: cap } : {}),
                });
              }}
            >
              {typeCfg.depthModes.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </>
        )}
      </div>

      <div className={styles.streams}>
        <div className={styles.streamPanel} style={{ aspectRatio: colorAspect }}>
          <img ref={colorRef} src={colorSrc} alt="color" />
          <canvas ref={overlayRef} className={styles.skelOverlay} />
          <span className={styles.streamLabel}>COLOR</span>
        </div>
        <div className={styles.streamDivider} />
        <div className={styles.streamPanel} style={{ aspectRatio: depthAspect }}>
          <img src={depthSrc} alt="depth" />
          <span className={styles.streamLabel}>DEPTH</span>
        </div>
      </div>

      <div className={styles.cardFooter}>
        {info?.color_shape ? (
          <>
            <span>color: {info.color_shape[1]}×{info.color_shape[0]}</span>
            {info.depth_shape && (
              <span>depth: {info.depth_shape[1]}×{info.depth_shape[0]}</span>
            )}
            {ci && (
              <span>
                fx={ci.fx.toFixed(1)} fy={ci.fy.toFixed(1)}
              </span>
            )}
          </>
        ) : (
          <span>—</span>
        )}
      </div>
    </div>
  );
}

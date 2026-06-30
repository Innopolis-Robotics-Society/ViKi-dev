// Imperative 3D skeleton view. Subscribes to the fused landmarks + view mode +
// selected camera and redraws the canvas via drawSkeleton3D. The EMA-smoothed
// center is carried in a ref (reset when the viz camera changes), mirroring the
// old module-level smoothedCenter.

import { useEffect, useRef } from "react";
import { useStore } from "../../shared/store/store";
import { drawSkeleton3D, type Vec3 } from "./skeleton3d";
import { SKEL_NAMES } from "./skeleton.constants";
import type { Point3D } from "./skeleton.types";
import styles from "./skeleton.module.css";

export function Skeleton3DCanvas() {
  const landmarks = useStore((s) => s.landmarks);
  const viewMode = useStore((s) => s.viewMode);
  const vizCam = useStore((s) => s.vizCam);
  const extrinsics = useStore((s) => s.extrinsics);

  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const smoothedCenter = useRef<Vec3 | null>(null);

  // Reset the smoothing when switching cameras (ports updateSkelVizCam).
  useEffect(() => {
    smoothedCenter.current = null;
  }, [vizCam]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const arr: Array<Point3D | null> = [];
    for (let i = 0; i < SKEL_NAMES.length; i++) arr[i] = landmarks[i] ?? null;

    drawSkeleton3D(canvas, container, arr, {
      viewMode,
      vizCam,
      extrinsics,
      smoothedCenter,
    });
  }, [landmarks, viewMode, vizCam, extrinsics]);

  return (
    <div ref={containerRef} className={styles.viz}>
      <canvas ref={canvasRef} className={styles.canvas} />
    </div>
  );
}

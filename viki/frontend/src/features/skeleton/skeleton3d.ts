// 3D skeleton rendering, ported verbatim from the old rodrigues() and
// drawSkeleton3D(). Pure canvas drawing — no React, no DOM lookups beyond the
// passed canvas/container — so it stays testable and reusable.

import type { Extrinsics } from "../calibration/calibration.types";
import type { Point3D } from "./skeleton.types";
import { SKEL_CONNS, SKEL_DRAW_INDICES } from "./skeleton.constants";
import { get2dContext } from "./overlay";

export type Vec3 = { x: number; y: number; z: number };

/** Rodrigues rotation vector -> 3x3 rotation matrix. */
export function rodrigues(rvec: number[]): number[][] {
  const theta = Math.sqrt(rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2]);
  if (theta < 1e-6) {
    return [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];
  }
  const ux = rvec[0] / theta;
  const uy = rvec[1] / theta;
  const uz = rvec[2] / theta;
  const cosT = Math.cos(theta);
  const sinT = Math.sin(theta);
  const k = 1 - cosT;
  return [
    [cosT + ux * ux * k, ux * uy * k - uz * sinT, ux * uz * k + uy * sinT],
    [uy * ux * k + uz * sinT, cosT + uy * uy * k, uy * uz * k - ux * sinT],
    [uz * ux * k - uy * sinT, uz * uy * k + ux * sinT, cosT + uz * uz * k],
  ];
}

export interface DrawOptions {
  viewMode: "projections" | "isometric" | "camera";
  vizCam: string | null;
  extrinsics: Record<string, Extrinsics>;
  /** Mutable EMA-smoothed center carried across frames. */
  smoothedCenter: { current: Vec3 | null };
}

export function drawSkeleton3D(
  canvas: HTMLCanvasElement,
  container: HTMLElement,
  landmarks: Array<Point3D | null>,
  opts: DrawOptions,
): void {
  const ctx = get2dContext(canvas);
  if (!ctx) return;

  canvas.width = container.clientWidth;
  canvas.height = container.clientHeight;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!landmarks || landmarks.length === 0) return;

  // Filter for physically plausible depth (> 10cm).
  const plausiblePoints = landmarks.filter(
    (p): p is Point3D => !!p && !isNaN(p[0]) && p[2] > 0.1,
  );
  if (plausiblePoints.length === 0) return;

  const rawCenter: Vec3 = {
    x: plausiblePoints.reduce((s, p) => s + p[0], 0) / plausiblePoints.length,
    y: plausiblePoints.reduce((s, p) => s + p[1], 0) / plausiblePoints.length,
    z: plausiblePoints.reduce((s, p) => s + p[2], 0) / plausiblePoints.length,
  };

  if (!opts.smoothedCenter.current) {
    opts.smoothedCenter.current = { ...rawCenter };
  } else {
    const alpha = 0.1;
    const sc = opts.smoothedCenter.current;
    sc.x += alpha * (rawCenter.x - sc.x);
    sc.y += alpha * (rawCenter.y - sc.y);
    sc.z += alpha * (rawCenter.z - sc.z);
  }

  // Use Shoulder (index 22) as the moving origin for the view.
  let projCenter: Vec3 = { x: 0, y: 0, z: 0 };
  const shoulder = landmarks[22];
  if (shoulder && !isNaN(shoulder[0])) {
    projCenter = { x: shoulder[0], y: shoulder[1], z: shoulder[2] };
  } else if (opts.smoothedCenter.current) {
    projCenter = { ...opts.smoothedCenter.current };
  }

  const filterCenter = opts.smoothedCenter.current;

  // Spatial filter: keep points within 1m of the center to reject background jumps.
  const isValid = (p: Point3D | null | undefined): p is Point3D => {
    if (!p || isNaN(p[0]) || p[2] <= 0.1) return false;
    const dist = Math.sqrt(
      Math.pow(p[0] - filterCenter.x, 2) +
        Math.pow(p[1] - filterCenter.y, 2) +
        Math.pow(p[2] - filterCenter.z, 2),
    );
    return dist < 1.0;
  };

  const validPoints = landmarks.filter(isValid);
  if (validPoints.length === 0) return;

  const scale = canvas.width / 5;

  if (opts.viewMode === "projections") {
    const viewW = canvas.width / 3;
    const viewH = canvas.height;

    const views = [
      {
        label: "TOP (X-Z)",
        proj: (p: Point3D) => ({ x: -(p[0] - projCenter.x), y: p[2] - projCenter.z }),
        offset: 0,
      },
      {
        label: "FRONT (X-Y)",
        proj: (p: Point3D) => ({ x: -(p[0] - projCenter.x), y: p[1] - projCenter.y }),
        offset: viewW,
      },
      {
        label: "SIDE (Z-Y)",
        proj: (p: Point3D) => ({ x: p[2] - projCenter.z, y: p[1] - projCenter.y }),
        offset: viewW * 2,
      },
    ];

    views.forEach((view) => {
      const centerX = view.offset + viewW / 2;
      const centerY = viewH / 2;

      ctx.fillStyle = "#6b6b80";
      ctx.font = "10px SF Mono";
      ctx.fillText(view.label, view.offset + 10, 20);

      ctx.strokeStyle = "#5b7fff";
      ctx.lineWidth = 2;
      ctx.fillStyle = "#fff";

      SKEL_CONNS.forEach(([a, b]) => {
        if (isValid(landmarks[a]) && isValid(landmarks[b])) {
          const p1 = view.proj(landmarks[a] as Point3D);
          const p2 = view.proj(landmarks[b] as Point3D);
          ctx.beginPath();
          ctx.moveTo(centerX + p1.x * scale, centerY - p1.y * scale);
          ctx.lineTo(centerX + p2.x * scale, centerY - p2.y * scale);
          ctx.stroke();
        }
      });

      landmarks.forEach((p, i) => {
        if (!SKEL_DRAW_INDICES.includes(i) || !isValid(p)) return;
        const proj = view.proj(p);
        ctx.beginPath();
        ctx.arc(centerX + proj.x * scale, centerY - proj.y * scale, 3, 0, Math.PI * 2);
        ctx.fill();
      });
    });
  } else if (opts.viewMode === "isometric") {
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;

    const projectIso = (p: Point3D) => {
      const dx = -(p[0] - projCenter.x);
      const dy = p[1] - projCenter.y;
      const dz = p[2] - projCenter.z;
      const x1 = dx * Math.cos(Math.PI / 4) + dz * Math.sin(Math.PI / 4);
      const z1 = -dx * Math.sin(Math.PI / 4) + dz * Math.cos(Math.PI / 4);
      const y2 = dy * Math.cos(Math.PI / 6) - z1 * Math.sin(Math.PI / 6);
      return { x: x1, y: y2 };
    };

    // World axes (X red, Y green, Z blue).
    const axisLen = 0.2;
    const projOrigin = projectIso([0, 0, 0]);
    const axes = [
      { p: [axisLen, 0, 0] as Point3D, color: "#ff5b5b", label: "X" },
      { p: [0, axisLen, 0] as Point3D, color: "#5bff5b", label: "Y" },
      { p: [0, 0, axisLen] as Point3D, color: "#5b7fff", label: "Z" },
    ];
    axes.forEach((axis) => {
      const pEnd = projectIso(axis.p);
      ctx.strokeStyle = axis.color;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(centerX + projOrigin.x * scale, centerY - projOrigin.y * scale);
      ctx.lineTo(centerX + pEnd.x * scale, centerY - pEnd.y * scale);
      ctx.stroke();
      ctx.fillStyle = axis.color;
      ctx.font = "10px SF Mono";
      ctx.fillText(axis.label, centerX + pEnd.x * scale + 2, centerY - pEnd.y * scale);
    });

    ctx.strokeStyle = "#5b7fff";
    ctx.lineWidth = 2;
    ctx.fillStyle = "#fff";

    SKEL_CONNS.forEach(([a, b]) => {
      if (isValid(landmarks[a]) && isValid(landmarks[b])) {
        const p1 = projectIso(landmarks[a] as Point3D);
        const p2 = projectIso(landmarks[b] as Point3D);
        ctx.beginPath();
        ctx.moveTo(centerX + p1.x * scale, centerY - p1.y * scale);
        ctx.lineTo(centerX + p2.x * scale, centerY - p2.y * scale);
        ctx.stroke();
      }
    });

    landmarks.forEach((p, i) => {
      if (!SKEL_DRAW_INDICES.includes(i) || !isValid(p)) return;
      const proj = projectIso(p);
      ctx.beginPath();
      ctx.arc(centerX + proj.x * scale, centerY - proj.y * scale, 3, 0, Math.PI * 2);
      ctx.fill();
    });
  } else if (opts.viewMode === "camera") {
    const extr = opts.vizCam ? opts.extrinsics[opts.vizCam] : undefined;
    if (!extr) {
      ctx.fillStyle = "#fff";
      ctx.textAlign = "center";
      ctx.fillText("No extrinsics calibrated for this camera", canvas.width / 2, canvas.height / 2);
      ctx.textAlign = "left";
      return;
    }

    const R = rodrigues(extr.rvec);
    const t = extr.tvec;
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;

    const projectCam = (p: Point3D) => {
      const xc = R[0][0] * p[0] + R[0][1] * p[1] + R[0][2] * p[2] + t[0];
      const yc = R[1][0] * p[0] + R[1][1] * p[1] + R[1][2] * p[2] + t[1];
      const zc = R[2][0] * p[0] + R[2][1] * p[1] + R[2][2] * p[2] + t[2];
      if (zc <= 0.1) return null; // behind camera
      return { x: xc / zc, y: yc / zc };
    };

    ctx.strokeStyle = "#5b7fff";
    ctx.lineWidth = 2;
    ctx.fillStyle = "#fff";

    SKEL_CONNS.forEach(([a, b]) => {
      if (isValid(landmarks[a]) && isValid(landmarks[b])) {
        const p1 = projectCam(landmarks[a] as Point3D);
        const p2 = projectCam(landmarks[b] as Point3D);
        if (p1 && p2) {
          ctx.beginPath();
          ctx.moveTo(centerX + p1.x * scale, centerY - p1.y * scale);
          ctx.lineTo(centerX + p2.x * scale, centerY - p2.y * scale);
          ctx.stroke();
        }
      }
    });

    landmarks.forEach((p, i) => {
      if (!SKEL_DRAW_INDICES.includes(i) || !isValid(p)) return;
      const proj = projectCam(p);
      if (proj) {
        ctx.beginPath();
        ctx.arc(centerX + proj.x * scale, centerY - proj.y * scale, 3, 0, Math.PI * 2);
        ctx.fill();
      }
    });
  }
}

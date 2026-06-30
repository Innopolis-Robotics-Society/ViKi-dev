// 2D skeleton overlay drawn on top of a camera's colour stream. Ports the
// per-device overlay block from the old startSkelStream onmessage handler.

import type { DetectionPoints } from "./skeleton.types";

// canvas.getContext throws in non-browser environments (jsdom without the canvas
// package); degrade to a no-op there instead of crashing the render.
export function get2dContext(canvas: HTMLCanvasElement): CanvasRenderingContext2D | null {
  try {
    return canvas.getContext("2d");
  } catch {
    return null;
  }
}

export function drawOverlay(
  canvas: HTMLCanvasElement,
  img: HTMLImageElement | null,
  detection: DetectionPoints | null | undefined,
  indices: number[],
): void {
  const ctx = get2dContext(canvas);
  if (!ctx) return;

  if (!detection || !img || !img.naturalWidth) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    return;
  }

  canvas.width = img.clientWidth;
  canvas.height = img.clientHeight;

  const scaleX = canvas.width / img.naturalWidth;
  const scaleY = canvas.height / img.naturalHeight;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#5b7fff";
  ctx.fillStyle = "#5b7fff";
  ctx.lineWidth = 3;

  const coords = indices.map((i) => {
    const p = detection[i];
    return p ? ([p[0] * scaleX, p[1] * scaleY] as [number, number]) : null;
  });

  ctx.beginPath();
  for (let i = 0; i < coords.length - 1; i++) {
    const a = coords[i];
    const b = coords[i + 1];
    if (a && b) {
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
    }
  }
  ctx.stroke();

  coords.forEach((p) => {
    if (p) {
      ctx.beginPath();
      ctx.arc(p[0], p[1], 4, 0, Math.PI * 2);
      ctx.fill();
    }
  });
}

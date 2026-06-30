// Shapes for the skeleton WebSocket stream and status, verified against
// viki/server/routes/skeleton.py (the stream sends { ts, landmarks, detections }).

export type Point2D = [number, number];
export type Point3D = [number, number, number];

/** Per-device 2D keypoints, keyed by landmark index. */
export type DetectionPoints = Record<string, Point2D | null>;

/** Fused 3D landmarks, keyed by landmark index. */
export type Landmarks = Record<string, Point3D | null>;

/** One message from /api/skeleton/stream. */
export interface SkeletonFrame {
  ts: number;
  landmarks: Landmarks;
  detections: Record<string, DetectionPoints | null>;
}

/** GET /api/skeleton/status */
export interface SkeletonStatus {
  enabled: boolean;
  recording: boolean;
}

export type SkelViewMode = "projections" | "isometric" | "camera";

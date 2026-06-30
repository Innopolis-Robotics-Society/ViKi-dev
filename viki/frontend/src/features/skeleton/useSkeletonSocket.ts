// Opens the /api/skeleton/stream WebSocket while estimation is enabled and feeds
// each message into the store. Ports startSkelStream/stopSkelStream. The URL is
// built from window.location so it works behind the dev proxy and in prod.
//
// Mounted once at the app root (not inside the panel): the per-camera overlays
// must keep updating even when the skeleton panel is closed, exactly as before.

import { useEffect } from "react";
import { useStore } from "../../shared/store/store";
import type { SkeletonFrame } from "./skeleton.types";

export function useSkeletonSocket(): void {
  const estimating = useStore((s) => s.estimating);
  const onStreamFrame = useStore((s) => s.onStreamFrame);
  const log = useStore((s) => s.log);

  useEffect(() => {
    if (!estimating) return;

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${window.location.host}/api/skeleton/stream`);

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as SkeletonFrame;
        onStreamFrame(data);
      } catch {
        /* ignore malformed frame */
      }
    };
    ws.onclose = () => log("Skeleton stream closed");

    return () => ws.close();
  }, [estimating, onStreamFrame, log]);
}

// Manages an MJPEG <img> source declaratively. When the camera starts we wait a
// short delay (so the worker has produced its first frame, as the old
// attachStreams did) then expose the stream URL; when it stops we clear it.
// The returned string is bound directly to <img src> — the browser pulls the
// stream itself, never through React state/fetch.

import { useEffect, useState } from "react";

export function useMjpegStream(
  running: boolean,
  makeUrl: () => string,
  delayMs = 1500,
): string {
  const [src, setSrc] = useState("");

  useEffect(() => {
    if (!running) {
      setSrc("");
      return;
    }
    const id = setTimeout(() => setSrc(makeUrl()), delayMs);
    return () => clearTimeout(id);
    // makeUrl is a fresh closure each render; intentionally keyed on `running`.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running, delayMs]);

  return src;
}

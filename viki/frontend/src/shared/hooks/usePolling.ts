import { useEffect, useRef } from "react";

// Run `fn` immediately and then every `intervalMs` while `enabled`.
// Used to poll backend status endpoints without blocking renders.
export function usePolling(fn: () => void, intervalMs: number, enabled = true): void {
  const saved = useRef(fn);
  saved.current = fn;

  useEffect(() => {
    if (!enabled) return;
    saved.current();
    const id = setInterval(() => saved.current(), intervalMs);
    return () => clearInterval(id);
  }, [intervalMs, enabled]);
}

// Cross-feature activity log (ports the old `log()` helper). Any slice or
// component appends entries; the LogConsole renders them. Kept in the shared
// store because every feature writes to the same console.

import type { StateCreator } from "zustand";
import type { RootState } from "./store";

export type LogLevel = "" | "ok" | "error";

export interface LogEntry {
  id: number;
  time: string;
  message: string;
  level: LogLevel;
}

export interface LogSlice {
  logs: LogEntry[];
  log: (message: string, level?: LogLevel) => void;
}

const MAX_ENTRIES = 40;
let nextId = 0;

export const createLogSlice: StateCreator<RootState, [], [], LogSlice> = (set) => ({
  logs: [],
  log: (message, level = "") => {
    const entry: LogEntry = {
      id: nextId++,
      time: new Date().toLocaleTimeString(),
      message,
      level,
    };
    // Newest first, capped — mirrors el.prepend + trim in the old file.
    set((s) => ({ logs: [entry, ...s.logs].slice(0, MAX_ENTRIES) }));
  },
});

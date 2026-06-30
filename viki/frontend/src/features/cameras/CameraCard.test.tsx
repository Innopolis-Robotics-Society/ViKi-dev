import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { CameraCard } from "./CameraCard";
import { useStore } from "../../shared/store/store";
import type { Device } from "./cameras.types";

const device: Device = { id: "cam0", type: "realsense" };

function seedStore(running: boolean) {
  useStore.setState({
    running: { cam0: running },
    info: {},
    detections: {},
    cardConfig: {
      cam0: { color_width: 640, color_height: 480, fps: 30, depth_mode: "NFOV_UNBINNED" },
    },
  });
}

beforeEach(() => {
  // fetchInfo polling fires immediately when running; keep it offline.
  vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("CameraCard", () => {
  it("idle: Start enabled, Stop disabled, no LIVE tag", () => {
    seedStore(false);
    render(<CameraCard device={device} />);

    expect(screen.getByText("cam0")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Start/ })).not.toBeDisabled();
    expect(screen.getByRole("button", { name: /Stop/ })).toBeDisabled();
    expect(screen.queryByText("LIVE")).not.toBeInTheDocument();
  });

  it("running: Start disabled, Stop enabled, LIVE shown", () => {
    seedStore(true);
    render(<CameraCard device={device} />);

    expect(screen.getByRole("button", { name: /Start/ })).toBeDisabled();
    expect(screen.getByRole("button", { name: /Stop/ })).not.toBeDisabled();
    expect(screen.getByText("LIVE")).toBeInTheDocument();
  });

  it("renders the resolution/fps controls from cardConfig", () => {
    seedStore(false);
    render(<CameraCard device={device} />);
    // RealSense has no depth control.
    expect(screen.queryByText("depth")).not.toBeInTheDocument();
    expect(screen.getByText("res")).toBeInTheDocument();
    expect(screen.getByText("fps")).toBeInTheDocument();
  });
});

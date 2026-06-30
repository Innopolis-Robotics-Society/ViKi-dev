import { beforeEach, describe, expect, it } from "vitest";
import { useStore } from "./store";

// Exercises real slice action logic against the composed store (no network).
describe("store slices", () => {
  beforeEach(() => {
    useStore.setState({
      logs: [],
      viewMode: "projections",
      detections: {},
      detectionStatus: {},
      landmarks: {},
      devices: [{ id: "cam0", type: "realsense" }],
    });
  });

  it("log() prepends newest-first and caps at 40 entries", () => {
    const { log } = useStore.getState();
    for (let i = 0; i < 45; i++) log(`msg ${i}`);
    const { logs } = useStore.getState();
    expect(logs).toHaveLength(40);
    expect(logs[0].message).toBe("msg 44");
    expect(logs[0].level).toBe("");
  });

  it("cycleViewMode cycles projections -> isometric -> camera -> projections", () => {
    const { cycleViewMode } = useStore.getState();
    cycleViewMode();
    expect(useStore.getState().viewMode).toBe("isometric");
    cycleViewMode();
    expect(useStore.getState().viewMode).toBe("camera");
    cycleViewMode();
    expect(useStore.getState().viewMode).toBe("projections");
  });

  it("onStreamFrame stores detections, derives status, and keeps last pose when empty", () => {
    const { onStreamFrame } = useStore.getState();
    onStreamFrame({
      ts: 1,
      detections: { cam0: { "0": [10, 20] } },
      landmarks: { "0": [0.1, 0.2, 0.3] },
    });
    expect(useStore.getState().detectionStatus.cam0).toBe(true);
    expect(useStore.getState().landmarks["0"]).toEqual([0.1, 0.2, 0.3]);

    // A frame with no fused landmarks must not wipe the last pose.
    onStreamFrame({ ts: 2, detections: { cam0: null }, landmarks: {} });
    expect(useStore.getState().detectionStatus.cam0).toBe(false);
    expect(useStore.getState().landmarks["0"]).toEqual([0.1, 0.2, 0.3]);
  });
});

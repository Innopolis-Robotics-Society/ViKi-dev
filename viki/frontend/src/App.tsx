import { useEffect } from "react";
import { TopBar } from "./features/topbar/TopBar";
import { CameraGrid } from "./features/cameras/CameraGrid";
import { ConfigPanel } from "./features/config/ConfigPanel";
import { CalibrationPanel } from "./features/calibration/CalibrationPanel";
import { SkeletonPanel } from "./features/skeleton/SkeletonPanel";
import { LogConsole } from "./shared/ui/LogConsole";
import { usePolling } from "./shared/hooks/usePolling";
import { useSkeletonSocket } from "./features/skeleton/useSkeletonSocket";
import { useStore } from "./shared/store/store";

// Page shell. Sections are panels on one page (no router), mirroring the
// original single-page DOM order (toolbar → panels → camera grid → log).
export function App() {
  const initConfig = useStore((s) => s.initConfig);
  const scanDevices = useStore((s) => s.scanDevices);
  const checkServer = useStore((s) => s.checkServer);
  const initBoardFields = useStore((s) => s.initBoardFields);

  // Startup sequence ported from init(): load config, derive calibration board
  // defaults, then scan devices.
  useEffect(() => {
    void (async () => {
      await initConfig();
      initBoardFields();
      await scanDevices();
    })();
  }, [initConfig, scanDevices, initBoardFields]);

  // Server-alive heartbeat every 5s.
  usePolling(checkServer, 5000);

  // Live skeleton stream feeds per-camera overlays + the 3D view while enabled,
  // independent of the skeleton panel being open.
  useSkeletonSocket();

  return (
    <div>
      <TopBar />
      <ConfigPanel />
      <CalibrationPanel />
      <SkeletonPanel />
      <CameraGrid />
      <LogConsole />
    </div>
  );
}

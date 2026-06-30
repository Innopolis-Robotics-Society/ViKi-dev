// Extrinsics calibration trigger. Ports the "Calibrate Extrinsics" button.

import { useStore } from "../../shared/store/store";

export function ExtrinsicsControls() {
  const extrinsicsCalibration = useStore((s) => s.extrinsicsCalibration);
  return (
    <button className="primary" onClick={() => void extrinsicsCalibration()}>
      Calibrate Extrinsics
    </button>
  );
}

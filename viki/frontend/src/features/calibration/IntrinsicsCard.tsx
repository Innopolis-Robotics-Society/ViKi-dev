// Intrinsics calibration trigger. Ports intrinsicsCalibration(), which existed
// in the old script but had no button wired to it; surfaced here per the spec's
// component breakdown.

import { useStore } from "../../shared/store/store";

export function IntrinsicsCard() {
  const intrinsicsCalibration = useStore((s) => s.intrinsicsCalibration);
  return (
    <button onClick={() => void intrinsicsCalibration()}>Calibrate Intrinsics</button>
  );
}

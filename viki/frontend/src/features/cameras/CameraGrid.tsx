// Device grid: one CameraCard per discovered device, or an empty state.
// Ports renderCards + the empty-state markup.

import { useStore } from "../../shared/store/store";
import { CameraCard } from "./CameraCard";
import styles from "./cameras.module.css";

export function CameraGrid() {
  const devices = useStore((s) => s.devices);

  return (
    <section className={styles.grid} aria-label="cameras">
      {devices.length === 0 ? (
        <div className={styles.emptyState}>
          <h2>No cameras detected</h2>
          <p>Make sure cameras are connected and container has USB access.</p>
        </div>
      ) : (
        devices.map((d) => <CameraCard key={d.id} device={d} />)
      )}
    </section>
  );
}

// Collapsible section shell shared by Config / Calibration / Skeleton panels.
// In the old file these were three near-identical blocks (.config-panel,
// .calib-panel, .skeleton-panel) toggled via display:none. Here the parent
// conditionally renders instead. Body layout is supplied by each feature.

import type { ReactNode } from "react";
import styles from "./Panel.module.css";

interface PanelProps {
  title: string;
  onClose: () => void;
  children: ReactNode;
}

export function Panel({ title, onClose, children }: PanelProps) {
  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <span className={styles.title}>{title}</span>
        <button className="danger" onClick={onClose}>
          ✕ Close
        </button>
      </div>
      {children}
    </div>
  );
}

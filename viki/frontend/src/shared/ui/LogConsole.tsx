// Activity console. Renders the shared log slice. Pure presentation:
// reads `logs` from the store, no DOM mutation (unlike the old `log()`).

import { useStore } from "../store/store";
import styles from "./LogConsole.module.css";

export function LogConsole() {
  const logs = useStore((s) => s.logs);
  return (
    <div className={styles.log}>
      {logs.map((entry) => (
        <div key={entry.id} className={`${styles.entry} ${entry.level ? styles[entry.level] : ""}`}>
          [{entry.time}] {entry.message}
        </div>
      ))}
    </div>
  );
}

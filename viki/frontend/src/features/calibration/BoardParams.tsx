// Board parameter inputs (chessboard / ChArUco). Ports the .board-params block
// and toggleBoardFields/syncBoardParameters wiring.

import { useStore } from "../../shared/store/store";
import { ARUCO_DICTS } from "./calibration.types";

export function BoardParams() {
  const boardType = useStore((s) => s.boardType);
  const boardWidth = useStore((s) => s.boardWidth);
  const boardHeight = useStore((s) => s.boardHeight);
  const squareSize = useStore((s) => s.squareSize);
  const markerSize = useStore((s) => s.markerSize);
  const arucoDict = useStore((s) => s.arucoDict);
  const setBoardType = useStore((s) => s.setBoardType);
  const setBoardField = useStore((s) => s.setBoardField);

  return (
    <div>
      <div>
        <label htmlFor="board-type">Board type</label>
        <select
          id="board-type"
          value={boardType}
          onChange={(e) => void setBoardType(e.target.value as "chess" | "aruco")}
        >
          <option value="chess">Chessboard</option>
          <option value="aruco">ChArUco</option>
        </select>
      </div>

      <div>
        <label>Board width</label>
        <input
          type="number"
          min={1}
          value={boardWidth}
          onChange={(e) => void setBoardField("boardWidth", Number(e.target.value))}
        />
      </div>
      <div>
        <label>Board height</label>
        <input
          type="number"
          min={1}
          value={boardHeight}
          onChange={(e) => void setBoardField("boardHeight", Number(e.target.value))}
        />
      </div>
      <div>
        <label>Square size (m)</label>
        <input
          type="number"
          step={0.001}
          min={0.01}
          value={squareSize}
          onChange={(e) => void setBoardField("squareSize", Number(e.target.value))}
        />
      </div>

      {boardType === "aruco" && (
        <div>
          <div>
            <label>Marker size (m)</label>
            <input
              type="number"
              step={0.001}
              min={0.01}
              value={markerSize}
              onChange={(e) => void setBoardField("markerSize", Number(e.target.value))}
            />
          </div>
          <div>
            <label>Dictionary</label>
            <select
              value={arucoDict}
              onChange={(e) => void setBoardField("arucoDict", e.target.value)}
            >
              {ARUCO_DICTS.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
          </div>
        </div>
      )}
    </div>
  );
}

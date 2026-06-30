// Skeleton landmark names and the drawn connections, ported verbatim from the
// old SKEL_NAMES / SKEL_CONNS.

export const SKEL_NAMES = [
  "Wrist", "Thumb CMC", "Thumb MCP", "Thumb IP", "Thumb Tip",
  "Index MCP", "Index PIP", "Index DIP", "Index Tip",
  "Middle MCP", "Middle PIP", "Middle DIP", "Middle Tip",
  "Ring MCP", "Ring PIP", "Ring DIP", "Ring Tip",
  "Pinky MCP", "Pinky PIP", "Pinky DIP", "Pinky Tip",
  "Elbow", "Shoulder",
];

// Drawn bones: Shoulder(22)-Elbow(21)-Wrist(0).
export const SKEL_CONNS: Array<[number, number]> = [
  [22, 21],
  [21, 0],
];

// Landmarks rendered as joints / used for 2D overlays.
export const SKEL_DRAW_INDICES = [0, 21, 22];

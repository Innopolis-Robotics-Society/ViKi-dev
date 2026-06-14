"""
viki.config
-----------
Centralised tunables for the ViKi capture server.

Values were previously scattered as literals across the server module.
Keeping them here makes the streaming/visualisation behaviour easy to
tweak without hunting through request handlers.
"""

# ── Camera-start defaults (mirror StartRequest) ────────────────────────────────
DEFAULT_FPS = 30
DEFAULT_COLOR_WIDTH = 640
DEFAULT_COLOR_HEIGHT = 480
DEFAULT_DEPTH_MODE = "NFOV_UNBINNED"

# ── Depth visualisation ────────────────────────────────────────────────────────
# EMA smoothing factor for the displayed depth range; range settles over
# ~20 frames (~0.67 s at 30 fps).
DEPTH_EMA_ALPHA = 0.05
# Minimum fraction of valid (non-zero) pixels for a depth frame to be displayed.
# Frames below this are dropped and the last good image is held instead.
DEPTH_MIN_VALID_FRACTION = 0.05

# ── Streaming / encoding ───────────────────────────────────────────────────────
JPEG_QUALITY = 80
STREAM_IDLE_SLEEP = 0.005  # seconds to wait when no new frame is available
PLACEHOLDER_SIZE = (640, 480)  # (width, height) of placeholder / mosaic tiles

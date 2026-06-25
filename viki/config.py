"""
viki.config
-----------
Centralised tunables for the ViKi capture server.

Values were previously scattered as literals across the server module.
Keeping them here makes the streaming/visualisation behaviour easy to
tweak without hunting through request handlers.
"""

INTRINSICS_FILENAME = "data/intrinsics_calibration.json"
EXTRINSICS_FILENAME = "data/extrinsics_calibration.json"

# ── Camera-start defaults (mirror StartRequest)
DEFAULT_FPS = 30
DEFAULT_COLOR_WIDTH = 1280, 720 # TODO changed from 640, 480
DEFAULT_COLOR_HEIGHT = 480
DEFAULT_DEPTH_MODE = "NFOV_UNBINNED"

# ── Depth visualisation
DEPTH_EMA_ALPHA = 0.05

DEPTH_MIN_VALID_FRACTION = 0.05

# ── Streaming / encoding
JPEG_QUALITY = 80
STREAM_IDLE_SLEEP = 0.005
PLACEHOLDER_SIZE = (1280, 720) # TODO changed from 640, 480

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

# ── Camera-start defaults
DEFAULT_FPS = 15
DEFAULT_COLOR_WIDTH = 1280
DEFAULT_COLOR_HEIGHT = 720
DEFAULT_DEPTH_MODE = "NFOV_UNBINNED"
DEFAULT_TIMEOUT_MS = 5000

# ── Kinect Sync defaults
DEFAULT_WIRED_SYNC_MODE = (
    0  # dont change, this is just a fallback. 0: Standalone, 1: Master, 2: Subordinate
)
DEFAULT_SUBORDINATE_DELAY_US = 0  # Delay in microseconds, probably dont change
DEFAULT_SYNCHRONIZED_IMAGES_ONLY = False

# ── Buffer settings
FRAME_BUFFER_SIZE = 12  # Number of frames kept per camera for sync queries

# ── Depth visualisation, probably dont change
DEPTH_EMA_ALPHA = 0.05
DEPTH_MIN_VALID_FRACTION = 0.05

# ── Streaming / encoding
JPEG_QUALITY = 80
STREAM_IDLE_SLEEP = 0.005
PLACEHOLDER_SIZE = (1280, 720)

# ── Recording
RECORD_DEPTH = False  # is depth recorded in a .npy format for each frame alongside the mp4 videos,
                      # can be disk-space consuming
DEPTH_PROJECTION_DEBUG = True # is the debug plotting of depth projection is enabled

SKELETON_DEPTH_SAMP_RADIUS = 20
SKELETON_DEPTH_BASE_DIR = "data/depth_bases/"
SKELETON_ENABLE_DEPTH_VALIDATION = True
MEDIAPIPE_ESTIMATION_CHECK = False

# -- SDK backend validation
DEPTH_VALIDATION_ENABLED = False # make the estimation model back up the depth lookup
DEPTH_VALIDATION_THRESHOLD_MM = 50

Z_CONVERGENCE_THRESHOLD = 0.1 # meters
# If the projected depth and MediaPipe estimate are within this range, they are averaged; 
# otherwise, the closer value is chosen.



# ── Detection
# CAMERAS_MIRRORED = False
HAND_TO_DETECT = "right"

# ── Calibration defaults
CALIB_MODE = "manual"  # "auto" (worker captures) or "manual" (via add_sample)
CALIB_BOARD_TYPE = "aruco"  # "chess" or "aruco"

# Manual bone lengths (meters). If provided, these override EMA tracking.
# Based on typical human proportions: Wrist-Elbow ~0.25m, Elbow-Shoulder ~0.30m
BONE_LENGTHS = {
    # (Parent LM, Child LM): length_m
    # Note: These are defaults; if you have precise measurements, update them here.
    # (22, 21): 0.30, # Shoulder -> Elbow
    # (21, 0): 0.25,  # Elbow -> Wrist
}
BONE_TOLERANCE = 0.1  # 5% tolerance range for soft kinematic constraints

# Chessboard parameters
CALIB_CHESS_BOARD_SIZE = (8, 6)  # (cols, rows)
CALIB_CHESS_SQUARE_SIZE = 0.025  # metres

# Aruco parameters
CALIB_ARUCO_BOARD_SIZE = (
    10,
    8,
)  # (cols, rows) dont use 8, 10 for the board we've been using,
#              it is specifically 10, 8
CALIB_ARUCO_SQUARE_SIZE = 0.05  # metres, the size of the black square
CALIB_ARUCO_MARKER_SIZE = (
    0.035  # metres, the size of the markers inside the white squares
)
# The Aruco dictionary ID is an integer used by OpenCV to identify which
# marker set to use. Common IDs:
# 0: DICT_4X4_50, 1: DICT_4X4_100, 2: DICT_4X4_250, 3: DICT_4X4_1000
# 4: DICT_5X5_50, 5: DICT_5X5_100, 6: DICT_5X5_250, 7: DICT_5X5_1000
# 8: DICT_6X6_50, 9: DICT_6X6_100, 10: DICT_6X6_250, 11: DICT_6X6_1000
# 12: DICT_7X7_50, 13: DICT_7X7_100, 14: DICT_7X7_250, 15: DICT_7X7_1000
CALIB_ARUCO_DICT = 4  # cv2.aruco.DICT_5X5_50, we have 40 markers each 5x5 squares

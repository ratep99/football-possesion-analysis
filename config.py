# config.py

# ===========================
# PATH CONFIGURATIONS
# ===========================

# Paths for video input, model, cache, and output
VIDEO_PATH = 'data/rfkzel.mp4'          # Path to the input video file
MODEL_PATH = 'model/1280res100ep.pt'    # Path to the YOLO model file
CACHE_PATH = 'cache/tracks_cache.pkl'   # Path for storing cached tracks
OUTPUT_PATH = 'data/output.avi'         # Path for the output video file


# ===========================
# TEAM AND OBJECT COLORS
# ===========================

# Default colors for different teams and objects in BGR format
HOME_TEAM_COLOR = [0, 0, 255]           # Red color for home team
AWAY_TEAM_COLOR = [255, 255, 255]       # White color for away team
REFEREE_COLOR = [0, 255, 255]           # Yellow color for referees
DEFAULT_BALL_COLOR = [0, 255, 0]        # Green color for ball when not possessed


# ===========================
# THRESHOLD SETTINGS
# ===========================

# Thresholds for detecting changes and overlaps
COLOR_CHANGE_THRESHOLD = 100            # Threshold for color change detection
OVERLAP_THRESHOLD = 0.2                 # Threshold for overlap detection (IoU > 0.2)
INITIALIZATION_FRAMES = 5               # Number of frames for initialization


# ===========================
# BALL ASSIGNER CONFIGURATION
# ===========================

# Configuration for ball possession assignment
BALL_ASSIGNER_DISTANCE_THRESHOLD = 150  # Distance threshold for assigning possession
BALL_ASSIGNER_POSSESSION_TIME_THRESHOLD = 12  # Time threshold for maintaining possession


# ===========================
# OBJECT DETECTOR SETTINGS
# ===========================

# Parameters for the object detector (YOLO)
DETECTOR_CONFIDENCE_THRESHOLD = 0.5     # Confidence threshold for object detection
DETECTOR_BATCH_SIZE = 20                # Batch size for YOLO model predictions
DETECTOR_IMAGE_SIZE = 1920              # Image size for YOLO predictions


# ===========================
# OBJECT TRACKER SETTINGS
# ===========================

# Parameters for the object tracker
TRACKER_TRACK_ACTIVATION_THRESHOLD = 0.6  # Activation threshold for starting new tracks
TRACKER_LOST_TRACK_BUFFER = 120           # Buffer size for lost tracks
TRACKER_MINIMUM_MATCHING_THRESHOLD = 1    # Minimum threshold for matching detections to tracks
TRACKER_MINIMUM_CONSECUTIVE_FRAMES = 5    # Minimum number of consecutive frames for track confirmation
FRAME_RATE = 30                           # Frame rate of the video

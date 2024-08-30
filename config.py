VIDEO_PATH = 'data/rfkzel.mp4'
MODEL_PATH = 'model/1280res100ep.pt'
CACHE_PATH = 'cache/tracks_cache.pkl'
OUTPUT_PATH = 'data/output.avi'

# Parameters for ObjectDetector
DETECTOR_CONFIDENCE_THRESHOLD = 0.5
DETECTOR_BATCH_SIZE = 20

# Parameters for ObjectTracker
TRACKER_TRACK_ACTIVATION_THRESHOLD = 0.6
TRACKER_LOST_TRACK_BUFFER = 120
TRACKER_MINIMUM_MATCHING_THRESHOLD = 1
TRACKER_MINIMUM_CONSECUTIVE_FRAMES = 5
FRAME_RATE = 30  # ili bilo koja druga vrednost koja odgovara vašem videu
VIDEO_PATH = 'data/Gfkmet.mp4'
MODEL_PATH = 'model/yolov8mf.pt'
CACHE_PATH = 'cache/tracks_cache.pkl'
OUTPUT_PATH = 'data/output.avi'

# Parameters for ObjectDetector
DETECTOR_CONFIDENCE_THRESHOLD = 0.5
DETECTOR_BATCH_SIZE = 10

# Parameters for ObjectTracker
TRACKER_TRACK_ACTIVATION_THRESHOLD = 0.45
TRACKER_LOST_TRACK_BUFFER = 30
TRACKER_MINIMUM_MATCHING_THRESHOLD = 0.95
TRACKER_MINIMUM_CONSECUTIVE_FRAMES = 5
FRAME_RATE = 30  # ili bilo koja druga vrednost koja odgovara vašem videu
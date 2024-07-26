from ultralytics import YOLO
import supervision as sv
import os
from utils import geometry_utils, drawing_utils
from cache import cache_utils

class Tracker:
    def __init__(self, model_path, confidence_threshold=0.3, batch_size=10):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size

    def detect_objects_on_frames(self, frames):
        """Detects objects in video frames using YOLO model."""
        detections = []
        for i in range(0, len(frames), self.batch_size):
            detections_batch = self.model.predict(frames[i:i + self.batch_size], conf=self.confidence_threshold)
            detections += detections_batch
        return detections

    def initialize_tracking_dictionaries(self, frames_number):
        """Initializes empty dictionaries for players, referees, and ball for all frames."""
        tracks = {
            "players": [{} for _ in range(frames_number)],
            "referees": [{} for _ in range(frames_number)],
            "ball": [{} for _ in range(frames_number)]
        }
        return tracks

    def convert_goalkeeper_to_player(self, detection_supervision, class_names):
        """Converts goalkeeper class to player class in detections."""
        for object_ind, class_id in enumerate(detection_supervision.class_id):
            if class_names[class_id] == "goalkeeper":
                detection_supervision.class_id[object_ind] = 2 #player
        
    def update_player_and_referee_tracks(self, tracks, frame_number, detection_with_tracks):
        """Updates player and referee tracks with bounding boxes and class IDs."""
        for frame_detection in detection_with_tracks:
            bounding_box = frame_detection[0].tolist()
            class_id = frame_detection[3]
            track_id = frame_detection[4]
            
            if class_id == 2:
                if track_id not in tracks["players"][frame_number]:
                    tracks["players"][frame_number][track_id] = {"bounding_box": bounding_box}

            if class_id == 3:
                if track_id not in tracks["referees"][frame_number]:
                    tracks["referees"][frame_number][track_id] = {"bounding_box": bounding_box}

    def update_ball_tracks(self, tracks, frame_number, detection_supervision):
        """Updates ball tracks with bounding boxes and class IDs."""
        for frame_detection in detection_supervision:
            bounding_box = frame_detection[0].tolist()
            class_id = frame_detection[3]

            if class_id == 0: #ball
                tracks["ball"][frame_number][1] = {"bounding_box": bounding_box}

    def get_object_tracks(self, frames, read_from_cache=False, cache_path=None):
        """Gets object tracks from video frames."""
        if read_from_cache and cache_path and os.path.exists(cache_path):
            return cache_utils.load_tracks_from_cache(cache_path)

        detections = self.detect_objects_on_frames(frames)
        tracks = self.initialize_tracking_dictionaries(len(frames))

        for frame_number, detection in enumerate(detections):
            #{0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
            cls_names = detection.names
            detection_supervision = sv.Detections.from_ultralytics(detection)
            self.convert_goalkeeper_to_player(detection_supervision, cls_names)

            #Tracking logics
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            self.update_player_and_referee_tracks(tracks, frame_number, detection_with_tracks)
            self.update_ball_tracks(tracks, frame_number, detection_supervision)

        if cache_path:
            cache_utils.save_tracks_to_cache(tracks, cache_path)

        return tracks

from ultralytics import YOLO
import supervision as sv
import os

# Importing modules from the same directory
from utils import geometry_utils, drawing_utils
from cache import cache_utils

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_objects_on_frames(self, frames, batch_size=20, conf=0.35):
        """Detects objects in video frames using YOLO model."""
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=conf)
            detections += detections_batch
        return detections

    def initialize_tracking_dictionaries(self, num_frames):
        """Initializes empty dictionaries for players, referees, and ball for all frames."""
        tracks = {
            "players": [{} for _ in range(num_frames)],
            "referees": [{} for _ in range(num_frames)],
            "ball": [{} for _ in range(num_frames)]
        }
        return tracks

    def convert_goalkeeper_to_player(self, detection_supervision, cls_names, cls_names_inv):
        """Converts goalkeeper class to player class in detections."""
        for object_ind, class_id in enumerate(detection_supervision.class_id):
            if cls_names[class_id] == "goalkeeper":
                detection_supervision.class_id[object_ind] = cls_names_inv["player"]
        
    def update_player_and_referee_tracks(self, tracks, frame_num, detection_with_tracks, cls_names_inv):
        """Updates player and referee tracks with bounding boxes and class IDs."""
        for frame_detection in detection_with_tracks:
            bbox = frame_detection[0].tolist()
            cls_id = frame_detection[3]
            track_id = frame_detection[4]

            if cls_id == cls_names_inv['player']:
                if track_id not in tracks["players"][frame_num]:
                    tracks["players"][frame_num][track_id] = {"bounding_box": bbox}

            if cls_id == cls_names_inv['referee']:
                if track_id not in tracks["referees"][frame_num]:
                    tracks["referees"][frame_num][track_id] = {"bounding_box": bbox}

    def update_ball_tracks(self, tracks, frame_num, detection_supervision, cls_names_inv):
        """Updates ball tracks with bounding boxes and class IDs."""
        for frame_detection in detection_supervision:
            bbox = frame_detection[0].tolist()
            cls_id = frame_detection[3]

            if cls_id == cls_names_inv['ball']:
                tracks["ball"][frame_num][1] = {"bounding_box": bbox}

    def get_object_tracks(self, frames, read_from_cache=False, cache_path=None):
        """Gets object tracks from video frames."""
        num_frames = len(frames)  # Dodavanje broja frejmova
        if read_from_cache and cache_path and os.path.exists(cache_path):
            return cache_utils.load_tracks_from_cache(cache_path)

        detections = self.detect_objects_on_frames(frames)
        tracks = self.initialize_tracking_dictionaries(num_frames)  # Prosleđivanje broja frejmova

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)
            self.convert_goalkeeper_to_player(detection_supervision, cls_names, cls_names_inv)
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            self.update_player_and_referee_tracks(tracks, frame_num, detection_with_tracks, cls_names_inv)
            self.update_ball_tracks(tracks, frame_num, detection_supervision, cls_names_inv)

        if cache_path:
            cache_utils.save_tracks_to_cache(tracks, cache_path)

        return tracks

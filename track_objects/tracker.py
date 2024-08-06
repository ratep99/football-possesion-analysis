from ultralytics import YOLO
import supervision as sv
import os
from utils import geometry_utils
from cache import cache_utils
import pandas as pd
import constants

class Tracker:
    def __init__(self, model_path, confidence_threshold=0.5, batch_size=10):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.45, 
            lost_track_buffer=30,            
            minimum_matching_threshold=0.95, 
            minimum_consecutive_frames=5    
        )        
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size

    def detect_objects_on_frames(self, frames):
        detections = []
        for i in range(0, len(frames), self.batch_size):
            detections_batch = self.model.predict(frames[i:i + self.batch_size], conf=self.confidence_threshold)
            detections += detections_batch
        return detections

    def initialize_tracking_dictionaries(self, frames_number):
        tracks = {
            constants.PLAYERS_KEY: [{} for _ in range(frames_number)],
            constants.REFEREES_KEY: [{} for _ in range(frames_number)],
            constants.BALL_KEY: [{} for _ in range(frames_number)]
        }
        return tracks

    def convert_goalkeeper_to_player(self, detection_supervision, class_names):
        for object_ind, class_id in enumerate(detection_supervision.class_id):
            if class_names[class_id] == constants.GOALKEEPER_KEY:
                detection_supervision.class_id[object_ind] = 2 #player

    def update_player_and_referee_tracks(self, tracks, frame_number, detection_with_tracks):
        for frame_detection in detection_with_tracks:
            bounding_box = frame_detection[0].tolist()
            class_id = frame_detection[3]
            track_id = frame_detection[4]

            if class_id == 2:
                tracks[constants.PLAYERS_KEY][frame_number][track_id] = {constants.BOUNDING_BOX_KEY: bounding_box}

            if class_id == 3:
                tracks[constants.REFEREES_KEY][frame_number][track_id] = {constants.BOUNDING_BOX_KEY: bounding_box}

    def update_ball_tracks(self, tracks, frame_number, detection_supervision):
        for frame_detection in detection_supervision:
            bounding_box = frame_detection[0].tolist()
            class_id = frame_detection[3]

            if class_id == 0: #ball
                tracks[constants.BALL_KEY][frame_number][1] = {constants.BOUNDING_BOX_KEY: bounding_box}

    def get_object_tracks(self, frames, read_from_cache=False, cache_path=None):
        if read_from_cache and cache_path and os.path.exists(cache_path):
            return cache_utils.load_tracks_from_cache(cache_path)

        detections = self.detect_objects_on_frames(frames)
        tracks = self.initialize_tracking_dictionaries(len(frames))

        for frame_number, detection in enumerate(detections):
            cls_names = detection.names
            detection_supervision = sv.Detections.from_ultralytics(detection)
            self.convert_goalkeeper_to_player(detection_supervision, cls_names)

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            self.update_player_and_referee_tracks(tracks, frame_number, detection_with_tracks)
            self.update_ball_tracks(tracks, frame_number, detection_supervision)
            self.interpolate_ball_positions(tracks[constants.BALL_KEY])

        if cache_path:
            cache_utils.save_tracks_to_cache(tracks, cache_path)

        return tracks

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get(constants.BOUNDING_BOX_KEY, []) for x in ball_positions]
        if not any(ball_positions):  # Check if all entries are empty
            print("No ball positions detected.")
            return ball_positions

        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {constants.BOUNDING_BOX_KEY: x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

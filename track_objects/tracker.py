import supervision as sv
import config
import constants
from cache import cache_utils  # Import za keširanje
import os

class ObjectTracker:
    def __init__(self):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=config.TRACKER_TRACK_ACTIVATION_THRESHOLD,
            lost_track_buffer=config.TRACKER_LOST_TRACK_BUFFER,
            minimum_matching_threshold=config.TRACKER_MINIMUM_MATCHING_THRESHOLD,
            minimum_consecutive_frames=config.TRACKER_MINIMUM_CONSECUTIVE_FRAMES
        )

    def initialize_tracking_dictionaries(self, frames_number):
        tracks = {
            constants.PLAYERS_KEY: [{} for _ in range(frames_number)],
            constants.REFEREES_KEY: [{} for _ in range(frames_number)],
            constants.BALL_KEY: [{} for _ in range(frames_number)]
        }
        return tracks

    def get_cached_tracks(self, cache_path):
        """
        Returns cached tracks if they exist. Otherwise, returns None.
        """
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached tracks from {cache_path}")
            return cache_utils.load_tracks_from_cache(cache_path)
        else:
            print("No cached tracks found.")
            return None

    def track_objects(self, detections, frames_number):
        tracks = self.initialize_tracking_dictionaries(frames_number)

        for frame_number, detection in enumerate(detections):
            cls_names = detection.names
            detection_supervision = sv.Detections.from_ultralytics(detection)

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            self.update_player_and_referee_tracks(tracks, frame_number, detection_with_tracks)
            self.update_ball_tracks(tracks, frame_number, detection_supervision)

        return tracks

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

            if class_id == 0:  # Ball class ID
                tracks[constants.BALL_KEY][frame_number][1] = {constants.BOUNDING_BOX_KEY: bounding_box}

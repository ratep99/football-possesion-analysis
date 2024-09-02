import supervision as sv
import config
import constants
from cache import cache_utils
import os
import pandas as pd
from typing import List, Dict, Optional

class ObjectTracker:
    def __init__(self):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=config.TRACKER_TRACK_ACTIVATION_THRESHOLD,
            lost_track_buffer=config.TRACKER_LOST_TRACK_BUFFER,
            minimum_matching_threshold=config.TRACKER_MINIMUM_MATCHING_THRESHOLD,
            minimum_consecutive_frames=config.TRACKER_MINIMUM_CONSECUTIVE_FRAMES
        )

    def initialize_tracking_dictionaries(self, frames_number: int) -> Dict[str, List[Dict[int, Dict]]]:
        """
        Initializes tracking dictionaries for players, referees, and the ball.

        :param frames_number: Number of frames in the video.
        :return: A dictionary containing empty tracking information.
        """
        return {
            constants.PLAYERS_KEY: [{} for _ in range(frames_number)],
            constants.REFEREES_KEY: [{} for _ in range(frames_number)],
            constants.BALL_KEY: [{} for _ in range(frames_number)]
        }

    def get_cached_tracks(self, cache_path: str) -> Optional[Dict[str, List[Dict[int, Dict]]]]:
        """
        Returns cached tracks if they exist. Otherwise, returns None.

        :param cache_path: Path to the cached tracks file.
        :return: Cached tracks or None if not available.
        """
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached tracks from {cache_path}")
            return cache_utils.load_tracks_from_cache(cache_path)
        else:
            print("No cached tracks found.")
            return None

    def track_objects(self, detections: List, frames_number: int) -> Dict[str, List[Dict[int, Dict]]]:
        """
        Tracks objects (players, referees, and ball) across video frames.

        :param detections: List of detections for each frame.
        :param frames_number: Number of frames in the video.
        :return: Updated tracks with players, referees, and ball information.
        """
        tracks = self.initialize_tracking_dictionaries(frames_number)

        for frame_number, detection in enumerate(detections):
            detection_supervision = sv.Detections.from_ultralytics(detection)
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            self.update_player_and_referee_tracks(tracks, frame_number, detection_with_tracks)
            self.update_ball_tracks(tracks, frame_number, detection_supervision)

        return tracks

    def update_player_and_referee_tracks(self, tracks: Dict[str, List[Dict[int, Dict]]], frame_number: int, 
                                         detection_with_tracks: List):
        """
        Updates player and referee tracks in the tracking data.

        :param tracks: The tracking data dictionary.
        :param frame_number: The current frame number.
        :param detection_with_tracks: Detections that have been assigned a track.
        """
        for frame_detection in detection_with_tracks:
            bounding_box = frame_detection[0].tolist()
            class_id = frame_detection[3]
            track_id = frame_detection[4]

            if class_id == constants.PLAYER_CLASS_ID:
                tracks[constants.PLAYERS_KEY][frame_number][track_id] = {constants.BOUNDING_BOX_KEY: bounding_box}

            elif class_id == constants.REFEREE_CLASS_ID:
                tracks[constants.REFEREES_KEY][frame_number][track_id] = {constants.BOUNDING_BOX_KEY: bounding_box}

    def update_ball_tracks(self, tracks: Dict[str, List[Dict[int, Dict]]], frame_number: int, 
                           detection_supervision: sv.Detections):
        """
        Updates ball tracks in the tracking data.

        :param tracks: The tracking data dictionary.
        :param frame_number: The current frame number.
        :param detection_supervision: Detections in the current frame.
        """
        for frame_detection in detection_supervision:
            bounding_box = frame_detection[0].tolist()
            class_id = frame_detection[3]

            if class_id == constants.BALL_CLASS_ID:
                tracks[constants.BALL_KEY][frame_number][1] = {constants.BOUNDING_BOX_KEY: bounding_box}

    def interpolate_ball_positions(self, ball_positions: List[Dict[int, Dict]], min_consecutive_frames: int = 8) -> List[Dict[int, Dict]]:
        """
        Interpolates missing ball positions to ensure smoother tracking,
        discarding false positives based on consecutive frame threshold.

        :param ball_positions: List of dictionaries representing ball positions per frame.
        :param min_consecutive_frames: Minimum number of consecutive frames required to consider a new position valid.
        :return: Updated list with interpolated positions.
        """
        ball_positions_data = [
            entry.get(1, {}).get(constants.BOUNDING_BOX_KEY, [None, None, None, None]) 
            for entry in ball_positions
        ]

        valid_positions = self.filter_false_positives(ball_positions_data, min_consecutive_frames)

        df_ball_positions = pd.DataFrame(valid_positions, columns=['x1', 'y1', 'x2', 'y2'])
        df_ball_positions.interpolate(method='linear', limit_direction='both', inplace=True)
        df_ball_positions.bfill(inplace=True)

        interpolated_positions = df_ball_positions.to_numpy().tolist()
        updated_ball_positions = [
            {1: {constants.BOUNDING_BOX_KEY: pos}} if pos != [None, None, None, None] else {}
            for pos in interpolated_positions
        ]

        return updated_ball_positions

    def filter_false_positives(self, positions: List[List[Optional[float]]], min_consecutive_frames: int) -> List[List[Optional[float]]]:
        """
        Filters out false positives in ball positions based on consecutive frames threshold.

        :param positions: List of positions per frame.
        :param min_consecutive_frames: Minimum number of consecutive frames required to consider a new position valid.
        :return: List of positions with false positives filtered out.
        """
        valid_positions = []
        last_valid_position = None
        consecutive_frames = 0

        for pos in positions:
            if pos == [None, None, None, None]:
                valid_positions.append(pos)
                consecutive_frames = 0
                continue

            if last_valid_position is not None:
                current_center = ((pos[0] + pos[2]) / 2, (pos[1] + pos[3]) / 2)
                last_center = ((last_valid_position[0] + last_valid_position[2]) / 2, (last_valid_position[1] + last_valid_position[3]) / 2)

                if current_center != last_center:
                    consecutive_frames += 1
                else:
                    consecutive_frames = 0

                if consecutive_frames >= min_consecutive_frames:
                    last_valid_position = pos
                    valid_positions.append(pos)
                else:
                    valid_positions.append([None, None, None, None])
            else:
                last_valid_position = pos
                valid_positions.append(pos)
                consecutive_frames = 1

        return valid_positions

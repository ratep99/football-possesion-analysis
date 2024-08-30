import cv2
import numpy as np
from utils import video_control_utils
from track_objects import ObjectTracker
from detect_objects import ObjectDetector
from draw import Drawer
from classify_players import TeamClassifier
from assign_ball import BallAssigner
from calculate_possession import PossessionCalculator
from cache import cache_utils
import config
import constants

class FootballAnalyzer:
    def __init__(self):
        self.video_path = config.VIDEO_PATH
        self.model_path = config.MODEL_PATH
        self.cache_path = config.CACHE_PATH
        self.output_path = config.OUTPUT_PATH

        self.video = None
        self.tracks = None
        
        self.detector = ObjectDetector() 
        self.tracker = ObjectTracker()
        self.team_assigner = TeamClassifier()
        self.player_assigner = BallAssigner()
        self.possession_calculator = PossessionCalculator()
        self.drawer = Drawer()

    def run(self):
        self.video = video_control_utils.read_video(self.video_path)
        
        if not self.video or self.video[0].shape[0] == 0 or self.video[0].shape[1] == 0:
            raise ValueError("Invalid video frame dimensions.")
        
        self.tracks = self.tracker.get_cached_tracks(self.cache_path)

        if self.tracks is None:
            detections = self.detector.detect_objects_on_frames(self.video)
            self.tracks = self.tracker.track_objects(detections, len(self.video))
            cache_utils.save_tracks_to_cache(self.tracks, self.cache_path)

        for frame_num, player_detections in enumerate(self.tracks[constants.PLAYERS_KEY]):
            if player_detections:
                self.team_assigner.assign_team_colors(self.video[frame_num], player_detections)
        
        self.team_assigner.assign_teams_to_players(self.tracks, self.video)
       
        team_ball_control = self.player_assigner.assign_ball_control(self.tracks)

        video_frames = []

        for frame_num in range(len(self.video)):
            home_team_time, away_team_time, home_team_possession, away_team_possession = self.possession_calculator.calculate_possession(team_ball_control[:frame_num + 1])

            frame = self.drawer.draw_annotations(
                frame_num,  # Prosleđivanje trenutnog frejma
                self.video[frame_num],  # Frejm
                self.tracks, 
                team_ball_control,
                home_team_time, 
                away_team_time, 
                home_team_possession, 
                away_team_possession,
                self.team_assigner.home_team_color,  # Dodavanje boje domaćeg tima
                self.team_assigner.away_team_color   # Dodavanje boje gostujućeg tima
            )


            video_frames.append(frame)

        video_control_utils.save_video(video_frames, self.output_path)

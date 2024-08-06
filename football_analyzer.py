import cv2
import numpy as np
from utils import video_control_utils
from track_objects import Tracker
from draw import Drawer
from classify_players import TeamClassifier
from assign_ball import BallAssigner
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
        
        self.team_assigner = TeamClassifier()
        self.tracker = Tracker(self.model_path)
        self.drawer = Drawer()
        self.player_assigner = BallAssigner()

    def run(self):
        self.video = video_control_utils.read_video(self.video_path)
        self.tracks = self.tracker.get_object_tracks(self.video, read_from_cache=True, cache_path=self.cache_path)
        for frame_num, player_detections in enumerate(self.tracks[constants.PLAYERS_KEY]):
            if player_detections:
                self.team_assigner.assign_team_color(self.video[frame_num], player_detections) 
        
        self.team_assigner.assign_teams_to_players(self.tracks, self.video)
        team_ball_control = self.player_assigner.assign_ball_control(self.tracks, self.video)
        output_video = self.drawer.draw_annotations(self.video, self.tracks, team_ball_control)
        video_control_utils.save_video(output_video, self.output_path)

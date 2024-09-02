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
        """
        Initializes the FootballAnalyzer with paths and components for video analysis.
        """
        # Paths configuration
        self.video_path = config.VIDEO_PATH
        self.model_path = config.MODEL_PATH
        self.cache_path = config.CACHE_PATH
        self.output_path = config.OUTPUT_PATH

        # Video and tracking data initialization
        self.video = None
        self.tracks = None
        
        # Initialize components for object detection, tracking, and analysis
        self.detector = ObjectDetector() 
        self.tracker = ObjectTracker()
        self.team_assigner = TeamClassifier()
        self.player_assigner = BallAssigner()
        self.possession_calculator = PossessionCalculator()
        self.drawer = Drawer()

    def run(self):
        """
        Main method to run the football analysis pipeline.
        """
        # Load video
        self.video = video_control_utils.read_video(self.video_path)
        
        # Validate the loaded video frames
        if not self.video or self.video[0].shape[0] == 0 or self.video[0].shape[1] == 0:
            raise ValueError("Invalid video frame dimensions.")
        
        # Load cached tracks if available, otherwise perform detection and tracking
        self.tracks = self.tracker.get_cached_tracks(self.cache_path)

        if self.tracks is None:
            # Detect objects in video frames
            detections = self.detector.detect_objects_on_frames(self.video)
            # Track detected objects across frames
            self.tracks = self.tracker.track_objects(detections, len(self.video))
            # Cache the tracks to avoid recomputation in future runs
            cache_utils.save_tracks_to_cache(self.tracks, self.cache_path)

        # Interpolate missing ball positions to improve continuity in tracking
        self.tracks[constants.BALL_KEY] = self.tracker.interpolate_ball_positions(self.tracks[constants.BALL_KEY])

        # Initialize team colors based on initial frames to improve accuracy in team assignment
        frames_for_initialization = [self.video[i] for i in range(self.team_assigner.initialization_frames)]
        player_detections_list = [self.tracks[constants.PLAYERS_KEY][i] for i in range(self.team_assigner.initialization_frames)]
        self.team_assigner.initialize_team_colors(frames_for_initialization, player_detections_list)

        # Assign teams to players after initialization of team colors
        self.team_assigner.assign_teams_to_players(self.tracks, self.video)
       
        # Assign ball control to players to determine which team is in possession
        team_ball_control = self.player_assigner.assign_ball_control(self.tracks)

        # Prepare list to hold processed video frames
        video_frames = []

        # Process each frame to annotate and calculate possession statistics
        for frame_num in range(len(self.video)):
            # Calculate possession statistics up to the current frame
            home_team_time, away_team_time, home_team_possession, away_team_possession = self.possession_calculator.calculate_possession(team_ball_control[:frame_num + 1])

            # Draw annotations on the current frame
            frame = self.drawer.draw_annotations(
                frame_num,  # Current frame number
                self.video[frame_num],  # Current frame
                self.tracks, 
                team_ball_control,
                home_team_time, 
                away_team_time, 
                home_team_possession, 
                away_team_possession,
                self.team_assigner.home_team_color,  # Home team color
                self.team_assigner.away_team_color   # Away team color
            )

            # Append the annotated frame to the output video list
            video_frames.append(frame)

        # Save the processed video with annotations
        video_control_utils.save_video(video_frames, self.output_path)


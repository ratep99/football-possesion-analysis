import sys 
import constants
sys.path.append('../')
from utils import geometry_utils
import numpy as np

class BallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70
        self.frame_rate = constants.FRAME_RATE

    def is_ball_detected(self, ball_tracks, frame_num):
        return 1 in ball_tracks[frame_num]

    def assign_ball_to_player(self, players, ball_bounding_box):
        ball_position = geometry_utils.get_center_of_bounding_box(ball_bounding_box)

        min_distance = float('inf')
        assigned_player = -1

        for player_id, player in players.items():
            player_bounding_box = player[constants.BOUNDING_BOX_KEY]

            distance_left = geometry_utils.measure_distance((player_bounding_box[0], player_bounding_box[1]), ball_position)
            distance_right = geometry_utils.measure_distance((player_bounding_box[2], player_bounding_box[3]), ball_position)
            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance and distance < min_distance:
                min_distance = distance
                assigned_player = player_id

        return assigned_player

    def assign_ball_control(self, tracks, video):
        team_ball_control = []
        team_1_time = 0
        team_2_time = 0

        for frame_num, player_track in enumerate(tracks[constants.PLAYERS_KEY]):
            if self.is_ball_detected(tracks[constants.BALL_KEY], frame_num):
                ball_bounding_box = tracks[constants.BALL_KEY][frame_num][1][constants.BOUNDING_BOX_KEY]
                assigned_player = self.assign_ball_to_player(player_track, ball_bounding_box)

                if assigned_player != -1:
                    tracks[constants.PLAYERS_KEY][frame_num][assigned_player][constants.HAS_BALL_KEY] = True
                    team_id = tracks[constants.PLAYERS_KEY][frame_num][assigned_player][constants.TEAM_KEY]
                    team_ball_control.append(team_id)
                    
                    if team_id == 1:
                        team_1_time += 1 / self.frame_rate
                    else:
                        team_2_time += 1 / self.frame_rate
            else:
                print(f"Frame {frame_num}: Ball not detected.")

        return np.array(team_ball_control), team_1_time, team_2_time

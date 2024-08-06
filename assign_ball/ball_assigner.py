import sys 
import constants
sys.path.append('../')
from utils import geometry_utils
import numpy as np

class BallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70
    
    def assign_ball_to_player(self,players,ball_bounding_box):
        ball_position = geometry_utils.get_center_of_bounding_box(ball_bounding_box)

        miniumum_distance = 99999
        assigned_player=-1

        for player_id, player in players.items():
            player_bounding_box = player[constants.BOUNDING_BOX_KEY]

            distance_left = geometry_utils.measure_distance((player_bounding_box[0],player_bounding_box[-1]),ball_position)
            distance_right = geometry_utils.measure_distance((player_bounding_box[2],player_bounding_box[-1]),ball_position)
            distance = min(distance_left,distance_right)

            if distance < self.max_player_ball_distance:
                if distance < miniumum_distance:
                    miniumum_distance = distance
                    assigned_player = player_id

        return assigned_player
    
    def assign_ball_control(self, tracks, video):
        team_ball_control = []
        for frame_num, player_track in enumerate(tracks[constants.PLAYERS_KEY]):
            if 1 in tracks[constants.BALL_KEY][frame_num]:
                ball_bounding_box = tracks[constants.BALL_KEY][frame_num][1][constants.BOUNDING_BOX_KEY]
                assigned_player = self.assign_ball_to_player(player_track, ball_bounding_box)
                if assigned_player != -1:
                    tracks[constants.PLAYERS_KEY][frame_num][assigned_player][constants.HAS_BALL_KEY] = True
                    team_ball_control.append(tracks[constants.PLAYERS_KEY][frame_num][assigned_player][constants.TEAM_KEY])
            else:
                print(f"Frame {frame_num}: Ball not detected.")
        return np.array(team_ball_control)
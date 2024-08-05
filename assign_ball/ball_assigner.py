import sys 
sys.path.append('../')
from utils import geometry_utils

class BallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70
    
    def assign_ball_to_player(self,players,ball_bounding_box):
        ball_position = geometry_utils.get_center_of_bounding_box(ball_bounding_box)

        miniumum_distance = 99999
        assigned_player=-1

        for player_id, player in players.items():
            player_bounding_box = player['bounding_box']

            distance_left = geometry_utils.measure_distance((player_bounding_box[0],player_bounding_box[-1]),ball_position)
            distance_right = geometry_utils.measure_distance((player_bounding_box[2],player_bounding_box[-1]),ball_position)
            distance = min(distance_left,distance_right)

            if distance < self.max_player_ball_distance:
                if distance < miniumum_distance:
                    miniumum_distance = distance
                    assigned_player = player_id

        return assigned_player
import constants
from utils import geometry_utils

class BallAssigner:
    def __init__(self):
        self.last_team_with_possession = None

    def assign_ball_control(self, tracks):
        team_ball_control = []

        for frame_num, ball_positions in enumerate(tracks[constants.BALL_KEY]):
            if ball_positions:
                ball_position = ball_positions[1][constants.BOUNDING_BOX_KEY]
                closest_player, team_id = self.get_closest_player(tracks[constants.PLAYERS_KEY][frame_num], ball_position)
                if closest_player is not None:
                    team_ball_control.append(team_id)
                    self.last_team_with_possession = team_id
                else:
                    # Ако нема најближег играча, користимо последњи тим са поседом
                    team_ball_control.append(self.last_team_with_possession)
            else:
                # Ако нема података о лопти, користимо последњи познати посед
                team_ball_control.append(self.last_team_with_possession)

        return team_ball_control

    def get_closest_player(self, players, ball_position):
        closest_distance = float('inf')
        closest_player = None
        closest_team_id = None

        for player_id, player_data in players.items():
            player_position = player_data[constants.BOUNDING_BOX_KEY]
            distance = geometry_utils.measure_distance(ball_position, player_position)

            if distance < closest_distance:
                closest_distance = distance
                closest_player = player_id
                closest_team_id = player_data[constants.TEAM_KEY]
        return closest_player, closest_team_id

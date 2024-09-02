import constants
import config
from utils import geometry_utils

class BallAssigner:
    def __init__(self):
        """
        Initializes the BallAssigner object for assigning ball control to players.
        Uses configuration values from config.py.
        """
        self.last_team_with_possession = None
        self.last_player_with_possession = None
        self.frames_with_ball = 0  # Counter for frames during which the ball is close to the same player
        
        # Load configuration values from config.py
        self.distance_threshold = config.BALL_ASSIGNER_DISTANCE_THRESHOLD
        self.possession_time_threshold = config.BALL_ASSIGNER_POSSESSION_TIME_THRESHOLD

    def assign_ball_control(self, tracks):
        """
        Determines ball possession for each frame based on player proximity to the ball.

        :param tracks: Tracking data for all frames containing players and ball positions.
        :return: List indicating which team has ball control for each frame.
        """
        team_ball_control = []

        for frame_num, ball_positions in enumerate(tracks[constants.BALL_KEY]):
            if ball_positions:
                ball_position = ball_positions[1][constants.BOUNDING_BOX_KEY]
                closest_player, team_id, distance = self.get_closest_player(
                    tracks[constants.PLAYERS_KEY][frame_num], ball_position)
                
                if closest_player is not None and distance <= self.distance_threshold:
                    if self.last_player_with_possession == closest_player:
                        self.frames_with_ball += 1
                    else:
                        self.frames_with_ball = 1  # Reset counter if a different player gets close
                        self.last_player_with_possession = closest_player

                    # Check if the ball has been with the same player long enough to be considered in possession
                    if self.frames_with_ball >= self.possession_time_threshold:
                        team_ball_control.append(team_id)
                        self.last_team_with_possession = team_id
                    else:
                        # Use the last known possession if the threshold is not met
                        team_ball_control.append(self.last_team_with_possession)
                else:
                    # If no close player or distance is too far, use the last known team with possession
                    team_ball_control.append(self.last_team_with_possession)
            else:
                # If no ball data, use the last known possession
                team_ball_control.append(self.last_team_with_possession)

        return team_ball_control

    def get_closest_player(self, players, ball_position):
        """
        Finds the closest player to the ball.

        :param players: Dictionary containing player positions for a frame.
        :param ball_position: The current position of the ball.
        :return: Tuple containing the closest player ID, their team ID, and the distance to the ball.
        """
        closest_distance = float('inf')
        closest_player = None
        closest_team_id = None

        for player_id, player_data in players.items():
            player_bounding_box = player_data[constants.BOUNDING_BOX_KEY]
            player_foot_position = geometry_utils.get_foot_position(player_bounding_box)
            distance = geometry_utils.measure_distance(ball_position, player_foot_position)

            if distance < closest_distance:
                closest_distance = distance
                closest_player = player_id
                closest_team_id = player_data[constants.TEAM_KEY]
                
        return closest_player, closest_team_id, closest_distance

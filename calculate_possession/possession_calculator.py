import numpy as np
import constants
import config

class PossessionCalculator:
    def __init__(self):
        """
        Initializes the PossessionCalculator with the frame rate from the configuration.
        """
        self.frame_rate = config.FRAME_RATE

    def calculate_possession(self, team_ball_control):
        """
        Calculates the possession time and percentage for both teams based on ball control data.

        :param team_ball_control: List indicating which team has ball control for each frame.
        :return: Tuple containing home team possession time, away team possession time,
                 home team possession percentage, and away team possession percentage.
        """
        # Total time of the match in seconds
        total_time = len(team_ball_control) / self.frame_rate
        
        # Calculate possession time in seconds for each team
        home_team_time = np.sum(np.array(team_ball_control) == constants.HOME_TEAM_ID) / self.frame_rate
        away_team_time = np.sum(np.array(team_ball_control) == constants.AWAY_TEAM_ID) / self.frame_rate

        # Calculate possession percentage, rounded to the nearest whole number
        home_team_possession = round((home_team_time / total_time) * 100) if total_time > 0 else 50
        away_team_possession = 100 - home_team_possession  # Ensures total is always 100%

        return home_team_time, away_team_time, home_team_possession, away_team_possession

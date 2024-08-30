import numpy as np
import constants
import config

class PossessionCalculator:
    def __init__(self):
        self.frame_rate = config.FRAME_RATE

    def calculate_possession(self, team_ball_control):
        total_time = len(team_ball_control) / self.frame_rate
        home_team_time = np.sum(np.array(team_ball_control) == constants.HOME_TEAM_ID) / self.frame_rate
        away_team_time = np.sum(np.array(team_ball_control) == constants.AWAY_TEAM_ID) / self.frame_rate

        # Израчунавање процента поседа са заокруживањем на најближи цели број
        home_team_possession = round((home_team_time / total_time) * 100) if total_time > 0 else 50
        away_team_possession = 100 - home_team_possession  # Корекција да збир буде увек 100%

        return home_team_time, away_team_time, home_team_possession, away_team_possession

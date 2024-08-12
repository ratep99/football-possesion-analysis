import cv2
import numpy as np
from utils import geometry_utils
import constants
from .scoreboard import Scoreboard

class Drawer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.scoreboard = Scoreboard()

    def draw_ellipse(self, frame, bounding_box, color, track_id=None):
        y2 = int(bounding_box[3])
        x_center, _ = geometry_utils.get_center_of_bounding_box(bounding_box)
        width = geometry_utils.get_bounding_box_width(bounding_box)

        color = tuple(map(int, color))

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA  # Poboljšanje linije da bude glatka
        )

        if track_id is not None:
            text_y = y2 + 15
            cv2.putText(frame, f"{track_id}", (x_center - 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return frame

    def draw_triangle(self, frame, bounding_box, color):
        y = int(bounding_box[1])
        x, _ = geometry_utils.get_center_of_bounding_box(bounding_box)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control, home_team_time, away_team_time, home_team_possession, away_team_possession):
        frame = self.scoreboard.draw(frame, home_team_possession, away_team_possession, home_team_time, away_team_time)
        return frame

    def draw_annotations(self, frame_num, frame, tracks, team_ball_control, home_team_time, away_team_time, home_team_possession, away_team_possession):
        player_dict = tracks[constants.PLAYERS_KEY][frame_num]
        ball_dict = tracks[constants.BALL_KEY][frame_num]
        referee_dict = tracks[constants.REFEREES_KEY][frame_num]

        # Crtanje igrača
        for track_id, player in player_dict.items():
            color = player.get(constants.TEAM_COLOR_KEY, (0, 0, 255))
            frame = self.draw_ellipse(frame, player[constants.BOUNDING_BOX_KEY], color, track_id)

            # Crtanje crvenog trougla ako igrač ima loptu
            if player.get(constants.HAS_BALL_KEY, False):
                frame = self.draw_triangle(frame, player[constants.BOUNDING_BOX_KEY], (0, 0, 255))

        # Crtanje sudija
        for _, referee in referee_dict.items():
            frame = self.draw_ellipse(frame, referee[constants.BOUNDING_BOX_KEY], (0, 255, 255))

        # Crtanje lopte
        for track_id, ball in ball_dict.items():
            frame = self.draw_triangle(frame, ball[constants.BOUNDING_BOX_KEY], (0, 255, 0))
        
        # Crtanje timskog poseda i vremena
        frame = self.draw_team_ball_control(frame, frame_num, team_ball_control, home_team_time, away_team_time, home_team_possession, away_team_possession)

        return frame

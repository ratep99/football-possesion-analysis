import cv2
import numpy as np
from utils import geometry_utils
import constants
import config
from .scoreboard import Scoreboard

class Drawer:
    def __init__(self):
        """
        Initializes the Drawer with default settings.
        """
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.scoreboard = Scoreboard()

        # Load colors from config or constants
        self.home_team_color = np.array(config.HOME_TEAM_COLOR)
        self.away_team_color = np.array(config.AWAY_TEAM_COLOR)
        self.referee_color = np.array(config.REFEREE_COLOR)
        self.default_ball_color = np.array(config.DEFAULT_BALL_COLOR)

    def draw_ellipse(self, frame, bounding_box, color, track_id=None):
        """
        Draws an ellipse around the player or referee.

        :param frame: The video frame to draw on.
        :param bounding_box: The bounding box of the object.
        :param color: The color of the ellipse.
        :param track_id: Optional track ID to draw next to the ellipse.
        :return: The frame with the ellipse drawn.
        """
        y2 = int(bounding_box[3])
        x_center, _ = geometry_utils.get_center_of_bounding_box(bounding_box)
        width = geometry_utils.get_bounding_box_width(bounding_box)

        color = tuple(map(int, color))  # Ensure color is a tuple of integers

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-60,
            endAngle=230,
            color=color,
            thickness=3,
            lineType=cv2.LINE_AA
        )

        if track_id is not None:
            text_y = y2 + 15
            # cv2.putText(frame, f"{track_id}", (x_center - 10, text_y), self.font, 0.5, (255, 255, 255), 2)

        return frame

    def draw_triangle(self, frame, bounding_box, color):
        """
        Draws a triangle above the ball or player to indicate possession.

        :param frame: The video frame to draw on.
        :param bounding_box: The bounding box of the object.
        :param color: The color of the triangle.
        :return: The frame with the triangle drawn.
        """
        if any(np.isnan(bounding_box)):  # Check for NaN values
            return frame

        # Ensure color is a tuple of integers
        color = tuple(map(int, color))

        y = int(bounding_box[1])
        x, _ = geometry_utils.get_center_of_bounding_box(bounding_box)

        triangle_points = np.array([
            [x, y],
            [x - 8, y - 15],
            [x + 8, y - 15]
        ])

        # Draw the filled triangle
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        # Draw the outline of the triangle
        cv2.drawContours(frame, [triangle_points], 0, color, 2)

        return frame


    def draw_team_ball_control(self, frame, frame_num, team_ball_control, home_team_time, away_team_time, home_team_possession, away_team_possession):
        """
        Draws the scoreboard showing team possession and time control.

        :param frame: The video frame to draw on.
        :param frame_num: The current frame number.
        :param team_ball_control: List of ball control states per frame.
        :param home_team_time: Total possession time for the home team.
        :param away_team_time: Total possession time for the away team.
        :param home_team_possession: Percentage possession for the home team.
        :param away_team_possession: Percentage possession for the away team.
        :return: The frame with the scoreboard drawn.
        """
        frame = self.scoreboard.draw(frame, home_team_possession, away_team_possession, home_team_time, away_team_time)
        return frame

    def draw_annotations(self, frame_num, frame, tracks, team_ball_control, home_team_time, away_team_time, home_team_possession, away_team_possession, home_team_color, away_team_color):
        """
        Draws all the annotations on the frame including players, referees, and ball.

        :param frame_num: The current frame number.
        :param frame: The video frame to draw on.
        :param tracks: Tracking data for all frames.
        :param team_ball_control: List of ball control states per frame.
        :param home_team_time: Total possession time for the home team.
        :param away_team_time: Total possession time for the away team.
        :param home_team_possession: Percentage possession for the home team.
        :param away_team_possession: Percentage possession for the away team.
        :param home_team_color: Color to use for home team annotations.
        :param away_team_color: Color to use for away team annotations.
        :return: The frame with all annotations drawn.
        """
        player_dict = tracks[constants.PLAYERS_KEY][frame_num]
        ball_dict = tracks[constants.BALL_KEY][frame_num]
        referee_dict = tracks[constants.REFEREES_KEY][frame_num]

        # Draw players
        for track_id, player in player_dict.items():
            color = player.get(constants.TEAM_COLOR_KEY, home_team_color)
            frame = self.draw_ellipse(frame, player[constants.BOUNDING_BOX_KEY], color, track_id)

            # Draw triangle above the player if they have the ball
            if player.get(constants.HAS_BALL_KEY, False):
                frame = self.draw_triangle(frame, player[constants.BOUNDING_BOX_KEY], color)

        # Draw referees
        for _, referee in referee_dict.items():
            frame = self.draw_ellipse(frame, referee[constants.BOUNDING_BOX_KEY], self.referee_color)

        # Draw ball
        for track_id, ball in ball_dict.items():
            if team_ball_control[frame_num] == constants.HOME_TEAM_ID:
                ball_color = home_team_color
            elif team_ball_control[frame_num] == constants.AWAY_TEAM_ID:
                ball_color = away_team_color
            else:
                ball_color = self.default_ball_color

            frame = self.draw_triangle(frame, ball[constants.BOUNDING_BOX_KEY], ball_color)
        
        # Draw team possession and time control
        frame = self.draw_team_ball_control(frame, frame_num, team_ball_control, home_team_time, away_team_time, home_team_possession, away_team_possession)

        return frame

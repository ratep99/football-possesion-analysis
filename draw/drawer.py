import cv2
import numpy as np
from utils import geometry_utils
import constants

class Drawer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.score_image = None

    def draw_ellipse(self, frame, bounding_box, color, track_id=None):
        y2 = int(bounding_box[3])
        x_center, _ = geometry_utils.get_center_of_bounding_box(bounding_box)
        width = geometry_utils.get_bounding_box_width(bounding_box)

        # Конверзија боје у tuple са целим бројевима
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
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

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

    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        # Draw a semi-transparent rectaggle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (50, 50), (500,300), (255,255,255), -1 )
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)


        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(70,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(70,140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame


    def draw_annotations(self, video_frames, tracks,team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks[constants.PLAYERS_KEY][frame_num]
            ball_dict = tracks[constants.BALL_KEY][frame_num]
            referee_dict = tracks[constants.REFEREES_KEY][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get(constants.TEAM_COLOR_KEY, (0, 0, 255))
                frame = self.draw_ellipse(frame, player[constants.BOUNDING_BOX_KEY], color, track_id)

                if player.get(constants.HAS_BALL_KEY, False):
                    frame = self.draw_triangle(frame, player[constants.BOUNDING_BOX_KEY], (0, 0, 255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee[constants.BOUNDING_BOX_KEY], (0, 255, 255))

            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball[constants.BOUNDING_BOX_KEY], (0, 255, 0))
            
            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)


            output_video_frames.append(frame)

        return output_video_frames

import cv2
import numpy as np
from utils import geometry_utils
import constants

class Drawer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.team1_logo = cv2.imread('data/dubocica.png', cv2.IMREAD_UNCHANGED)
        self.team2_logo = cv2.imread('data/mladostgat.png', cv2.IMREAD_UNCHANGED)

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

    def overlay_image(self, background, overlay, pos_x, pos_y):
        bh, bw = background.shape[:2]
        h, w = overlay.shape[:2]

        if pos_x + w > bw or pos_y + h > bh:
            raise ValueError("The overlay image is out of bounds of the background image.")

        if overlay.shape[2] == 4:  # check for alpha channel
            alpha = overlay[:, :, 3] / 255.0
            overlay = overlay[:, :, :3]
        else:
            alpha = np.ones((h, w))

        for c in range(3):
            background[pos_y:pos_y + h, pos_x:pos_x + w, c] = \
                (alpha * overlay[:, :, c] + (1 - alpha) * background[pos_y:pos_y + h, pos_x + w, c])

        return background

    def draw_team_ball_control(self, frame, frame_num, team_ball_control, team_1_time, team_2_time):
        overlay = frame.copy()
        alpha = 0.4

        # Dimenzije i pozicija pravougaonika
        pos_x, pos_y = 50, 50
        box_width, box_height = 500, 100
        padding = 10

        # Boje i fontovi
        bg_color = (255, 255, 255)  # Bela pozadina
        text_color = (0, 0, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2

        # Dimenzije i pozicija logotipa
        logo_height, logo_width = 60, 60
        team1_logo_resized = cv2.resize(self.team1_logo, (logo_width, logo_height), interpolation=cv2.INTER_AREA)
        team2_logo_resized = cv2.resize(self.team2_logo, (logo_width, logo_height), interpolation=cv2.INTER_AREA)

        # Crtanje pozadine za tekst
        cv2.rectangle(overlay, (pos_x - padding, pos_y - padding),
                      (pos_x + box_width + padding, pos_y + box_height + padding), bg_color, -1)

        # Dodavanje transparentnosti
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Priprema za tekst
        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        team_1_num_frames = np.sum(team_ball_control_till_frame == 1)
        team_2_num_frames = np.sum(team_ball_control_till_frame == 2)
        total_frames = team_1_num_frames + team_2_num_frames
        team_1_possession = (team_1_num_frames / total_frames) * 100 if total_frames != 0 else 0
        team_2_possession = (team_2_num_frames / total_frames) * 100 if total_frames != 0 else 0

        text1 = f"{team_1_possession:.0f}%"
        text2 = f"{team_2_possession:.0f}%"
        time1 = f"{int(team_1_time // 60):02d}:{int(team_1_time % 60):02d}"
        time2 = f"{int(team_2_time // 60):02d}:{int(team_2_time % 60):02d}"

        # Postavljanje logotipa na sliku
        frame = self.overlay_image(frame, team1_logo_resized, pos_x, pos_y + (box_height // 2 - logo_height // 2))
        frame = self.overlay_image(frame, team2_logo_resized, pos_x + box_width - logo_width, pos_y + (box_height // 2 - logo_height // 2))

        # Postavljanje teksta na sliku
        text_size1 = cv2.getTextSize(text1, font, font_scale, thickness)[0]
        text_x1 = pos_x + logo_width + (box_width // 2) - (text_size1[0] // 2) - 50
        text_y1 = pos_y + (box_height // 2) + (text_size1[1] // 2)
        text_size2 = cv2.getTextSize(text2, font, font_scale, thickness)[0]
        text_x2 = pos_x + logo_width + (box_width // 2) - (text_size2[0] // 2) + 50
        text_y2 = pos_y + (box_height // 2) + (text_size2[1] // 2)

        cv2.putText(frame, text1, (text_x1, text_y1), font, font_scale, text_color, thickness, cv2.LINE_AA)
        cv2.putText(frame, text2, (text_x2, text_y2), font, font_scale, text_color, thickness, cv2.LINE_AA)
        cv2.putText(frame, time1, (text_x1, text_y1 + 40), font, font_scale, text_color, thickness, cv2.LINE_AA)
        cv2.putText(frame, time2, (text_x2, text_y2 + 40), font, font_scale, text_color, thickness, cv2.LINE_AA)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control, team_1_time, team_2_time):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks[constants.PLAYERS_KEY][frame_num]
            ball_dict = tracks[constants.BALL_KEY][frame_num]
            referee_dict = tracks[constants.REFEREES_KEY][frame_num]

            # Crtanje igrača
            for track_id, player in player_dict.items():
                color = player.get(constants.TEAM_COLOR_KEY, (0, 0, 255))
                frame = self.draw_ellipse(frame, player[constants.BOUNDING_BOX_KEY], color, track_id)

                if player.get(constants.HAS_BALL_KEY, False):
                    frame = self.draw_triangle(frame, player[constants.BOUNDING_BOX_KEY], (0, 0, 255))

            # Crtanje sudije
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee[constants.BOUNDING_BOX_KEY], (0, 255, 255))

            # Crtanje lopte
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball[constants.BOUNDING_BOX_KEY], (0, 255, 0))
            
            # Crtanje kontrole lopte po timovima
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control, team_1_time, team_2_time)

            output_video_frames.append(frame)

        return output_video_frames

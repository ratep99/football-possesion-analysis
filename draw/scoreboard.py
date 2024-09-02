import cv2
import numpy as np

# Константе за хардкодиране вредности
FONT = cv2.FONT_HERSHEY_DUPLEX
BACKGROUND_IMAGE_PATH = 'data/pozadina.png'
FONT_SCALE = 0.8
FONT_COLOR = (255, 255, 255, 200)  # Бело са благом транспарентношћу
FONT_SHADOW_COLOR = (30, 30, 30, 255)  # Већа сенка
FONT_THICKNESS = 2
PROGRESS_BAR_HEIGHT = 15
PROGRESS_BAR_MARGIN = 50
TEXT_OFFSET_X = 60
TEXT_OFFSET_Y = 210
TIME_OFFSET_Y = 150
TIME_OFFSET_X = 80
HOME_TEAM_COLOR = (0, 0, 255)  # Црвена боја за домаћи тим
AWAY_TEAM_COLOR = (255, 255, 255)  # Бела боја за гостујући тим

class Scoreboard:
    def __init__(self):
        """
        Initializes the Scoreboard with default settings.
        """
        self.font = FONT
        self.background_image = cv2.imread(BACKGROUND_IMAGE_PATH, cv2.IMREAD_UNCHANGED)

        # Text and progress bar drawing parameters
        self.font_scale = FONT_SCALE
        self.font_color = FONT_COLOR
        self.font_shadow_color = FONT_SHADOW_COLOR
        self.font_thickness = FONT_THICKNESS
        self.progress_bar_height = PROGRESS_BAR_HEIGHT
        self.progress_bar_margin = PROGRESS_BAR_MARGIN
        self.text_offset_x = TEXT_OFFSET_X
        self.text_offset_y = TEXT_OFFSET_Y
        self.time_offset_y = TIME_OFFSET_Y
        self.time_offset_x = TIME_OFFSET_X
        self.home_team_color = HOME_TEAM_COLOR
        self.away_team_color = AWAY_TEAM_COLOR

    def draw(self, frame, home_team_possession, away_team_possession, home_team_time, away_team_time):
        """
        Draws the scoreboard with team possession and times.

        :param frame: The video frame to draw on.
        :param home_team_possession: Possession percentage of the home team.
        :param away_team_possession: Possession percentage of the away team.
        :param home_team_time: Total possession time for the home team.
        :param away_team_time: Total possession time for the away team.
        :return: The frame with the scoreboard drawn.
        """
        frame_height, frame_width = frame.shape[:2]

        # Calculate overlay dimensions
        overlay_width, overlay_height = self._calculate_overlay_dimensions(frame_width)

        # Position the scoreboard at the top, centered horizontally
        pos_x, pos_y = self._calculate_top_center_overlay_position(frame_width, overlay_width, frame_height, overlay_height)

        # Add background (overlay) to the frame with less transparency
        frame = self._overlay_image(frame, overlay_width, overlay_height, pos_x, pos_y, alpha=0.8)

        # Draw text for time and possession
        self._draw_time_and_possession(frame, home_team_possession, away_team_possession, home_team_time, away_team_time, overlay_width, pos_x, pos_y)

        # Draw progress bar
        self._draw_progress_bar(frame, home_team_possession, overlay_width, overlay_height, pos_x, pos_y)

        return frame

    def _calculate_overlay_dimensions(self, frame_width):
        """
        Calculates the dimensions for the scoreboard overlay based on the frame width.

        :param frame_width: Width of the video frame.
        :return: Width and height for the overlay.
        """
        resized_overlay_width = int(0.25 * frame_width)
        resized_overlay_height = int(self.background_image.shape[0] * resized_overlay_width / self.background_image.shape[1])
        
        if resized_overlay_width <= 0 or resized_overlay_height <= 0:
            raise ValueError("Calculated overlay dimensions are invalid.")

        return resized_overlay_width, resized_overlay_height

    def _calculate_top_center_overlay_position(self, frame_width, overlay_width, frame_height, overlay_height):
        """
        Calculates the position to place the overlay at the top center of the frame.

        :param frame_width: Width of the video frame.
        :param overlay_width: Width of the overlay.
        :param frame_height: Height of the video frame.
        :param overlay_height: Height of the overlay.
        :return: X and Y positions for the overlay.
        """
        pos_x = (frame_width - overlay_width) // 2
        pos_y = 10  # Slight margin from the top
        return pos_x, pos_y

    def _overlay_image(self, frame, overlay_width, overlay_height, pos_x, pos_y, alpha=0.8):
        """
        Overlays a semi-transparent image onto the frame.

        :param frame: The video frame to draw on.
        :param overlay_width: Width of the overlay.
        :param overlay_height: Height of the overlay.
        :param pos_x: X position to place the overlay.
        :param pos_y: Y position to place the overlay.
        :param alpha: Alpha transparency for the overlay.
        :return: The frame with the overlay.
        """
        overlay = cv2.resize(self.background_image, (overlay_width, overlay_height))
        return self._overlay_image_on_frame(frame, overlay, pos_x, pos_y, alpha)

    def _draw_time_and_possession(self, frame, home_team_possession, away_team_possession, home_team_time, away_team_time, overlay_width, pos_x, pos_y):
        """
        Draws the time and possession text on the scoreboard.

        :param frame: The video frame to draw on.
        :param home_team_possession: Possession percentage of the home team.
        :param away_team_possession: Possession percentage of the away team.
        :param home_team_time: Total possession time for the home team.
        :param away_team_time: Total possession time for the away team.
        :param overlay_width: Width of the overlay.
        :param pos_x: X position to draw text.
        :param pos_y: Y position to draw text.
        """
        time1 = f"{int(home_team_time // 60):02d}:{int(home_team_time % 60):02d}"
        time2 = f"{int(away_team_time // 60):02d}:{int(away_team_time % 60):02d}"
        text1 = f"{home_team_possession:.0f}%"
        text2 = f"{away_team_possession:.0f}%"

        # Calculate text width for positioning
        text_width_time, _ = cv2.getTextSize(time2, self.font, self.font_scale, self.font_thickness)[0]
        text_width_possession, _ = cv2.getTextSize(text2, self.font, self.font_scale, self.font_thickness)[0]

        # Draw time and possession percentages with offsets
        self._add_text_with_shadow(frame, time1, pos_x + self.time_offset_x, pos_y + self.time_offset_y, self.font_scale, self.font_color, self.font_thickness)
        self._add_text_with_shadow(frame, time2, pos_x + overlay_width - self.time_offset_x - text_width_time, pos_y + self.time_offset_y, self.font_scale, self.font_color, self.font_thickness)

        self._add_text_with_shadow(frame, text1, pos_x + self.text_offset_x, pos_y + self.text_offset_y - 5, self.font_scale, self.font_color, self.font_thickness)
        self._add_text_with_shadow(frame, text2, pos_x + overlay_width - self.text_offset_x - text_width_possession, pos_y + self.text_offset_y - 5, self.font_scale, self.font_color, self.font_thickness)

    def _draw_progress_bar(self, frame, home_team_possession, overlay_width, overlay_height, pos_x, pos_y):
        """
        Draws the progress bar showing possession percentage for both teams.

        :param frame: The video frame to draw on.
        :param home_team_possession: Possession percentage of the home team.
        :param overlay_width: Width of the overlay.
        :param overlay_height: Height of the overlay.
        :param pos_x: X position to draw the progress bar.
        :param pos_y: Y position to draw the progress bar.
        """
        bar_width = int(overlay_width * 0.8)
        home_team_bar_length = int((home_team_possession / 100) * bar_width)
        bar_x = pos_x + (overlay_width - bar_width) // 2
        bar_y = pos_y + overlay_height - self.progress_bar_margin

        # Draw home team progress bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + home_team_bar_length, bar_y + self.progress_bar_height), self.home_team_color, -1)

        # Draw away team progress bar with straight edge
        cv2.rectangle(frame, (bar_x + home_team_bar_length, bar_y), (bar_x + bar_width, bar_y + self.progress_bar_height), self.away_team_color, -1)

        # Draw white dividing line between progress bars
        cv2.line(frame, (bar_x + home_team_bar_length, bar_y), (bar_x + home_team_bar_length, bar_y + self.progress_bar_height), (255, 255, 255), 2)

    def _overlay_image_on_frame(self, background, overlay, pos_x, pos_y, alpha=1.0):
        """
        Overlays an image with transparency on top of another image.

        :param background: Background image/frame.
        :param overlay: Overlay image.
        :param pos_x: X position to place the overlay.
        :param pos_y: Y position to place the overlay.
        :param alpha: Alpha transparency of the overlay.
        :return: Combined image.
        """
        bh, bw = background.shape[:2]
        h, w = overlay.shape[:2]

        if pos_x + w > bw or pos_y + h > bh:
            raise ValueError("The overlay image is out of bounds of the background image.")

        if overlay.shape[2] == 4:  # Check for alpha channel
            overlay_alpha = overlay[:, :, 3] / 255.0  # Normalize alpha channel to [0, 1]
            overlay_rgb = overlay[:, :, :3]
        else:
            overlay_alpha = np.ones((h, w))
            overlay_rgb = overlay

        for c in range(3):
            background[pos_y:pos_y + h, pos_x:pos_x + w, c] = \
                (overlay_alpha * overlay_rgb[:, :, c] * alpha + 
                 (1 - overlay_alpha * alpha) * background[pos_y:pos_y + h, pos_x:pos_x + w, c])

        return background

    def _add_text_with_shadow(self, frame, text, x, y, font_scale, color, thickness):
        """
        Adds text with a shadow to the frame.

        :param frame: The video frame to draw on.
        :param text: The text to draw.
        :param x: X position for the text.
        :param y: Y position for the text.
        :param font_scale: Font scale (size).
        :param color: Font color.
        :param thickness: Font thickness.
        """
        # Draw shadow
        shadow_offset = 2  # Larger shadow
        cv2.putText(frame, text, (x + shadow_offset, y + shadow_offset), self.font, font_scale, self.font_shadow_color, thickness, cv2.LINE_AA)
        
        # Draw text with slight transparency
        overlay = frame.copy()
        cv2.putText(overlay, text, (x, y), self.font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

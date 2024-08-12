import cv2
import numpy as np

class Scoreboard:
    def __init__(self):
        # Основне поставке
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.background_image = cv2.imread('data/pozadina.png', cv2.IMREAD_UNCHANGED)

        # Параметри за цртање текста и прогрес бара
        self.font_scale = 0.5
        self.font_color = (255, 255, 255, 255)
        self.font_thickness = 1
        self.progress_bar_height = 15
        self.progress_bar_margin = 50
        self.text_offset_x = 100
        self.text_offset_y = 175
        self.time_offset_y = 110
        self.time_offset_x = 120

    def draw(self, frame, home_team_possession, away_team_possession, home_team_time, away_team_time):
        # Добијање димензија фрејма
        frame_height, frame_width = frame.shape[:2]

        # Рачунање димензија позадине (overlay)
        overlay_width, overlay_height = self.calculate_overlay_dimensions(frame_width)

        # Постављање позиције scoreboard-а на фрејму
        pos_x, pos_y = self.calculate_overlay_position(frame_width, overlay_width)

        # Додавање позадине (overlay) на фрејм
        frame = self.overlay_image(frame, overlay_width, overlay_height, pos_x, pos_y)

        # Постављање и исцртавање текста
        self.draw_time_and_possession(frame, home_team_possession, away_team_possession, home_team_time, away_team_time, overlay_width, pos_x, pos_y)

        # Цртање прогрес бара
        self.draw_progress_bar(frame, home_team_possession, overlay_width, overlay_height, pos_x, pos_y)

        return frame

    def calculate_overlay_dimensions(self, frame_width):
        resized_overlay_width = int(0.25 * frame_width)
        resized_overlay_height = int(self.background_image.shape[0] * resized_overlay_width / self.background_image.shape[1])
        
        if resized_overlay_width <= 0 or resized_overlay_height <= 0:
            raise ValueError("Calculated overlay dimensions are invalid.")

        return resized_overlay_width, resized_overlay_height

    def calculate_overlay_position(self, frame_width, overlay_width):
        pos_x = frame_width - overlay_width - 10
        pos_y = 10
        return pos_x, pos_y

    def overlay_image(self, frame, overlay_width, overlay_height, pos_x, pos_y):
        overlay = cv2.resize(self.background_image, (overlay_width, overlay_height))
        return self.overlay_image_on_frame(frame, overlay, pos_x, pos_y, alpha=0.8)

    def draw_time_and_possession(self, frame, home_team_possession, away_team_possession, home_team_time, away_team_time, overlay_width, pos_x, pos_y):
        # Текст за време и проценат поседа
        time1 = f"{int(home_team_time // 60):02d}:{int(home_team_time % 60):02d}"
        time2 = f"{int(away_team_time // 60):02d}:{int(away_team_time % 60):02d}"
        text1 = f"{home_team_possession:.0f}%"
        text2 = f"{away_team_possession:.0f}%"

        # Време поред грбова, ближе центру
        self.add_text_to_frame(frame, time1, pos_x + self.time_offset_x, pos_y + self.time_offset_y, self.font_scale, self.font_color, self.font_thickness)
        self.add_text_to_frame(frame, time2, pos_x + overlay_width - self.time_offset_x, pos_y + self.time_offset_y, self.font_scale, self.font_color, self.font_thickness)

        # Проценти испод грбова
        self.add_text_to_frame(frame, text1, pos_x + self.text_offset_x, pos_y + self.text_offset_y, self.font_scale, self.font_color, self.font_thickness)
        self.add_text_to_frame(frame, text2, pos_x + overlay_width - self.text_offset_x, pos_y + self.text_offset_y, self.font_scale, self.font_color, self.font_thickness)

    def draw_progress_bar(self, frame, home_team_possession, overlay_width, overlay_height, pos_x, pos_y):
        bar_width = int(overlay_width * 0.8)
        home_team_bar_length = int((home_team_possession / 100) * bar_width)
        bar_x = pos_x + (overlay_width - bar_width) // 2
        bar_y = pos_y + overlay_height - self.progress_bar_margin

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + home_team_bar_length, bar_y + self.progress_bar_height), (0, 0, 255), cv2.FILLED)
        cv2.rectangle(frame, (bar_x + home_team_bar_length, bar_y), (bar_x + bar_width, bar_y + self.progress_bar_height), (255, 0, 0), cv2.FILLED)

    def overlay_image_on_frame(self, background, overlay, pos_x, pos_y, alpha=1.0):
        bh, bw = background.shape[:2]
        h, w = overlay.shape[:2]

        if pos_x + w > bw or pos_y + h > bh:
            raise ValueError("The overlay image is out of bounds of the background image.")

        if overlay.shape[2] == 4:  # Провера за алфа канал
            overlay_alpha = overlay[:, :, 3] / 255.0  # Нормализација алфа канала у [0, 1]
            overlay_rgb = overlay[:, :, :3]
        else:
            overlay_alpha = np.ones((h, w))
            overlay_rgb = overlay

        for c in range(3):
            background[pos_y:pos_y + h, pos_x:pos_x + w, c] = \
                (overlay_alpha * overlay_rgb[:, :, c] * alpha + 
                 (1 - overlay_alpha * alpha) * background[pos_y:pos_y + h, pos_x:pos_x + w, c])

        return background

    def add_text_to_frame(self, frame, text, x, y, font_scale, color, thickness):
        text_size = cv2.getTextSize(text, self.font, font_scale, thickness)[0]
        text_x = x - (text_size[0] // 2)
        text_y = y + (text_size[1] // 2)
        cv2.putText(frame, text, (text_x, text_y), self.font, font_scale, color, thickness, cv2.LINE_AA)
        return frame

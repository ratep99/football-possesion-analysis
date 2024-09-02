import pandas as pd

def get_center_of_bounding_box(bounding_box):
    x1, y1, x2, y2 = bounding_box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_bounding_box_width(bounding_box):
    return bounding_box[2] - bounding_box[0]

def measure_distance(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

def measure_xy_distance(point1, point2):
    return point1[0] - point2[0], point1[1] - point2[1]

def get_foot_position(bounding_box):
    x1, y1, x2, y2 = bounding_box
    return int((x1 + x2) / 2), int(y2)

def get_triangle_from_bounding_box(bounding_box):
    x1, y1, x2, y2 = bounding_box
    center_x, center_y = get_center_of_bounding_box(bounding_box)
    top_point = (center_x, y1)
    bottom_left_point = (x1, y2)
    bottom_right_point = (x2, y2)
    return [top_point, bottom_left_point, bottom_right_point]

def interpolate_ball_positions(self, ball_positions):
        # Извлачење bounding box-ова лопте
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]

        # Креирање DataFrame-а за лакшу интерполацију
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Интерполација недостајућих вредности
        df_ball_positions = df_ball_positions.interpolate(method='linear', limit_direction='both')
        df_ball_positions = df_ball_positions.bfill()

        # Конвертовање назад у оригинални формат
        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions
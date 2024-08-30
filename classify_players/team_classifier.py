import cv2
import numpy as np
from sklearn.cluster import KMeans
import constants

class TeamClassifier:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.home_team = constants.HOME_TEAM_ID  # Crveni tim
        self.away_team = constants.AWAY_TEAM_ID  # Beli tim

        # Definišemo originalne boje u BGR formatu za crvenu i belu
        self.red_bgr = np.array([0, 0, 255])  # Crvena u BGR
        self.white_bgr = np.array([255, 255, 255])  # Bela u BGR
        self.color_change_threshold = 100  # Prag za promenu boje
        self.overlap_threshold = 0.2  # Prag za detekciju preklapanja (IoU > 0.2)
        self.initialization_frames = 5  # Broj frejmova za inicijalizaciju
        self.initialized = False  # Flag za inicijalizaciju

    def remove_green_pixels(self, image):
        # Definišemo opseg za zelene piksele u BGR formatu
        lower_green = np.array([50, 150, 50])
        upper_green = np.array([120, 190, 100])
        
        # Kreiramo masku za piksele koji nisu zeleni
        mask = cv2.inRange(image, lower_green, upper_green)
        mask_inv = cv2.bitwise_not(mask)
        
        # Primena maske da bi se uklonili zeleni pikseli
        image_no_green = cv2.bitwise_and(image, image, mask=mask_inv)
        
        return image_no_green

    def extract_dominant_color(self, image):
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1, random_state=42)
        kmeans.fit(image_2d)
        dominant_cluster = np.argmax(np.bincount(kmeans.labels_))
        dominant_color = kmeans.cluster_centers_[dominant_cluster]
        return dominant_color.astype(int)

    def get_player_color(self, frame, bounding_box):
        # Uzmi samo centralni deo bounding box-a za ekstrakciju boje
        x1, y1, x2, y2 = map(int, bounding_box)
        player_image = frame[y1:y2, x1:x2]
        height, width, _ = player_image.shape
        central_part = player_image[int(height / 4):int(3 * height / 4), int(width / 4):int(3 * width / 4)]

        # Ukloni zelene piksele direktno u BGR prostoru
        central_part_no_green = self.remove_green_pixels(central_part)
        
        # Pronađi dominantnu boju
        dominant_color = self.extract_dominant_color(central_part_no_green)

        return dominant_color

    def assign_team_colors(self, frame, player_detections):
        # Funkcija koja klasifikuje timske boje koristeći KMeans klasterisanje
        player_colors = []
        for _, player_detection in player_detections.items():
            bounding_box = player_detection[constants.BOUNDING_BOX_KEY]
            player_color = self.get_player_color(frame, bounding_box)
            player_colors.append(player_color)
        
        if len(player_colors) < 2:
            print("Not enough player colors for KMeans clustering")
            return

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1, random_state=42)
        kmeans.fit(player_colors)

        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

        # Proveri koji klaster odgovara crvenoj boji i koji beloj
        if np.linalg.norm(self.team_colors[1] - self.red_bgr) < np.linalg.norm(self.team_colors[2] - self.red_bgr):
            self.home_team_color = self.team_colors[1]
            self.away_team_color = self.team_colors[2]
        else:
            self.home_team_color = self.team_colors[2]
            self.away_team_color = self.team_colors[1]

    def initialize_team_colors(self, frames, player_detections_list):
        # Koristi nekoliko frejmova za inicijalizaciju boje tima
        all_player_colors = []
        for frame, player_detections in zip(frames, player_detections_list):
            for _, player_detection in player_detections.items():
                bounding_box = player_detection[constants.BOUNDING_BOX_KEY]
                player_color = self.get_player_color(frame, bounding_box)
                all_player_colors.append(player_color)

        if len(all_player_colors) < 2:
            print("Not enough player colors for KMeans clustering")
            return

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1, random_state=42)
        kmeans.fit(all_player_colors)

        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

        # Proveri koji klaster odgovara crvenoj boji i koji beloj
        if np.linalg.norm(self.team_colors[1] - self.red_bgr) < np.linalg.norm(self.team_colors[2] - self.red_bgr):
            self.home_team_color = self.team_colors[1]
            self.away_team_color = self.team_colors[2]
        else:
            self.home_team_color = self.team_colors[2]
            self.away_team_color = self.team_colors[1]

        self.initialized = True  # Obeležava da je inicijalizacija završena

    def calculate_iou(self, boxA, boxB):
        # Izračunavanje Intersection over Union (IoU) za dva bounding box-a
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def assign_teams_to_players(self, tracks, video):
        # Ako nismo inicijalizovali boje tima, koristimo početne frejmove
        if not self.initialized:
            frames_for_initialization = [video[i] for i in range(self.initialization_frames)]
            player_detections_list = [tracks[constants.PLAYERS_KEY][i] for i in range(self.initialization_frames)]
            self.initialize_team_colors(frames_for_initialization, player_detections_list)

        # Nastavljamo sa regularnom klasifikacijom nakon inicijalizacije
        for frame_num, player_track in enumerate(tracks[constants.PLAYERS_KEY]):
            for player_id, track in player_track.items():
                detected_color = self.get_player_color(video[frame_num], track[constants.BOUNDING_BOX_KEY])
                
                # Provera konzistentnosti boje
                if player_id in self.player_team_dict:
                    previous_color = self.player_team_dict[player_id]['color']
                    color_distance = np.linalg.norm(detected_color - previous_color)
                    if color_distance > self.color_change_threshold:
                        print(f"Player ID {player_id} color changed significantly, possible tracking error.")
                        detected_color = previous_color  # Zadrži prethodnu boju

                # Provera preklapanja sa drugim igračima
                for other_id, other_track in player_track.items():
                    if player_id != other_id:
                        iou = self.calculate_iou(track[constants.BOUNDING_BOX_KEY], other_track[constants.BOUNDING_BOX_KEY])
                        if iou > self.overlap_threshold:
                            print(f"Player ID {player_id} is overlapping with Player ID {other_id}, recalculating color.")
                            detected_color = self.get_player_color(video[frame_num], track[constants.BOUNDING_BOX_KEY])

                team = self.get_player_team(detected_color, player_id)
                tracks[constants.PLAYERS_KEY][frame_num][player_id][constants.TEAM_KEY] = team

                # Assign the appropriate team color based on the cluster assignment
                if team == self.home_team:
                    tracks[constants.PLAYERS_KEY][frame_num][player_id][constants.TEAM_COLOR_KEY] = self.red_bgr
                else:
                    tracks[constants.PLAYERS_KEY][frame_num][player_id][constants.TEAM_COLOR_KEY] = self.white_bgr


    def get_player_team(self, player_color, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]['team']

        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0] + 1
        self.player_team_dict[player_id] = {'team': team_id, 'color': player_color}
        return team_id
import cv2
import numpy as np
from sklearn.cluster import KMeans
import constants
import config

class TeamClassifier:
    def __init__(self):
        # Initialize team colors and player tracking dictionary
        self.team_colors = {}
        self.player_team_dict = {}
        self.home_team = constants.HOME_TEAM_ID
        self.away_team = constants.AWAY_TEAM_ID

        # Load team colors from config
        self.home_team_color = np.array(config.HOME_TEAM_COLOR)
        self.away_team_color = np.array(config.AWAY_TEAM_COLOR)

        # Configuration parameters
        self.color_change_threshold = config.COLOR_CHANGE_THRESHOLD
        self.overlap_threshold = config.OVERLAP_THRESHOLD
        self.initialization_frames = config.INITIALIZATION_FRAMES
        self.initialized = False

    def remove_green_pixels(self, image):
        """
        Removes green pixels from the image to avoid interference from the field.

        :param image: The input image from which to remove green pixels.
        :return: Image with green pixels removed.
        """
        lower_green = np.array([50, 150, 50])
        upper_green = np.array([120, 190, 100])
        mask = cv2.inRange(image, lower_green, upper_green)
        mask_inv = cv2.bitwise_not(mask)
        return cv2.bitwise_and(image, image, mask=mask_inv)

    def extract_dominant_color(self, image):
        """
        Extracts the dominant color from the image using KMeans clustering.

        :param image: The input image to extract the dominant color from.
        :return: The dominant color in BGR format.
        """
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1, random_state=42)
        kmeans.fit(image_2d)
        dominant_cluster = np.argmax(np.bincount(kmeans.labels_))
        return kmeans.cluster_centers_[dominant_cluster].astype(int)

    def get_player_color(self, frame, bounding_box):
        """
        Extracts the dominant color of a player from their bounding box in the frame.

        :param frame: The current video frame.
        :param bounding_box: The bounding box of the player.
        :return: The dominant color of the player in BGR format.
        """
        x1, y1, x2, y2 = map(int, bounding_box)
        player_image = frame[y1:y2, x1:x2]
        height, width, _ = player_image.shape
        central_part = player_image[int(height / 4):int(3 * height / 4), int(width / 4):int(3 * width / 4)]

        # Remove green pixels and find dominant color
        central_part_no_green = self.remove_green_pixels(central_part)
        return self.extract_dominant_color(central_part_no_green)

    def perform_kmeans_clustering(self, player_colors):
        """
        Performs KMeans clustering on player colors to determine team colors.

        :param player_colors: List of player colors to cluster.
        """
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1, random_state=42)
        kmeans.fit(player_colors)
        self.kmeans = kmeans
        self.team_colors[1], self.team_colors[2] = kmeans.cluster_centers_

        if np.linalg.norm(self.team_colors[1] - self.home_team_color) < np.linalg.norm(self.team_colors[2] - self.home_team_color):
            self.home_team_color, self.away_team_color = self.team_colors[1], self.team_colors[2]
        else:
            self.home_team_color, self.away_team_color = self.team_colors[2], self.team_colors[1]

    def initialize_team_colors(self, frames, player_detections_list):
        """
        Initializes team colors based on initial frames using KMeans clustering.

        :param frames: List of initial video frames.
        :param player_detections_list: List of player detections for the initial frames.
        """
        all_player_colors = [
            self.get_player_color(frame, player_detection[constants.BOUNDING_BOX_KEY])
            for frame, player_detections in zip(frames, player_detections_list)
            for player_detection in player_detections.values()
        ]

        if len(all_player_colors) < 2:
            print("Not enough player colors for KMeans clustering")
            return

        self.perform_kmeans_clustering(all_player_colors)
        self.initialized = True

    def calculate_iou(self, boxA, boxB):
        """
        Calculates the Intersection over Union (IoU) for two bounding boxes.

        :param boxA: First bounding box (x1, y1, x2, y2).
        :param boxB: Second bounding box (x1, y1, x2, y2).
        :return: IoU value.
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)

    def assign_teams_to_players(self, tracks, video):
        """
        Assigns teams to players based on their colors after initialization.

        :param tracks: Tracking data for all frames.
        :param video: List of video frames.
        """
        if not self.initialized:
            frames_for_initialization = [video[i] for i in range(self.initialization_frames)]
            player_detections_list = [tracks[constants.PLAYERS_KEY][i] for i in range(self.initialization_frames)]
            self.initialize_team_colors(frames_for_initialization, player_detections_list)

        for frame_num, player_track in enumerate(tracks[constants.PLAYERS_KEY]):
            for player_id, track in player_track.items():
                detected_color = self.get_player_color(video[frame_num], track[constants.BOUNDING_BOX_KEY])

                if player_id in self.player_team_dict:
                    previous_color = self.player_team_dict[player_id]['color']
                    color_distance = np.linalg.norm(detected_color - previous_color)
                    if color_distance > self.color_change_threshold:
                        print(f"Player ID {player_id} color changed significantly, possible tracking error.")
                        detected_color = previous_color

                for other_id, other_track in player_track.items():
                    if player_id != other_id:
                        iou = self.calculate_iou(track[constants.BOUNDING_BOX_KEY], other_track[constants.BOUNDING_BOX_KEY])
                        if iou > self.overlap_threshold:
                            print(f"Player ID {player_id} is overlapping with Player ID {other_id}, recalculating color.")
                            detected_color = self.get_player_color(video[frame_num], track[constants.BOUNDING_BOX_KEY])

                team = self.get_player_team(detected_color, player_id)
                tracks[constants.PLAYERS_KEY][frame_num][player_id][constants.TEAM_KEY] = team

                if team == self.home_team:
                    tracks[constants.PLAYERS_KEY][frame_num][player_id][constants.TEAM_COLOR_KEY] = self.home_team_color
                else:
                    tracks[constants.PLAYERS_KEY][frame_num][player_id][constants.TEAM_COLOR_KEY] = self.away_team_color

    def get_player_team(self, player_color, player_id):
        """
        Predicts the team of a player based on their color using KMeans clustering.

        :param player_color: The color of the player.
        :param player_id: The ID of the player.
        :return: The predicted team ID.
        """
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]['team']

        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0] + 1
        self.player_team_dict[player_id] = {'team': team_id, 'color': player_color}
        return team_id

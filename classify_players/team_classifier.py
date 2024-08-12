import cv2
import numpy as np
from sklearn.cluster import KMeans
import constants

class TeamClassifier:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.home_team = constants.HOME_TEAM_ID  # Crveni tim
        self.away_team = constants.AWAY_TEAM_ID  # Plavi tim

        # Define original RGB colors for red and blue
        self.red_rgb = np.array([0, 0, 255])  # Original red in BGR
        self.blue_rgb = np.array([255, 0, 0])  # Original blue in BGR

        # Define softer RGB colors for red and blue in BGR format
        self.soft_red_rgb = np.array([64, 34, 229])  # Softer red in BGR
        self.soft_blue_rgb = np.array([136, 51, 21])  # Softer blue in BGR

    def get_clustering_model(self, image):
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, frame, bounding_box):
        image = frame[int(bounding_box[1]):int(bounding_box[3]), int(bounding_box[0]):int(bounding_box[2])]
        top_half_image = image[0:int(image.shape[0] / 2), :]
        top_half_image_hsv = cv2.cvtColor(top_half_image, cv2.COLOR_BGR2HSV)
        kmeans = self.get_clustering_model(top_half_image_hsv)
        labels = kmeans.labels_
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster
        player_color_hsv = kmeans.cluster_centers_[player_cluster]
        player_color_hsv = np.uint8([[player_color_hsv]])
        player_color_rgb = cv2.cvtColor(player_color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        return player_color_rgb

    def assign_team_colors(self, frame, player_detections):
        player_colors = []
        for _, player_detection in player_detections.items():
            bounding_box = player_detection[constants.BOUNDING_BOX_KEY]
            player_color = self.get_player_color(frame, bounding_box)
            player_colors.append(player_color)
        
        if len(player_colors) < 2:
            print("Not enough player colors for KMeans clustering")
            return

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

        # Assign red and blue to home and away teams based on average color brightness
        if np.mean(self.team_colors[self.home_team]) < np.mean(self.team_colors[self.away_team]):
            self.home_team = constants.HOME_TEAM_ID
            self.away_team = constants.AWAY_TEAM_ID
        else:
            self.home_team = constants.AWAY_TEAM_ID
            self.away_team = constants.HOME_TEAM_ID

    def assign_teams_to_players(self, tracks, video):
        for frame_num, player_track in enumerate(tracks[constants.PLAYERS_KEY]):
            for player_id, track in player_track.items():
                team = self.get_player_team(video[frame_num], track[constants.BOUNDING_BOX_KEY], player_id)
                tracks[constants.PLAYERS_KEY][frame_num][player_id][constants.TEAM_KEY] = team

                # Assign the appropriate color (soft or original) based on the team
                if team == self.home_team:
                    tracks[constants.PLAYERS_KEY][frame_num][player_id][constants.TEAM_COLOR_KEY] = self.soft_red_rgb
                else:
                    tracks[constants.PLAYERS_KEY][frame_num][player_id][constants.TEAM_COLOR_KEY] = self.soft_blue_rgb

    def get_player_team(self, frame, player_bounding_box, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bounding_box)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0] + 1

        self.player_team_dict[player_id] = team_id
        return team_id

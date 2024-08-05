import cv2
from sklearn.cluster import KMeans
import numpy as np

class TeamClassifier:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
    
    def get_clustering_model(self, image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1, 3)

        # Preform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bounding_box):
        image = frame[int(bounding_box[1]):int(bounding_box[3]), int(bounding_box[0]):int(bounding_box[2])]

        top_half_image = image[0:int(image.shape[0] / 2), :]

        # Convert the image from RGB to HSV
        top_half_image_hsv = cv2.cvtColor(top_half_image, cv2.COLOR_BGR2HSV)

        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image_hsv)

        # Get the cluster labels for each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Get the player cluster
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color_hsv = kmeans.cluster_centers_[player_cluster]

        # Convert the player color from HSV to RGB
        player_color_hsv = np.uint8([[player_color_hsv]])
        player_color_rgb = cv2.cvtColor(player_color_hsv, cv2.COLOR_HSV2BGR)[0][0]

        return player_color_rgb

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player_detection in player_detections.items():
            bounding_box = player_detection["bounding_box"]
            player_color = self.get_player_color(frame, bounding_box)
            player_colors.append(player_color)
        
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bounding_box, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bounding_box)

        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1

        if player_id == 91:
            team_id = 1

        self.player_team_dict[player_id] = team_id

        return team_id

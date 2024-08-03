from sklearn.cluster import KMeans
import numpy as np

class TeamClassifier:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None  # Initialize kmeans attribute to None

    def get_clustering_model(self, image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1, 3)

        # Perform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bounding_box):
        print(f"Processing bounding_box: {bounding_box}")
        image = frame[int(bounding_box[1]):int(bounding_box[3]), int(bounding_box[0]):int(bounding_box[2])]

        if image.size == 0:
            print("Empty image for bounding_box:", bounding_box)
            return None

        top_half_image = image[0:int(image.shape[0] / 2), :]

        if top_half_image.size == 0:
            print("Empty top half image for bounding_box:", bounding_box)
            return None

        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels for each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Get the player cluster
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        print(f"Player color for bounding_box {bounding_box}: {player_color}")
        return player_color

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for player_id, player_detection in player_detections.items():
            bounding_box = player_detection["bounding_box"]
            print(f"Player ID {player_id} bounding_box: {bounding_box}")
            player_color = self.get_player_color(frame, bounding_box)
            if player_color is not None:
                player_colors.append(player_color)
            else:
                print(f"No player color found for player ID {player_id} bounding_box: {bounding_box}")
        
        if len(player_colors) == 0:
            raise ValueError("No player colors found. Check the player detections and bounding boxes.")

        player_colors = np.array(player_colors)

        print("Player colors array shape:", player_colors.shape)
        if player_colors.ndim == 1:
            player_colors = player_colors.reshape(1, -1)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]
        print(f"Assigned team colors: {self.team_colors}")

    def get_player_team(self, frame, player_bounding_box, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bounding_box)
        if player_color is None:
            print(f"No player color found for player ID {player_id} with bounding_box {player_bounding_box}")
            return None

        if self.kmeans is None:
            raise ValueError("KMeans model is not initialized.")

        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1

        self.player_team_dict[player_id] = team_id

        return team_id

    def classify_players(self, video_frames, tracks):
        print(f"Video frames: {len(video_frames)}, Tracks: {len(tracks['players'])}")

        for frame_num, player_track in enumerate(tracks['players']):
            if not player_track:
                print(f"No players detected in frame {frame_num}. Skipping frame.")
                continue
            
            if self.kmeans is None:
                self.assign_team_color(video_frames[frame_num], player_track)

            for player_id, track in player_track.items():
                team = self.get_player_team(video_frames[frame_num], track['bounding_box'], player_id)
                if team is not None:
                    tracks['players'][frame_num][player_id]['team'] = team
                    tracks['players'][frame_num][player_id]['team_color'] = self.team_colors[team]
                else:
                    print(f"Could not assign team for player ID {player_id} in frame {frame_num}")

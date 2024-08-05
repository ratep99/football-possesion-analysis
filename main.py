import cv2
import numpy as np
from utils import video_control_utils
from track_objects import Tracker
from draw import Drawer
from classify_players import TeamClassifier
from assign_ball import BallAssigner


def main():
    video = video_control_utils.read_video('data/proba.mp4')
    tracker = Tracker('model/yolov8mf.pt')
    tracks = tracker.get_object_tracks(video, read_from_cache=True, cache_path='cache/tracks_cache.pkl')
    
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    team_assigner = TeamClassifier()
    
    initial_frame_found = False
    for frame_num, player_detections in enumerate(tracks['players']):
        print(f"Frame {frame_num}: {len(player_detections)} players detected")
        if player_detections:
            print(f"Players detected in frame {frame_num}")
            team_assigner.assign_team_color(video[frame_num], player_detections)
            initial_frame_found = True
            break
    
    if not initial_frame_found:
        print("No players detected in any frame. Exiting.")
        return

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video[frame_num], track['bounding_box'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    player_assigner = BallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        if 1 in tracks['ball'][frame_num]:  # Ensure the ball track exists
            ball_bounding_box = tracks['ball'][frame_num][1]['bounding_box']
            assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bounding_box)

            if assigned_player != -1:
                tracks['players'][frame_num][assigned_player]['has_ball'] = True
                team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            print(f"Frame {frame_num}: Ball not detected.")
    
    team_ball_control = np.array(team_ball_control)
    print(f"Team Ball Control: {team_ball_control}")  # Proverite sadržaj team_ball_control

    drawer = Drawer()
    output_video = drawer.draw_annotations(video, tracks, team_ball_control)

    video_control_utils.save_video(output_video, 'data/output.avi')

if __name__ == '__main__':
    main()
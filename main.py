import cv2
from utils import video_control_utils
from object_tracking import Tracker

def main():
    video = video_control_utils.read_video('data/trackertest.mp4')
    # Initialize Tracker
    tracker = Tracker('model/yolov8mf.pt')

    tracks = tracker.get_object_tracks(video,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    # Get object positions 
    #tracker.add_position_to_tracks(tracks)
    
    output_video = tracker.draw_annotations(video,tracks)
    video_control_utils.save_video(output_video,'data/output.avi')


if __name__ == '__main__':
    main()
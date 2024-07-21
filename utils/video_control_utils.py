import cv2

def read_video(video_path):
    """
    Reads a video and returns a list of frames.

    Args:
        video_path (str): Path to the video file.

    Returns:
        list: List of frames as numpy array objects.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames

def save_video(video_frames, output_video_path, fps=24, codec='XVID'):
    """
    Saves a list of frames as a video file.

    Args:
        video_frames (list): List of frames as numpy array objects.
        output_video_path (str): Path where the video will be saved.
        fps (int): Frames per second for the output video. Default is 24.
        codec (str): Fourcc codec. Default is 'XVID'.
    """
    if not video_frames:
        raise ValueError("The list of frames is empty, cannot save video.")
    
    height, width = video_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    for frame in video_frames:
        out.write(frame)
    
    out.release()

# Example usage:
# frames = read_video('input_video.mp4')
# save_video(frames, 'output_video.avi')
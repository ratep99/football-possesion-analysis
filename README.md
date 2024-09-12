# Football Possession Analysis

This project analyzes football video footage to track players, determine ball possession, and calculate team possession statistics. It employs YOLO for object detection, ByteTrack for object tracking, and various utilities for video frame annotation.

## Demo

Watch the demo of the project in action:

![Football Possession Analysis Demo](./rfkzel.gif)



## Features

- **Object Detection**: Utilizes a YOLO model to detect players, referees, and the ball.
- **Object Tracking**: Tracks objects across frames with ByteTrack.
- **Team Classification**: Classifies players into home or away teams based on jersey colors.
- **Ball Possession Analysis**: Identifies which team is in possession of the ball at any given moment.
- **Video Annotation**: Annotates frames with positions of players, the ball, team possession, etc.
- **Caching**: Implements caching for detected and tracked data for faster subsequent runs.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ratep99/football-possession-analysis.git
   cd football-possession-analysis

2. **Install required dependencies:**

    Ensure you have Python 3.8 or later installed. Then, install the required Python packages:
   ```bash
   pip install -r requirements.txt
3. **Download the YOLO model:**
    Make sure you have the YOLO model file in the model/ directory. You can download a pre-trained model from the Ultralytics YOLO repository.

## Usage

To run the football analysis pipeline, execute the following command:

    python main.py

## Configuration

Configuration settings for the project are located in `config.py` and `constants.py`. You can adjust these settings according to your needs:

- **Paths**: Set paths for the input video, model, cache, and output video.
- **Team and Object Colors**: Set colors for teams, referees, and ball in BGR format.
- **Thresholds**: Set thresholds for color change detection, overlap detection, and initialization frames.
- **Ball Assigner Configuration**: Adjust thresholds for assigning ball possession.
- **Detector and Tracker Settings**: Adjust parameters for object detection and tracking.

## Dependencies

- **Python** 3.8 or later
- **OpenCV**: For video processing and drawing annotations.
- **NumPy**: For numerical operations.
- **pandas**: For handling and interpolating data.
- **Supervision**: For object detection and tracking.
- **Ultralytics YOLO**: YOLO model for object detection.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.



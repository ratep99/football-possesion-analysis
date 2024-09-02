# Football Possession Analysis

This project is designed to analyze football video footage to determine player tracking, ball possession, and team possession statistics. The project uses a YOLO model for object detection, ByteTrack for object tracking, and various utilities for processing and annotating video frames.



## Features

- **Object Detection**: Uses a YOLO model to detect players, referees, and the ball in video frames.
- **Object Tracking**: Tracks detected objects across frames using ByteTrack.
- **Team Classification**: Classifies players into home or away teams based on their jersey colors.
- **Ball Possession Analysis**: Determines which team is in possession of the ball at any given time.
- **Video Annotation**: Annotates video frames with player positions, ball positions, team possession, and other relevant data.
- **Caching**: Caches detected and tracked data for faster processing on subsequent runs.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/football-possession-analysis.git
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

## Project Structure

\```
football-possession-analysis/
│
├── data/                           # Directory for input video and output files
│   ├── rfkzel.mp4                  # Example input video file
│   └── output.avi                  # Output video with annotations
│
├── model/                          # Directory for the YOLO model file
│   └── 1280res100ep.pt             # Example YOLO model file
│
├── cache/                          # Directory for caching tracks data
│   └── tracks_cache.pkl            # Cached tracks data file
│
├── utils/                          # Utility functions
│   ├── video_control_utils.py      # Utilities for video processing
│   └── geometry_utils.py           # Utilities for geometric calculations
│
├── track_objects.py                # Object tracking module
├── detect_objects.py               # Object detection module
├── draw/                           # Drawing and annotation modules
│   ├── drawer.py                   # Frame annotation logic
│   └── scoreboard.py               # Scoreboard drawing logic
│
├── classify_players.py             # Player team classification module
├── assign_ball.py                  # Ball possession assignment module
├── calculate_possession.py         # Possession calculation module
├── football_analyzer.py            # Main analysis class
├── config.py                       # Configuration settings
├── constants.py                    # Constant values used throughout the project
├── requirements.txt                # List of Python dependencies
└── README.md                       # Project documentation (this file)
\```

## Dependencies

- **Python** 3.8 or later
- **OpenCV**: For video processing and drawing annotations.
- **NumPy**: For numerical operations.
- **pandas**: For handling and interpolating data.
- **Supervision**: For object detection and tracking.
- **Ultralytics YOLO**: YOLO model for object detection.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.



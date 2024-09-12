# Football Possession Analysis

This project analyzes football video footage to track players, determine ball possession, and calculate team possession statistics. The system uses YOLOv8 for object detection, ByteTrack for object tracking, and custom methods for team classification and ball possession analysis.

![Football Possession Analysis Demo](./rfkzel.gif)

## Process Overview

The project follows a structured pipeline:

1. **Model Training**: Fine-tuning the YOLOv8 model for player, referee, and ball detection.
2. **Object Detection**: Using YOLOv8 to detect objects in football videos.
3. **Object Tracking**: Using ByteTrack to ensure smooth player and ball tracking across video frames.
4. **Team Classification**: Classifying players into home or away teams based on jersey colors.
5. **Ball Possession Analysis**: Determining which player is in possession of the ball using proximity calculations.
6. **Result Visualization**: Annotating video frames with player positions, ball location, and possession information.

---

## Model Training

The object detection model was trained and fine-tuned on a dataset consisting of football match footage from the **Serbian SuperLiga** match between **Radnički Kragujevac** and **Železničar Pančevo**, played at the **Dubočica stadium in Leskovac**. The training process involved:

1. **Dataset Preparation**:
   - The dataset was prepared and annotated using **Roboflow**.
   - The resolution of the training data was set to **1920x1080** to capture high-quality details in the video frames.
   - Objects such as players, referees, and the ball were labeled manually.

2. **YOLOv8 Fine-tuning**:
   - A pre-trained YOLOv8 model was fine-tuned using the football match dataset to improve detection accuracy.
   - Training was conducted on **Kaggle** using **2x T4 GPUs** for faster processing and model convergence.
   - The model was optimized for detection in high-resolution video, ensuring precise player and ball localization.

3. **Tracking and Inference**:
   - After detection, **ByteTrack** was used to maintain the identity of players and the ball across frames, ensuring continuous tracking throughout the match.

---

## Features

- **Object Detection**: 
  - YOLOv8 is used to detect players, referees, and the ball in every frame of the video.
  
- **Object Tracking**:
  - ByteTrack algorithm maintains tracking of each detected player and ball over time, ensuring smooth continuity of positions across frames.

- **Team Classification**:
  - Players are classified into two teams (home and away) based on their jersey colors. 
  - K-Means clustering is used on detected player bounding boxes to assign each player to their respective team.

- **Ball Possession Analysis**:
  - A proximity-based algorithm determines which player is in possession of the ball at any given time.
  - The closest player to the ball is assigned possession, and temporal smoothing is applied to avoid frequent switching.

- **Video Annotation**:
  - The system generates output videos with visual annotations that show player positions, team affiliation, and ball possession.

---

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository**:
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



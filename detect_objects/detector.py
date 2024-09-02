from ultralytics import YOLO
import constants
import config
import supervision as sv
import logging

class ObjectDetector:
    def __init__(self):
        """
        Initializes the object detector with the YOLO model and configuration parameters.
        """
        self.model = YOLO(config.MODEL_PATH)
        self.confidence_threshold = config.DETECTOR_CONFIDENCE_THRESHOLD
        self.batch_size = config.DETECTOR_BATCH_SIZE
        self.imgsz = config.DETECTOR_IMAGE_SIZE  # Image size for YOLO predictions

    def detect_objects_on_frames(self, frames):
        """
        Detects objects in a list of frames using the YOLO model.

        :param frames: List of frames to perform detection on.
        :return: List of detections for each frame.
        """
        detections = []
        for i in range(0, len(frames), self.batch_size):
            # Perform batch prediction
            detections_batch = self.model.predict(
                frames[i:i + self.batch_size], 
                imgsz=self.imgsz,  # Use image size from config
                conf=self.confidence_threshold
            )

            # Process each detection batch
            for detection in detections_batch:
                detection_supervision = sv.Detections.from_ultralytics(detection)
                self.convert_goalkeeper_to_player(detection_supervision, detection.names)

            detections.extend(detections_batch)  # Efficient appending
        return detections

    def convert_goalkeeper_to_player(self, detection_supervision, class_names):
        """
        Converts detected goalkeeper class to player class.

        :param detection_supervision: Supervision detection object.
        :param class_names: List of class names.
        """
        for object_ind, class_id in enumerate(detection_supervision.class_id):
            if class_names[class_id] == constants.GOALKEEPER_KEY:
                detection_supervision.class_id[object_ind] = 2

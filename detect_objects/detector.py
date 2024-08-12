from ultralytics import YOLO
import constants
import config
import supervision as sv

class ObjectDetector:
    def __init__(self):
        self.model = YOLO(config.MODEL_PATH)
        self.confidence_threshold = config.DETECTOR_CONFIDENCE_THRESHOLD
        self.batch_size = config.DETECTOR_BATCH_SIZE

    def detect_objects_on_frames(self, frames):
        detections = []
        for i in range(0, len(frames), self.batch_size):
            detections_batch = self.model.predict(frames[i:i + self.batch_size], conf=self.confidence_threshold)
            for detection in detections_batch:
                detection_supervision = sv.Detections.from_ultralytics(detection)
                self.convert_goalkeeper_to_player(detection_supervision, detection.names)  # Примењујемо конверзију
            detections += detections_batch
        return detections

    def convert_goalkeeper_to_player(self, detection_supervision, class_names):

        for object_ind, class_id in enumerate(detection_supervision.class_id):
            if class_names[class_id] == constants.GOALKEEPER_KEY:
                detection_supervision.class_id[object_ind] = 2  # Претварање у класификацију за играча

# src/detection/yolo.py
from ultralytics import YOLO

class YOLOv8Detector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame)
        detections = []
        
        for det in results[0].boxes:
            x1, y1, x2, y2, conf, cls = det.xyxy.tolist()[0]
            if cls == 0:  # Asegúrate de que el índice de clase para patentes es 0
                detections.append((x1, y1, x2, y2, conf))
        
        return detections
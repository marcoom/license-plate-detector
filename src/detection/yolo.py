# src/detection/yolo.py

from ultralytics import YOLO
from config import MODEL_PATH, YOLO_THRESHOLD

def load_yolo_model():
    """
    Carga y retorna el modelo YOLO.
    """
    return YOLO(MODEL_PATH)

def get_detections(frame, model):
    """
    Obtiene las detecciones de objetos usando el modelo YOLO.
    """
    results = model(frame)[0]
    detections = []
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > YOLO_THRESHOLD:
            bbox = [x1, y1, x2 - x1, y2 - y1]
            detections.append((bbox, score, class_id))
    return detections

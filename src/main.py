# src/main.py

import sys
import os

# Agrega el directorio src al sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ultralytics import YOLO
import cv2
from detection.yolo import YOLOv8Detector
from ocr.ocr import EasyOCRRecognizer
from tracking.tracker import DeepSortTracker
from config import MODEL_PATH

def process_frame(frame, detector, recognizer, tracker):
    detections = detector.detect(frame)
    tracks = tracker.update(detections, frame)
    
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        
        plate_img = frame[y1:y2, x1:x2]
        plate_text = recognizer.recognize(plate_img)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f'ID: {track_id}, Plate: {plate_text}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return frame

def process_video(video_path):
    detector = YOLOv8Detector(MODEL_PATH)
    recognizer = EasyOCRRecognizer()
    tracker = DeepSortTracker()
    
    cap = cv2.VideoCapture(video_path)
    processed_frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(frame, detector, recognizer, tracker)
        processed_frames.append(frame)
    
    cap.release()
    return processed_frames

if __name__ == "__main__":
    # Ejemplo de cómo llamar a process_video
    video_path = "data/1_base.mp4"
    process_video(video_path)

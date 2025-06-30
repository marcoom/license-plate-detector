# This file is part of License Plate Detection System with YOLO and OCR.
# Copyright (C) 2025 Marco Mongi
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from ultralytics import YOLO
from config import MODEL_PATH, YOLO_THRESHOLD
import numpy as np

class YOLODetector:
    """YOLO Detector class.

    This class loads a YOLO model and uses it to detect objects in a given frame.
    """

    def __init__(self) -> None:
        """Initialize the YOLO detector.

        Load the YOLO model from the MODEL_PATH in the config file and set the threshold.
        """
        self.model = YOLO(MODEL_PATH)
        self.threshold = YOLO_THRESHOLD

    def detect(self, frame: np.ndarray) -> list[tuple[list[float], float, int]]:
        """Detect objects in a given frame using YOLO.

        Args:
            frame (np.ndarray): The frame to detect objects in.

        Returns:
            list[tuple[list[float], float, int]]: A list of tuples containing the bounding box coordinates, score and class id for each detection.
        """
        results = self.model.predict(frame, verbose=False)[0]
        detections = []
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > self.threshold:
                bbox = [x1, y1, x2 - x1, y2 - y1]
                detections.append((bbox, score, class_id))
        return detections
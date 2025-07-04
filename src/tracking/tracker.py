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

from typing import Any, Dict, List, Tuple
import config as cfg
from deep_sort_realtime.deepsort_tracker import DeepSort
from ui.interface import VideoInterface
from ocr.ocr import OCRReader
from ui.drawing import draw_plate_on_frame, draw_trajectory
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Tracker:
    """
    Encapsulates DeepSort tracking, OCR history, and trajectory management.
    """
    def __init__(self) -> None:
        """
        Initialize the DeepSort tracker and supporting data structures.
        """
        self.tracker = DeepSort(
            max_age=cfg.MAX_AGE,
            n_init=cfg.N_INIT,
            embedder="mobilenet",
            max_cosine_distance=cfg.COSINE_DISTANCE_THRESHOLD,
            nn_budget=None
        )
        logger.info("DeepSort tracker initialized with max_age=%d, n_init=%d", cfg.MAX_AGE, cfg.N_INIT)
        self.ocr_history: Dict[int, List[str]] = {}
        self.trajectories: Dict[int, List[Tuple[int, int]]] = {}

    def update_tracks(self, detections: List[Tuple[Any, float, int]], frame: np.ndarray) -> None:
        """
        Update tracks using the DeepSort tracker.

        Args:
            detections (List[Tuple[Any, float, int]]): List of detections (bbox, score, class_id).
            frame (np.ndarray): Current video frame.
        """
        logger.debug("Updating tracks with %d detections", len(detections))
        self.tracker.update_tracks(detections, frame=frame)

    def process_single_track(self, track: Any, frame: np.ndarray, ocr_reader: Any) -> None:
        """
        Process a single tracked object: apply OCR and update OCR history.

        Args:
            track (Any): Track object from DeepSort.
            frame (np.ndarray): Current video frame.
            ocr_reader (Any): OCR reader instance (e.g., EasyOCR reader).
        """
        track_id = track.track_id
        logger.debug("Processing single track: %s", track_id)
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        H, W, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 - x1 > 0 and y2 - y1 > 0:
            img_license = frame[y1:y2, x1:x2]
            ocr_results = ocr_reader.readtext(img_license)
            if track_id not in self.ocr_history:
                self.ocr_history[track_id] = []
                logger.debug("Created new OCR history for track %s", track_id)
            if ocr_results:
                _, text, confidence = ocr_results[0]
                logger.debug("OCR result for track %s: '%s' (confidence: %.2f)", track_id, text, confidence)
                if isinstance(text, str) and confidence >= cfg.OCR_CONFIDENCE_THRESHOLD:
                    self.ocr_history[track_id].append(text)
            most_common_plate = OCRReader.get_most_common_plate(self.ocr_history, track_id)
            draw_plate_on_frame(frame, most_common_plate, x1, y1, x2, y2, track_id)

    def update_trajectory(self, track: Any, frame: np.ndarray) -> None:
        """
        Update and draw trajectory for a single tracked object.

        Args:
            track (Any): Track object from DeepSort.
        """
        track_id = track.track_id
        ltrb = track.to_ltrb()
        center_x, center_y = int((ltrb[0] + ltrb[2]) / 2), int((ltrb[1] + ltrb[3]) / 2)
        if track_id not in self.trajectories:
            self.trajectories[track_id] = []
        self.trajectories[track_id].append((center_x, center_y))
        if len(self.trajectories[track_id]) > cfg.TRAJECTORY_LENGTH:
            self.trajectories[track_id] = self.trajectories[track_id][-cfg.TRAJECTORY_LENGTH:]
        if cfg.SHOW_TRAJECTORY:
            draw_trajectory(frame, self.trajectories[track_id])

    def process_detections(self, detections: List[Tuple[Any, float, int]], frame: np.ndarray, ocr_reader: Any) -> None:
        """
        Process detections for the current frame, update tracks, OCR history, and draw trajectories.

        Args:
            detections (List[Tuple[Any, float, int]]): List of detections (bbox, score, class_id).
            frame (np.ndarray): Current video frame.
            ocr_reader (Any): OCR reader instance (e.g., EasyOCR reader).
        """
        self.update_tracks(detections, frame)
        for track in self.tracker.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            self.process_single_track(track, frame, ocr_reader)
            self.update_trajectory(track, frame)
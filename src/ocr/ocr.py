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

from collections import Counter
from typing import Dict, List, Optional
import easyocr
import logging

logger = logging.getLogger(__name__)


class OCRReader:
    """
    Handles OCR reading and OCR history management for tracked objects.
    """

    def __init__(self, languages: Optional[List[str]] = ["en"]) -> None:
        """
        Initialize the OCR reader.

        Args:
            languages (Optional[List[str]]): List of language codes for OCR. Defaults to English.
        """
        self.reader = easyocr.Reader(languages)
        self.ocr_history: Dict[int, List[str]] = {}
        logger.info("OCRReader initialized with languages: %s", languages)

    def read_plate(
        self, img_license, track_id: int, confidence_threshold: float
    ) -> Optional[str]:
        """
        Perform OCR on the license plate image and update OCR history.

        Args:
            img_license: Cropped image of the license plate.
            track_id (int): ID of the tracked object.
            confidence_threshold (float): Minimum confidence for valid OCR result.

        Returns:
            str: Most common plate text for the track, or 'Plate' if not available.
        """
        ocr_results = self.reader.readtext(img_license)
        logger.debug("OCR read for track_id=%s, results=%s", track_id, ocr_results)
        if track_id not in self.ocr_history:
            self.ocr_history[track_id] = []
            logger.debug("Created new OCR history for track_id=%s", track_id)
        if ocr_results:
            _, text, confidence = ocr_results[0]
            logger.debug(
                "OCR result: text='%s', confidence=%.2f, threshold=%.2f",
                text,
                confidence,
                confidence_threshold,
            )
            if isinstance(text, str) and confidence >= confidence_threshold:
                self.ocr_history[track_id].append(text)
        return self.get_most_common_plate(self.ocr_history, track_id)

    @staticmethod
    def get_most_common_plate(ocr_history: Dict[int, List[str]], track_id: int) -> str:
        """
        Get the most frequent plate text for a given track ID.

        Args:
            ocr_history (Dict[int, List[str]]): OCR history by track ID.
            track_id (int): ID of the tracked object.
        Returns:
            str: Most common plate text, or 'Plate' if not available.
        """
        if ocr_history.get(track_id):
            return Counter(ocr_history[track_id]).most_common(1)[0][0]
        return "Plate"

    def get_history(self) -> Dict[int, List[str]]:
        """
        Get the full OCR history.

        Returns:
            Dict[int, List[str]]: OCR history by track ID.
        """
        return self.ocr_history

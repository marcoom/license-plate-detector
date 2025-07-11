from __future__ import annotations

from typing import List, Dict

import cv2
import numpy as np
import pytest


@pytest.fixture(scope="session")
def dummy_ocr_history() -> Dict[int, List[str]]:
    """Sample OCR history data for testing.

    Returns:
        dict: Maps track IDs to lists of detected license plate strings.
    """
    return {
        1: [
            "ABC 12345",
            "ABC 12345",
            "ABC 12345",
            "ABC 11111",  # A single typo that should be filtered out
            "ABC 12345",
            "ABC 12345",
        ]
    }


@pytest.fixture(scope="session")
def dummy_frame() -> np.ndarray:
    """Load a test image frame from disk.

    Returns:
        np.ndarray: Image data loaded from ./tests/assets/test_image.png.
    """
    image_path = "./tests/assets/test_image.png"
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not load test image: {image_path}")
    return frame


@pytest.fixture(scope="session")
def dummy_plate() -> np.ndarray:
    """Load a test license plate image from disk.

    Returns:
        np.ndarray: License plate image data loaded from tests/assets/test_plate.png.
    """
    plate_path = "./tests/assets/test_plate.png"
    plate = cv2.imread(plate_path)
    if plate is None:
        raise FileNotFoundError(f"Could not load test plate image: {plate_path}")
    return plate


@pytest.fixture(scope="session")
def dummy_video() -> cv2.VideoCapture:
    """Load a test video file from disk.

    Returns:
        cv2.VideoCapture: Video capture object for ./tests/assets/test_video.mp4.
    """
    video_path = "./tests/assets/test_video.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not load test video: {video_path}")
    return cap
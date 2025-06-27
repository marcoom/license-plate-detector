import cv2
from typing import List, Tuple, Optional
from config import SHOW_TRACKER_ID
import logging

logger = logging.getLogger(__name__)

def draw_plate_on_frame(
    frame: 'cv2.Mat',
    plate_text: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    track_id: Optional[int] = None
) -> None:
    """
    Draw the license plate text and bounding box on the frame. Adds a black background behind the text.
    If SHOW_TRACKER_ID is enabled, show the tracker ID.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    text = plate_text.upper()
    if SHOW_TRACKER_ID and track_id is not None:
        text = f'ID {track_id}: {text}'
        logger.debug("Drawing plate with tracker ID %s: %s", track_id, plate_text)
    else:
        logger.debug("Drawing plate: %s", plate_text)
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x, text_y = x1, y1 - 10
    box_x1, box_y1 = text_x, text_y - text_size[1]
    box_x2, box_y2 = text_x + text_size[0], text_y
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

def draw_trajectory(
    frame: 'cv2.Mat',
    trajectory: List[Tuple[int, int]]
) -> None:
    """
    Draw the trajectory of a tracked object on the frame. Connects the points representing the object's center.
    The current point is green, and subsequent points fade to white.
    """
    trajectory_length = len(trajectory)
    if trajectory_length == 0:
        logger.debug("No trajectory points to draw.")
    for i in range(1, trajectory_length):
        if trajectory[i - 1] is None or trajectory[i] is None:
            continue
        ratio = i / trajectory_length
        color = (
            int(255 * ratio),  # Red (increases towards white)
            255,               # Green (constant)
            int(255 * ratio)   # Blue (increases towards white)
        )
        cv2.line(frame, trajectory[i - 1], trajectory[i], color, 2)
        logger.debug("Drew trajectory segment from %s to %s", trajectory[i-1], trajectory[i])

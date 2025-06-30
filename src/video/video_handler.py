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

import cv2
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class VideoHandler:
    """
    Handles video capture, writing, and display operations.
    """
    def __init__(self, source: int | str, frame_width: int = None, frame_height: int = None, fps: float = None, output_path: Optional[str] = None) -> None:
        """
        Initialize the video handler.

        Args:
            source (int | str): Video source (webcam index or file path).
            frame_width (int, optional): Width of the video frames (required for writing).
            frame_height (int, optional): Height of the video frames (required for writing).
            fps (float, optional): Frames per second (required for writing).
            output_path (str, optional): Path to save the output video.
        """
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            logger.error("Failed to open video source: %s", source)
        else:
            logger.info("Opened video source: %s", source)
        self.writer = None
        self.output_path = output_path
        if output_path and frame_width and frame_height and fps:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            logger.info("Initialized video writer for: %s", output_path)

    def read_frame(self) -> tuple[bool, Optional[cv2.Mat]]:
        """
        Read a frame from the video source.

        Returns:
            tuple[bool, Optional[cv2.Mat]]: Success flag and the frame.
        """
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to read frame from video source.")
        return ret, frame

    def write_frame(self, frame: cv2.Mat) -> None:
        """
        Write a frame to the output video, if writer is enabled.

        Args:
            frame (cv2.Mat): The frame to write.
        """
        if self.writer:
            self.writer.write(frame)
            logger.debug("Wrote frame to output video.")

    def release(self) -> None:
        """
        Release the video capture and writer resources.
        """
        self.cap.release()
        logger.info("Released video capture resource.")
        if self.writer:
            self.writer.release()
            logger.info("Released video writer resource.")

    def is_opened(self) -> bool:
        """
        Check if the video source is opened.

        Returns:
            bool: True if opened, False otherwise.
        """
        return self.cap.isOpened()

    def show_frame(self, window_name: str, frame: cv2.Mat) -> None:
        """
        Display a frame in a window.

        Args:
            window_name (str): Name of the display window.
            frame (cv2.Mat): The frame to display.
        """
        cv2.imshow(window_name, frame)

    def wait_key(self, delay: int = 1) -> int:
        """
        Wait for a key event for a given delay.

        Args:
            delay (int): Delay in milliseconds.
        Returns:
            int: Key code.
        """
        return cv2.waitKey(delay)

    @staticmethod
    def destroy_all_windows() -> None:
        """
        Destroy all OpenCV windows.
        """
        cv2.destroyAllWindows()

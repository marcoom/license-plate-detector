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

import os
import cv2
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class VideoInterface:
    """
    Provides static methods for drawing overlays and setting up video output.
    """

    @staticmethod
    def setup_video_writer(
        input_video_path: str, frame_width: int, frame_height: int, fps: float
    ) -> Tuple["cv2.VideoWriter", str]:
        """
        Configure and return a VideoWriter for saving the processed video.

        Args:
            input_video_path (str): Path to the input video.
            frame_width (int): Width of the frames.
            frame_height (int): Height of the frames.
            fps (float): Frames per second.
        Returns:
            Tuple[cv2.VideoWriter, str]: The video writer and output video path.
        Raises:
            FileNotFoundError: If the input video path is invalid.
            ValueError: If the input video has an unsupported file format.
        """
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Input video not found: {input_video_path}")
        base_name, ext = os.path.splitext(input_video_path)
        if ext not in [
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".webm",
            ".flv",
            ".asf",
            ".wmv",
            ".mpg",
            ".mpeg",
            ".m4v",
        ]:
            raise ValueError(f"Unsupported file format: {ext}")
        output_video_path = f"{base_name}_processed{ext}"
        writer = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
            fps,
            (frame_width, frame_height),
        )
        logger.info("Video writer set up for output: %s", output_video_path)
        return writer, output_video_path

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
from typing import List, Tuple, Optional
from config import SHOW_TRACKER_ID
from .drawing import draw_plate_on_frame, draw_trajectory
import logging

logger = logging.getLogger(__name__)

class VideoInterface:
    """
    Provides static methods for drawing overlays and setting up video output.
    """


    @staticmethod
    def setup_video_writer(
        input_video_path: str,
        frame_width: int,
        frame_height: int,
        fps: float
    ) -> Tuple['cv2.VideoWriter', str]:
        """
        Configure and return a VideoWriter for saving the processed video.

        Args:
            input_video_path (str): Path to the input video.
            frame_width (int): Width of the frames.
            frame_height (int): Height of the frames.
            fps (float): Frames per second.
        Returns:
            Tuple[cv2.VideoWriter, str]: The video writer and output video path.
        """
        base_name, ext = os.path.splitext(input_video_path)
        output_video_path = f"{base_name}_processed{ext}"
        writer = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (frame_width, frame_height)
        )
        logger.info("Video writer set up for output: %s", output_video_path)
        return writer, output_video_path
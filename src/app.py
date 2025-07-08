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

from config import WEBCAM, INPUT_VIDEO, SAVE_TO_VIDEO, YOLO_THRESHOLD, SHOW_FPS
from utils.logger import LoggerConfigurator
from detection.yolo import YOLODetector
from tracking.tracker import Tracker
from ocr.ocr import OCRReader
from video.video_handler import VideoHandler
from ui.gradio_ui import build_interface
import threading
from ui.drawing import draw_fps_on_frame, FPSCounter
import cv2
from typing import Optional
import logging
import time

logger = logging.getLogger(__name__)

class LicensePlateDetectorApp:
    """
    Main application class for license plate detection, tracking, and OCR.
    """
    def __init__(self) -> None:
        self.yolo_detector = YOLODetector()
        self.tracker = Tracker()
        self.ocr_reader = OCRReader(languages=['en'])
        self.video_handler: Optional[VideoHandler] = None
        self.output_video_path: Optional[str] = None

    def setup_video(self, WEBCAM, INPUT_VIDEO) -> None:
        """
        Set up video capture and writer based on configuration.
        """
        logger.info("Setting up video source (webcam=%s, input=%s)", WEBCAM, INPUT_VIDEO)
        if WEBCAM:
            self.video_handler = VideoHandler(0)
            logger.info("Webcam video handler initialized.")
        else:
            cap = cv2.VideoCapture(INPUT_VIDEO)
            if not cap.isOpened():
                logger.error("Could not open video source: %s", INPUT_VIDEO)
                exit(1)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            output_path = None
            if SAVE_TO_VIDEO:
                output_path = INPUT_VIDEO.rsplit('.', 1)[0] + '_processed.mp4'
                self.output_video_path = output_path
                logger.info("Output video will be saved to: %s", output_path)
            self.video_handler = VideoHandler(INPUT_VIDEO, frame_width, frame_height, fps, output_path)
            logger.info("File video handler initialized.")
    
    def launch_gradio(self) -> None:
        """
        Launch the Gradio interface in a non-blocking way so that the main
        application loop can continue to run in parallel.
        """
        iface = build_interface()

        # Launch Gradio in a separate daemon thread so that the call does not
        # block the execution of the rest of the application. When the main
        # program exits, the daemon thread will automatically shut down.
        threading.Thread(
            target=iface.launch,
            kwargs={"share": False, "server_port": 7860},
            daemon=True,
        ).start()
        logger.info("Gradio interface initialized and running in background.")

    def run(self) -> None:
        """
        Run the main application loop.
        """
        logger.info("Initializing video handler and entering main loop.")
        self.setup_video()
        self.launch_gradio()
        if self.video_handler is None:
            logger.error("Video handler not initialized.")
            return
        frame_count = 0
        fps_counter = FPSCounter()
        while True:
            ret, frame = self.video_handler.read_frame()
            if not ret:
                logger.info("No more frames to read or error reading frame. Exiting loop.")
                break
            fps_counter.update()

            detections = self.yolo_detector.detect(frame)
            if detections:
                logger.debug("Detections found on frame %d: %s", frame_count, detections)
                self.tracker.process_detections(detections, frame, self.ocr_reader.reader)
            else:
                logger.debug("No detections on frame %d", frame_count)

            # Draw FPS if enabled (after all processing)
            if SHOW_FPS:
                draw_fps_on_frame(frame, fps_counter.get_fps())

            if self.video_handler.writer is not None:
                self.video_handler.write_frame(frame)

        self.video_handler.release()
        if self.output_video_path:
            logger.info("Processed video saved to: %s", self.output_video_path)

        logger.info("Exiting main loop.")

def main() -> None:
    """
    Entry point for the license plate detector app.
    """
    LoggerConfigurator().setup_logging()
    logger.info("Starting Gradio interface.")

    iface = build_interface()
    iface.launch(share=False, server_port=7860)

if __name__ == "__main__":
    main()

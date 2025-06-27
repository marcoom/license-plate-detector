# src/main.py
from config import WEBCAM, INPUT_VIDEO, SAVE_TO_VIDEO, CLOSE_WINDOW_KEY, YOLO_THRESHOLD, DISPLAY_VIDEO
from src.logging.logger import LoggerConfigurator
from detection.yolo import YOLODetector
from tracking.tracker import Tracker
from ocr.ocr import OCRReader
from video.video_handler import VideoHandler
import cv2
from typing import Optional
import logging

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
        self.close_key = ord(CLOSE_WINDOW_KEY.lower())

    def setup_video(self) -> None:
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

    def run(self) -> None:
        """
        Run the main application loop.
        """
        logger.info("Initializing video handler and entering main loop.")
        self.setup_video()
        if self.video_handler is None:
            logger.error("Video handler not initialized.")
            return
        frame_count = 0
        while True:
            ret, frame = self.video_handler.read_frame()
            if not ret:
                logger.info("No more frames to read or error reading frame. Exiting loop.")
                break
            detections = self.yolo_detector.detect(frame)
            if detections:
                logger.debug("Detections found on frame %d: %s", frame_count, detections)
                self.tracker.process_detections(detections, frame, self.ocr_reader.reader)
            else:
                logger.debug("No detections on frame %d", frame_count)
            if self.video_handler.writer is not None:
                self.video_handler.write_frame(frame)
            if DISPLAY_VIDEO:
                self.video_handler.show_frame("Processed Output", frame)
                key = self.video_handler.wait_key(1) & 0xFF
                if key == self.close_key:
                    logger.info("Exiting program: user pressed the close window key '%s' (code: %s)", chr(self.close_key), self.close_key)
                    break
        self.video_handler.release()
        if self.output_video_path:
            logger.info("Processed video saved to: %s", self.output_video_path)
        if DISPLAY_VIDEO:
            VideoHandler.destroy_all_windows()
        logger.info("Exiting main loop.")

def main() -> None:
    """
    Entry point for the license plate detector app.
    """
    LoggerConfigurator().setup_logging()
    logger.info("Starting License Plate Detector application.")
    app = LicensePlateDetectorApp()
    app.run()

if __name__ == "__main__":
    main()

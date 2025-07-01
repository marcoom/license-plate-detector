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


# Logging level configuration (e.g., 'INFO', 'DEBUG', 'WARNING')
LOG_LEVEL = "INFO"

# Video and processing
INPUT_VIDEO = './data/test_video_1.mp4'  # File to process
WEBCAM = False  # If True, use the webcam instead of the video file
DISPLAY_VIDEO = True  # If True, display the processed video in real-time
SAVE_TO_VIDEO = True  # If True, save the processed video when WEBCAM=False
SHOW_TRACKER_ID = True  # If True, show the tracker ID along with the plate
SHOW_TRAJECTORY = True  # If True, show the trajectory of the tracked objects
SHOW_FPS = True  # If True, display the FPS counter in the top-left corner
CLOSE_WINDOW_KEY = 'q'  # Key to close the video window

# Model and processing parameters
MODEL_PATH = './models/best_ncnn_model' # Model can be ncnn (./models/best_ncnn_model) or torch file (./models/car_plate.pt)
YOLO_THRESHOLD = 0.5  # Object detection threshold
OCR_CONFIDENCE_THRESHOLD = 0.02  # Minimum confidence for OCR results
MAX_AGE = 60  # Maximum number of frames an object can be absent before deleting the track
N_INIT = 10  # Number of detections needed to confirm an object
COSINE_DISTANCE_THRESHOLD = 0.4  # Cosine distance threshold for object re-identification
TRAJECTORY_LENGTH = 50  # Maximum length of the trajectory

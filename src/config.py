# src/config.py

# Logging level configuration (e.g., 'INFO', 'DEBUG', 'WARNING')
LOG_LEVEL = "INFO"

# Video and processing
INPUT_VIDEO = './data/3_multiple.mp4'  # File to process
WEBCAM = True  # If True, use the webcam instead of the video file
DISPLAY_VIDEO = True  # If True, display the processed video in real-time
SAVE_TO_VIDEO = True  # If True, save the processed video when WEBCAM=False
SHOW_TRACKER_ID = True  # If True, show the tracker ID along with the plate
SHOW_TRAJECTORY = True  # If True, show the trajectory of the tracked objects
CLOSE_WINDOW_KEY = 'q'  # Key to close the video window

# Model and processing parameters
MODEL_PATH = './models/car_plate.pt'
YOLO_THRESHOLD = 0.5  # Object detection threshold
OCR_CONFIDENCE_THRESHOLD = 0.02  # Minimum confidence for OCR results
MAX_AGE = 60  # Maximum number of frames an object can be absent before deleting the track
N_INIT = 10  # Number of detections needed to confirm an object
COSINE_DISTANCE_THRESHOLD = 0.4  # Cosine distance threshold for object re-identification
TRAJECTORY_LENGTH = 50  # Maximum length of the trajectory

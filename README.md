
# License Plate Detection System with YOLO and OCR

This project is a system for detecting vehicle license plates from video or real-time camera input. It uses a YOLO model for detecting license plates and EasyOCR for recognizing the characters. Additionally, it employs the DeepSORT library for object tracking throughout the video.

## Requirements

To run this project, you need to install the following requirements. You can install them using `pip`:

```bash
pip install -r requirements.txt
```

## Usage

### Configuration

Before running the project, you can configure the system behavior by editing the variables in `src/config.py`:

- `INPUT_VIDEO`: The path to the video file to process (e.g., `'./data/3_multiple.mp4'`). This will be used when `WEBCAM` is set to `False`.
- `WEBCAM`: Set to `True` to use the webcam as the video source instead of a video file. Set to `False` to process the specified `INPUT_VIDEO`.
- `DISPLAY_VIDEO`: Set to `True` to display the processed video in real-time. If `False`, the video will not be shown on the screen.
- `SAVE_TO_VIDEO`: Set to `True` to save the processed video when using a video file as input (`WEBCAM=False`).
- `SHOW_TRACKER_ID`: Set to `True` to display the tracker ID along with the detected license plate in the video.
- `SHOW_TRAJECTORY`: Set to `True` to show the trajectory of the tracked objects on the video.
- `CLOSE_WINDOW_KEY`: The key that will be used to close the video window (default is `'q'`).

- `MODEL_PATH`: Path to the YOLO model file (e.g., `'./models/car_plate.pt'`).
- `YOLO_THRESHOLD`: Confidence threshold for object detection. Only detections with confidence above this value will be considered (default is `0.5`).
- `OCR_CONFIDENCE_THRESHOLD`: Minimum confidence threshold for OCR results. Only OCR results with confidence above this value will be considered (default is `0.02`).
- `MAX_AGE`: The maximum number of frames an object can be absent before its track is deleted (default is `60`).
- `N_INIT`: The number of detections needed to confirm an object track (default is `10`).
- `COSINE_DISTANCE_THRESHOLD`: Threshold for cosine distance used in object re-identification (default is `0.4`).
- `TRAJECTORY_LENGTH`: The maximum length of the trajectory line to be drawn for each tracked object (default is `50`).

### Execution

Once the environment is set up and the parameters in `src/config.py` are adjusted, you can run the project as follows:

```bash
python src/main.py
```

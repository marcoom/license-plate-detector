
# License Plate Detection System with YOLO and OCR

This project is a system for detecting vehicle license plates from video or real-time camera input. It uses a YOLO model for detecting license plates and EasyOCR for recognizing the characters. Additionally, it employs the DeepSORT library for object tracking throughout the video.

![Detection Example](media/animation.gif)

## Features

- Detect license plates in video files or live webcam
- Track multiple vehicles and their plates across frames
- Recognize plate text using OCR with confidence filtering
- Visualize tracker IDs, object trajectories, and FPS
- Save processed video with overlays
- Interactive Gradio web UI for easy use and configuration
- Clean Python package structure with proper dependency management

---

## Project Structure

```
license-plate-detector/
├── src/                     # Source code package
│   ├── __init__.py         # Package definition
│   ├── app.py              # Main application entry point
│   ├── config.py           # Configuration settings
│   ├── detection/          # Object detection components
│   │   ├── __init__.py
│   │   └── yolo.py         # YOLO detector implementation
│   ├── ocr/                # Optical Character Recognition
│   │   ├── __init__.py
│   │   └── ocr.py          # OCR processing logic
│   ├── tracking/           # Object tracking
│   │   ├── __init__.py
│   │   └── tracker.py      # DeepSORT tracker implementation
│   ├── ui/                 # User interface components
│   │   ├── __init__.py
│   │   ├── gradio_ui.py    # Web-based UI using Gradio
│   │   ├── drawing.py      # Visualization and drawing utilities
│   │   └── interface.py    # Interface helpers
│   ├── video/              # Video processing
│   │   ├── __init__.py
│   │   └── video_handler.py # Video capture and processing
│   └── utils/              # Utility functions
│       ├── __init__.py
│       └── logger.py       # Logging configuration
├── data/                   # Sample video files for testing
├── docs/                   # Documentation
│   └── source/             # Source files for Sphinx documentation
├── media/                  # Media assets (screenshots, diagrams)
├── models/                 # Pre-trained model files
├── tests/                  # Test files
├── Dockerfile              # Container configuration
├── LICENSE                 # License file
├── Makefile                # Development tasks automation
├── NOTICE                  # Copyright notices
├── README.md               # This file
├── requirements.txt        # Runtime dependencies
├── requirements-dev.txt    # Development dependencies
└── setup.py                # Package installation script
```

---

## Installation

### Prerequisites
- Python 3.11 or higher
- pip (Python package manager)
- make (for development tasks)

### Minimal installation
The following commands creates a virtual environment and installs the package dependencies for running the application.
```bash
make install
```

### Development installation
Run the following command, that installs the required packages for running the application and also for development tasks (such as building documentation, executing tests or building packages)
```bash
make install-dev
```

---

## Configuration

Edit `src/config.py` to customize behavior:

**Video and Processing**
- `INPUT_VIDEO`: Path to video file to process (`'./data/test_video_1.mp4'` by default)
- `WEBCAM`: Use webcam (`True`) or video file (`False`)
- `SAVE_TO_VIDEO`: Save processed video when using a file (`False` by default)
- `SHOW_TRACKER_ID`: Show tracker ID with plate (`True` by default)
- `SHOW_FPS`: Show FPS counter in the top-left corner (`True` by default)
- `SHOW_TRAJECTORY`: Show tracked object trajectories (`True` by default)
- `TRAJECTORY_LENGTH`: Max length of trajectory line (`50` by default)

**Model and Processing Parameters**
- `MODEL_PATH`: Path to YOLO model (`'./models/best_ncnn_model'` by default, can be NCNN or PyTorch `.pt` format)
- `YOLO_THRESHOLD`: Object detection confidence threshold (default: `0.5`)
- `OCR_CONFIDENCE_THRESHOLD`: Minimum confidence for OCR results (default: `0.02`)
- `COSINE_DISTANCE_THRESHOLD`: Cosine distance threshold for object re-identification (default: `0.4`)
- `N_INIT`: Number of detections needed to confirm an object (default: `10`)
- `MAX_AGE`: Maximum number of frames an object can be absent before deleting the track (default: `60`)

**Logging**
- `LOG_LEVEL`: Logging level configuration (e.g., `'INFO'`, `'DEBUG'`, `'WARNING'`, default: `'INFO'`)

**Runtime Control**
- `STOP_REQUESTED`: Set by UI to interrupt processing (default: `False`)

---

## Usage
To run the app, execute:

```bash
make run
```

- By default, the Gradio web UI will launch at http://localhost:7860
- You can select video file or webcam, and adjust all settings in the UI

> **Note:** You can also run the application using Docker. This is particularly useful for ensuring consistent behavior across different environments. See the [Docker](#docker) section for detailed instructions.

---

## User Interface

The Gradio UI provides an easy way to upload/select video, switch between webcam and file, and adjust all detection/tracking parameters interactively.

![Demo video](media/demo.gif)

---

## Try it Online

You can try a demo of this application on Hugging Face Spaces:

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/marcoom/license-plate-detector)

> **Note:** 
> - The online demo runs on CPU, so performance may be slower than running locally with GPU acceleration.
> - The Hugging Face Space is hosted on an external platform and the link may become unavailable in the future.
> - The online demo only supports video file uploads. When running locally, the app automatically checks for webcam availability and enables the webcam input option if one is detected.

---

## Video resources
The video resources used in this project are:

| Asset | Details |
|-------|---------|
| test_video_1.mp4 | **Author:** Manuel Mata Gallego  <br> **Source:** [Vecteezy](https://es.vecteezy.com/video/36990287-trafico-carros-paso-en-la-carretera-con-asfalto-con-grietas-visto-desde-encima)  <br> **License:** Vecteezy Free License – attribution required.  <br> Stock footage by Vecteezy.com |
| test_video_2.mp4 | **Author:** Mike Bird  <br> **Source:** [Pexels](https://www.pexels.com/video/traffic-flow-in-the-highway-2103099/)  <br> **License:** Pexels License – attribution is not mandatory but appreciated.  <br> Stock footage by Mike Bird via Pexels.com |

---

## Detection Model
The detection model used is based on YOLOv8 and is trained on a custom dataset of license plates. The model and its training code is available at https://github.com/marcoom/yolo-licence-plate-training

---

## Docker

This project includes Docker support for easy deployment. The following commands are available:

### Build the Docker Image

```bash
make docker-build
```

This will build a Docker image named `license-plate-detector`.

### Run the Docker Container

```bash
make docker-run
```

This will run the application in a Docker container with:
- Webcam access (`/dev/video0`)
- Port 7860 exposed for the Gradio web interface
- Automatic cleanup when the container stops (`--rm` flag)

### Remove the Docker Image

```bash
make docker-remove
```

This will remove the `license-plate-detector` Docker image.

### Docker Hub

The Docker image is available on Docker Hub at https://hub.docker.com/r/marcoom/license-plate-detector

To pull the image, run:

```bash
docker pull marcoom/license-plate-detector:1.0.0
```

To run the container, run:

```bash
docker run -it --rm --device /dev/video0:/dev/video0 -p 7860:7860 marcoom/license-plate-detector:1.0.0
```

---

## Developer Tools

### Documentation

HTML and PDF documentation is generated with Sphinx. To build docs:

```bash
make docs
```
The documentation is generated inside the /docs/build directory.

More Make commands are available to generate only html (make docs-html), pdf (make docs-pdf) or remove documentation building files (make docs-clean).


### Building Distributions
To build source and wheel distributions for the package:

```bash
# Build source and wheel distributions
make dist

# Clean build artifacts
make clean
```

This will create both a source distribution (`.tar.gz`) and a wheel (`.whl`) in the `dist/` directory. The distributions will include all necessary files specified in `setup.py`.

To install the built wheel:

```bash
pip install dist/*.whl
```

Or install directly from the source distribution:

```bash
pip install dist/*.tar.gz
```

### Testing

To run tests:

```bash
make test
```

To run tests with coverage:

```bash
make test-coverage
```

### Code Quality

To check code quality:

```bash
make lint
```

To format code:

```bash
make format
```

To check types:

```bash
make type-check
```

---

### Help
To see all available make commands and their descriptions, run:

```bash
make help
```

---

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

By using, modifying, or distributing this software, you agree to the terms of the AGPL-3.0 license. If you deploy this software as a service (e.g., over a network), you must make the complete source code of the running version available to users. See the LICENSE file for details.

Third-party components are used under their respective licenses. See the NOTICE file for attributions and more information.

---

## Attribution

Third-party components are used under their respective licenses. See the `NOTICE` file for more information.

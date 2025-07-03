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

from __future__ import annotations  # Allows forward references in type hints

import gradio as gr
import cv2
import numpy as np
from typing import Generator, Optional, Dict, Any

import config as cfg  # Use runtime config values throughout the session
from ui.drawing import draw_fps_on_frame, FPSCounter

def _prepare_app(video_path: Optional[str]) -> 'LicensePlateDetectorApp':
    """Helper that prepares a new ``LicensePlateDetectorApp`` instance.

    The function dynamically adjusts the global configuration flags so that
    ``LicensePlateDetectorApp`` initialises the correct ``VideoHandler`` for
    the provided *video_path* (``None`` indicates that the webcam should be
    used).
    """
    import config as cfg  # Imported here to avoid circular imports
    from app import LicensePlateDetectorApp  # Local import to avoid circular dependency

    if video_path is None:
        cfg.WEBCAM = True
        #cfg.INPUT_VIDEO = ""
    else:
        cfg.WEBCAM = False
        cfg.INPUT_VIDEO = video_path

    return LicensePlateDetectorApp()


def process_video(video: Optional[str | Dict[str, Any]]) -> Generator[np.ndarray, None, None]:
    """Gradio callback that processes a video (or webcam) and **streams** the
    processed frames to the *Image* component.

    Parameters
    ----------
    video
        A string path to a video file provided by the *gr.Video* component or
        ``None`` when the *Run* button is pressed with no video selected
        (webcam mode).

    Yields
    ------
    numpy.ndarray
        Frames in **RGB** colour space ready to be displayed by Gradio.
    """
    # ``gr.File`` (with ``type='filepath'``) returns the selected file path as
    # a plain string. Older configurations may return a dict with a temporary
    # file path under the ``"name"`` key. Handle both.
    if isinstance(video, dict):
        video_path = video.get("name")
    elif isinstance(video, str):
        video_path = video
    else:
        # No file selected – fall back to default behaviour configured in cfg.
        # If the user wants the webcam, they should set ``cfg.WEBCAM = True`` beforehand.
        video_path = None if cfg.WEBCAM else cfg.INPUT_VIDEO

    if video_path:
        cfg.INPUT_VIDEO = video_path
        cfg.WEBCAM = False

    app = _prepare_app(video_path)
    app.setup_video()

    if app.video_handler is None:
        yield np.zeros((100, 100, 3), dtype=np.uint8)  # Informative blank frame
        return

    fps_counter = FPSCounter()

    while True:
        ret, frame = app.video_handler.read_frame()
        if not ret:
            break

        # Main processing pipeline (detection, tracking, OCR) -----------------
        detections = app.yolo_detector.detect(frame)
        if detections:
            app.tracker.process_detections(detections, frame, app.ocr_reader.reader)

        if cfg.SHOW_FPS:
            fps_counter.update()
            draw_fps_on_frame(frame, fps_counter.get_fps())

        # --------------------------------------------------------------------

        # Convert BGR (OpenCV) -> RGB for Gradio.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame_rgb

    app.video_handler.release()

    # When finished, yield one more frame so that Gradio knows the stream has
    # ended (optional – could also simply return)
    return


def build_interface():
    # Ensure default behaviour: unless the user explicitly toggles webcam elsewhere,
    # we process the demo video bundled with the project.
    cfg.WEBCAM = False

    with gr.Blocks() as demo:
        # ----------------------- CALLBACKS -----------------------------
        def update_cfg(attr: str):
            """Returns a callback that updates *attr* of ``config`` with the incoming value."""
            def _inner(value):
                setattr(cfg, attr, value)
                return None  # No component updates required
            return _inner

        def toggle_trajectory_slider(show: bool):
            """Show/hide and enable/disable the trajectory length slider when the checkbox changes."""
            # Update runtime configuration so that the rest of the app sees the change
            cfg.SHOW_TRAJECTORY = show
            # `visible` completely hides the component, `interactive` controls whether the
            # slider can be dragged when it *is* visible.
            return gr.update(visible=show, interactive=show)

        def on_model_type_change(model_type: str):
            # Map UI selection to model path stored in cfg
            if model_type == "PyTorch":
                cfg.MODEL_PATH = "./models/car_plate.pt"
            else:
                cfg.MODEL_PATH = "./models/best_ncnn_model"
            return None

        # -----------------------------------------------------------------

        output_image = gr.Image(show_label=False, interactive=False, visible=False)
        with gr.Row(equal_height=True, min_height=250):
            with gr.Column(scale=10):
                input_source = gr.Radio(
                    choices=["Video File", "Webcam"],
                    label="Input Source",
                    value="Video File"
                )
            with gr.Column(scale=1):
                input_video = gr.File(file_types=["video"], visible=True, type="filepath")

                def _on_source_change(src):
                    webcam = (src == "Webcam")
                    cfg.WEBCAM = webcam
                    # Show uploader only for video file
                    return gr.update(visible=not webcam)

                input_source.change(_on_source_change, input_source, input_video)

                def _on_file_selected(file):
                    if file is None:
                        return None
                    # Depending on ``type`` param, file may be a str (filepath) or dict
                    path = file if isinstance(file, str) else file.get("name") if isinstance(file, dict) else None
                    if path:
                        cfg.WEBCAM = False
                        cfg.INPUT_VIDEO = path
                    return None

                input_video.change(_on_file_selected, input_video, None)
            with gr.Column(scale=10):
                stop_button = gr.Button("Stop", elem_id="stop_button", variant="stop", visible=False)
            with gr.Column(scale=10):
                run_button = gr.Button("Run", elem_id="run_button", variant="primary")
        with gr.Accordion("Settings 🛠️", open=False): # When running, settings can't be modified (should be grayed out)
            with gr.Tab("System"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=300):
                        gr.Markdown(value="Detection")
                        slider_model_type = gr.Dropdown(
                            choices=['PyTorch', 'NCNN'],
                            label="Model Type",
                            value=("PyTorch" if cfg.MODEL_PATH.endswith(".pt") else "NCNN"),
                            multiselect=False,
                            info="Select the backend for the YOLO model"
                        )
                        slider_detection_threshold = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=cfg.YOLO_THRESHOLD,
                            step=0.01,
                            show_reset_button=True,
                            label="Confidence threshold",
                            info="Minimum confidence required for object detection"
                        )
                        gr.Markdown(value="Optical Character Recognition (OCR)")
                        slider_ocr_threshold = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=cfg.OCR_CONFIDENCE_THRESHOLD,
                            step=0.01,
                            label="Confidence threshold",
                            show_reset_button=True,
                            info="Minimum confidence required for OCR recognition"
                        )
                    with gr.Column(scale=1, min_width=300):
                        gr.Markdown(value="Tracking")
                        slider_tracking_cosine_distance_threshold = gr.Slider(
                            label="Cosine Distance Threshold",
                            value=cfg.COSINE_DISTANCE_THRESHOLD,
                            maximum=1.0,
                            info="Threshold for object re-identification in tracking"
                        )
                        slider_tracking_min_num_detections = gr.Slider(
                            minimum=1.0,
                            maximum=20.0,
                            value=cfg.N_INIT,
                            step=1.0,
                            label="Min. num. detections",
                            show_reset_button=True,
                            info="Detections required to confirm a new track"
                        )
                        slider_tracking_max_frames = gr.Slider(
                            minimum=1.0,
                            maximum=3600.0,
                            value=cfg.MAX_AGE,
                            step=1.0,
                            label="Max. frames",
                            info="Frames to keep a track without new detections"
                        )
            with gr.Tab("Visualization"):
                checkbox_show_track_id = gr.Checkbox(
                    value=cfg.SHOW_TRACKER_ID,
                    label="Show Track ID",
                    info="Display the tracker ID on detected license plates"
                )
                checkbox_show_fps = gr.Checkbox(
                    value=cfg.SHOW_FPS,
                    label="Show FPS",
                    info="Show frames per second on the video"
                )
                checkbox_show_trajectory = gr.Checkbox(
                    value=cfg.SHOW_TRAJECTORY,
                    label="Show Trajectory",
                    info="Draw the trajectory of tracked objects"
                )
                slider_trajectory_length = gr.Slider(
                    minimum=1.0,
                    maximum=200.0,
                    value=cfg.TRAJECTORY_LENGTH,
                    step=1.0,
                    label="Trajectory length",
                    info="Maximum length of the trajectory line for each object",
                    visible=cfg.SHOW_TRAJECTORY,
                    interactive=cfg.SHOW_TRAJECTORY
                )
            # Wire up callbacks ----------------------------------------------------
            slider_detection_threshold.change(update_cfg("YOLO_THRESHOLD"), slider_detection_threshold, None)
            slider_ocr_threshold.change(update_cfg("OCR_CONFIDENCE_THRESHOLD"), slider_ocr_threshold, None)
            slider_tracking_cosine_distance_threshold.change(update_cfg("COSINE_DISTANCE_THRESHOLD"), slider_tracking_cosine_distance_threshold, None)
            slider_tracking_min_num_detections.change(update_cfg("N_INIT"), slider_tracking_min_num_detections, None)
            slider_tracking_max_frames.change(update_cfg("MAX_AGE"), slider_tracking_max_frames, None)
            checkbox_show_track_id.change(update_cfg("SHOW_TRACKER_ID"), checkbox_show_track_id, None)
            checkbox_show_fps.change(update_cfg("SHOW_FPS"), checkbox_show_fps, None)
            checkbox_show_trajectory.change(toggle_trajectory_slider, checkbox_show_trajectory, slider_trajectory_length)
            slider_trajectory_length.change(update_cfg("TRAJECTORY_LENGTH"), slider_trajectory_length, None)
            slider_model_type.change(on_model_type_change, slider_model_type, None)
            # ---------------------------------------------------------------------

        # ----------------- Enable / Disable settings during processing ------------------
        

        # Components that should be disabled while processing is running
        settings_components = [
            slider_model_type,
            slider_detection_threshold,
            slider_ocr_threshold,
            slider_tracking_cosine_distance_threshold,
            slider_tracking_min_num_detections,
            slider_tracking_max_frames,
            checkbox_show_track_id,
            checkbox_show_fps,
            checkbox_show_trajectory,
            slider_trajectory_length,
            input_source,
            input_video,
        ]

        def _disable_settings():
            """Disable all settings components."""
            return [gr.update(interactive=False) for _ in settings_components]

        # ----------------- Wire Run button chain ------------------
        run_event = (
            run_button.click(_disable_settings, None, settings_components, queue=False)
            .then(lambda: gr.update(interactive=False), None, run_button, queue=False)
            .then(lambda: gr.update(visible=True), None, output_image, queue=False)
            .then(lambda: gr.update(visible=True), None, stop_button, queue=False)
            .then(process_video, inputs=input_video, outputs=output_image, queue=True)
            .then(lambda: gr.update(interactive=True), None, run_button, queue=False)
        )

        # Enable queuing so that generator outputs are streamed properly.
        demo.queue()

    return demo
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

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Generator

if TYPE_CHECKING:
    from app import LicensePlateDetectorApp

import cv2
import gradio as gr
import threading
import numpy as np

import config as cfg
from ui.drawing import FPSCounter, draw_fps_on_frame

logger = logging.getLogger(__name__)

stop_event = (
    threading.Event()
)  # Global event to signal the processing loop to stop gracefully when the Stop Button is pressed


def get_available_sources() -> list[str]:
    """Get available input sources based on system configuration.
    
    Returns:
        List of available input source options. Always includes 'Video File',
        and includes 'Webcam' if a webcam is detected.
    """
    import os
    webcam_available = os.path.exists('/dev/video0')
    return ["Video File", "Webcam"] if webcam_available else ["Video File"]


def _prepare_app() -> "LicensePlateDetectorApp":
    """Create an app instance based on the current `cfg` settings."""
    from app import LicensePlateDetectorApp  # Local import to break circularity

    # The cfg is assumed to be set correctly by the UI handlers.
    return LicensePlateDetectorApp()


def process_video() -> Generator[np.ndarray, None, None]:
    """Yield processed frames to Gradio based on the current `cfg` settings."""
    stop_event.clear()

    # The `cfg` module is the single source of truth for settings. It's
    # updated by the various UI component callbacks.
    app = _prepare_app()
    app.setup_video(cfg.WEBCAM, cfg.INPUT_VIDEO)

    if app.video_handler is None:
        yield np.zeros((100, 100, 3), dtype=np.uint8)  # Informative blank frame
        return

    fps_counter = FPSCounter()

    while True:
        # Check if the user pressed *Stop*
        if stop_event.is_set():
            logger.info("Stop event detected – exiting processing loop.")
            break

        ret, frame = app.video_handler.read_frame()
        if not ret or frame is None:
            break

        assert frame is not None

        detections = app.yolo_detector.detect(frame)
        if detections:
            app.tracker.process_detections(detections, frame, app.ocr_reader.reader)

        if cfg.SHOW_FPS:
            fps_counter.update()
            draw_fps_on_frame(frame, fps_counter.get_fps())

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame_rgb

    app.video_handler.release()

    return


def build_interface() -> gr.Blocks:
    """Construct and return the Gradio UI for the license-plate detector."""

    # Default to demo video unless the user selects otherwise.
    cfg.WEBCAM = False

    with gr.Blocks(title="License Plate Detector", theme="base") as demo:
        # ------------------------------ INPUT VIDEO VISUALIZER ------------------------------
        # Shows the processed video. Is visible only when processing.
        output_image = gr.Image(
            show_label=False, interactive=False, visible=False, height=480
        )
        with gr.Row(equal_height=True, min_height=250):
            # ------------------------------ SOURCE SELECTOR ------------------------------
            # Allows to select between webcam and video file.

            with gr.Column(scale=10):
                input_source = gr.Radio(
                    choices=get_available_sources(),
                    label="Input Source",
                    value="Video File",
                    info="If available, webcam can be selected",
                )

            # ------------------------------ FILE SELECTOR ------------------------------
            # If source is "Video File", shows a file uploader.

            with gr.Column(scale=1):
                input_video = gr.File(
                    file_types=["video"],
                    visible=True,
                    value=cfg.INPUT_VIDEO,
                    type="filepath",
                )

                def _on_source_change(src):
                    webcam = src == "Webcam"
                    cfg.WEBCAM = webcam
                    # Show uploader only for video file
                    return gr.update(visible=not webcam)

                input_source.change(_on_source_change, input_source, input_video)

                def _on_file_selected(file):
                    if file is None:
                        return gr.update()
                    path = (
                        file
                        if isinstance(file, str)
                        else file.get("name") if isinstance(file, dict) else None
                    )
                    if path:
                        cfg.WEBCAM = False
                        cfg.INPUT_VIDEO = path
                    # Ensure the radio is set to "Video File" (needed when a video file is selected via examples)
                    return gr.update(value="Video File")

                input_video.change(_on_file_selected, input_video, input_source)

            # ------------------------------ BUTTONS ------------------------------
            # Buttons start and stop the processing, and disable/enable the settings. Only one button is visible at a time.
            # The run and stop button events are defined in a later section, as other components need to be defined first.

            with gr.Column(scale=10):
                stop_button = gr.Button(
                    "Stop", elem_id="stop_button", variant="stop", visible=False
                )
                run_button = gr.Button(
                    "Run", elem_id="run_button", variant="primary", visible=True
                )

        # ------------------------------ SETTINGS ------------------------------
        # Settings control the behaviour of the program. They are grouped in an accordion that can be expanded/collapsed.
        # The callbacks update the cfg module, which is the single source of truth for settings.
        # Settings are organized into Tabs

        def update_cfg(attr: str):
            """Create a callback that writes *value* into ``cfg.attr``."""

            def _inner(value):
                setattr(cfg, attr, value)
                return None

            return _inner

        with gr.Row():
            with gr.Column(scale=4):
                with gr.Accordion("Settings 🛠️", open=False):
                    with gr.Tab("System"):
                        with gr.Row():
                            with gr.Column(scale=1, min_width=300):
                                # DETECTION SETTINGS
                                gr.Markdown(value="Detection")

                                # Model Type
                                slider_model_type = gr.Dropdown(
                                    choices=["PyTorch", "NCNN"],
                                    label="Model Type",
                                    value=(
                                        "PyTorch"
                                        if cfg.MODEL_PATH.endswith(".pt")
                                        else "NCNN"
                                    ),
                                    multiselect=False,
                                    info="Select the backend for the YOLO model",
                                )

                                def on_model_type_change(model_type: str):
                                    cfg.MODEL_PATH = (
                                        "./models/car_plate.pt"
                                        if model_type == "PyTorch"
                                        else "./models/best_ncnn_model"
                                    )

                                slider_model_type.change(
                                    on_model_type_change, slider_model_type, None
                                )

                                # Detection Threshold
                                slider_detection_threshold = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=cfg.YOLO_THRESHOLD,
                                    step=0.01,
                                    show_reset_button=True,
                                    label="Confidence threshold",
                                    info="Minimum confidence required for object detection",
                                )
                                slider_detection_threshold.change(
                                    update_cfg("YOLO_THRESHOLD"),
                                    slider_detection_threshold,
                                    None,
                                )

                                # OCR SETTINGS
                                gr.Markdown(value="Optical Character Recognition (OCR)")

                                # OCR Threshold
                                slider_ocr_threshold = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=cfg.OCR_CONFIDENCE_THRESHOLD,
                                    step=0.01,
                                    label="Confidence threshold",
                                    show_reset_button=True,
                                    info="Minimum confidence required for OCR recognition",
                                )
                                slider_ocr_threshold.change(
                                    update_cfg("OCR_CONFIDENCE_THRESHOLD"),
                                    slider_ocr_threshold,
                                    None,
                                )

                            with gr.Column(scale=1, min_width=300):
                                # TRACKING SETTINGS
                                gr.Markdown(value="Tracking")

                                # Cosine Distance Threshold
                                slider_tracking_cosine_distance_threshold = gr.Slider(
                                    label="Cosine Distance Threshold",
                                    value=cfg.COSINE_DISTANCE_THRESHOLD,
                                    maximum=1.0,
                                    info="Threshold for object re-identification in tracking",
                                )
                                slider_tracking_cosine_distance_threshold.change(
                                    update_cfg("COSINE_DISTANCE_THRESHOLD"),
                                    slider_tracking_cosine_distance_threshold,
                                    None,
                                )

                                # Min. num. detections
                                slider_tracking_min_num_detections = gr.Slider(
                                    minimum=1.0,
                                    maximum=20.0,
                                    value=cfg.N_INIT,
                                    step=1.0,
                                    label="Min. num. detections",
                                    show_reset_button=True,
                                    info="Detections required to confirm a new track",
                                )
                                slider_tracking_min_num_detections.change(
                                    update_cfg("N_INIT"),
                                    slider_tracking_min_num_detections,
                                    None,
                                )

                                # Max. frames
                                slider_tracking_max_frames = gr.Slider(
                                    minimum=1.0,
                                    maximum=3600.0,
                                    value=cfg.MAX_AGE,
                                    step=1.0,
                                    label="Max. frames",
                                    info="Frames to keep a track without new detections",
                                )
                                slider_tracking_max_frames.change(
                                    update_cfg("MAX_AGE"),
                                    slider_tracking_max_frames,
                                    None,
                                )

                    with gr.Tab("Visualization"):
                        # Show Track ID
                        checkbox_show_track_id = gr.Checkbox(
                            value=cfg.SHOW_TRACKER_ID,
                            label="Show Track ID",
                            info="Display the tracker ID on detected license plates",
                        )
                        checkbox_show_track_id.change(
                            update_cfg("SHOW_TRACKER_ID"), checkbox_show_track_id, None
                        )

                        # Show FPS
                        checkbox_show_fps = gr.Checkbox(
                            value=cfg.SHOW_FPS,
                            label="Show FPS",
                            info="Show frames per second on the video",
                        )
                        checkbox_show_fps.change(
                            update_cfg("SHOW_FPS"), checkbox_show_fps, None
                        )

                        # Show Trajectory
                        # The callback is defined in slider_trajectory_length (dependency)
                        checkbox_show_trajectory = gr.Checkbox(
                            value=cfg.SHOW_TRAJECTORY,
                            label="Show Trajectory",
                            info="Draw the trajectory of tracked objects",
                        )

                        # Trajectory Length
                        slider_trajectory_length = gr.Slider(
                            minimum=1.0,
                            maximum=200.0,
                            value=cfg.TRAJECTORY_LENGTH,
                            step=1.0,
                            label="Trajectory length",
                            info="Maximum length of the trajectory line for each object",
                            visible=cfg.SHOW_TRAJECTORY,
                            interactive=cfg.SHOW_TRAJECTORY,
                        )
                        slider_trajectory_length.change(
                            update_cfg("TRAJECTORY_LENGTH"),
                            slider_trajectory_length,
                            None,
                        )

                        def toggle_trajectory_slider(show: bool):
                            """
                            If Show Trajectory is selected, then the trajectory length slider is enabled.
                            Otherwise, the slider is disabled.
                            """
                            cfg.SHOW_TRAJECTORY = show
                            return gr.update(visible=show, interactive=show)

                        checkbox_show_trajectory.change(
                            toggle_trajectory_slider,
                            checkbox_show_trajectory,
                            slider_trajectory_length,
                        )

            # ------------------------------ DARK MODE ------------------------------
            # Dark Mode
            with gr.Column(scale=1):
                dark_mode = gr.Checkbox(label="Dark mode", value=True, interactive=True)
                # Bind checkbox to toggle dark mode using JavaScript
                dark_mode.change(
                    None,
                    dark_mode,
                    None,
                    js="""
                    (checked) => {
                        if (checked) {
                            document.body.classList.add('dark');
                        } else {
                            document.body.classList.remove('dark');
                        }
                        const bg = getComputedStyle(document.documentElement).getPropertyValue('--color-background-primary');
                        document.body.style.backgroundColor = bg;
                    }
                    """,
                )

                # Ensure dark mode is applied on initial interface load
                demo.load(
                    None,
                    None,
                    None,
                    js="""
                    () => {
                        document.body.classList.add('dark');
                        const bg = getComputedStyle(document.documentElement).getPropertyValue('--color-background-primary');
                        document.body.style.backgroundColor = bg;
                    }
                    """,
                )

        # ------------------------------ EXAMPLES ------------------------------
        # Allow to select an example video to run the program.
        gr.Examples(
            examples=[["./data/test_video_1.mp4"], ["./data/test_video_2.mp4"]],
            inputs=input_video,
            label="Example Videos",
        )

        # ------------------------------ RUN/STOP EVENTS ------------------------------
        # Actions triggered after Run and Stop button are pressed

        # Components that are disabled while processing is running
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

        def log_run():
            """Log current configuration when *Run* is pressed."""
            current_settings = {
                "INPUT_VIDEO": cfg.INPUT_VIDEO,
                "WEBCAM": cfg.WEBCAM,
                "MODEL_PATH": cfg.MODEL_PATH,
                "YOLO_THRESHOLD": cfg.YOLO_THRESHOLD,
                "OCR_CONFIDENCE_THRESHOLD": cfg.OCR_CONFIDENCE_THRESHOLD,
                "COSINE_DISTANCE_THRESHOLD": cfg.COSINE_DISTANCE_THRESHOLD,
                "N_INIT": cfg.N_INIT,
                "MAX_AGE": cfg.MAX_AGE,
                "SHOW_TRACKER_ID": cfg.SHOW_TRACKER_ID,
                "SHOW_FPS": cfg.SHOW_FPS,
                "SHOW_TRAJECTORY": cfg.SHOW_TRAJECTORY,
                "TRAJECTORY_LENGTH": cfg.TRAJECTORY_LENGTH,
            }
            logger.info("Run button clicked. Current settings: %s", current_settings)
            return None

        def _disable_settings():
            return [gr.update(interactive=False) for _ in settings_components]

        def _enable_settings():
            return [gr.update(interactive=True) for _ in settings_components]

        # Actions to perform when *Run* button is clicked
        run_click_event = (
            run_button.click(log_run, None, None, queue=False)
            .then(_disable_settings, None, settings_components, queue=False)
            .then(lambda: gr.update(visible=True), None, output_image, queue=False)
            .then(
                lambda: gr.update(interactive=False, visible=False),
                None,
                run_button,
                queue=False,
            )
            .then(lambda: gr.update(visible=True), None, stop_button, queue=False)
            .then(process_video, inputs=None, outputs=output_image, queue=True)
        )

        def stop_process():
            """Callback for the *Stop* button – requests the loop to terminate."""
            logger.info("Stop button clicked – signaling processing loop to stop.")
            stop_event.set()

        # Actions to perform when *Stop* button is clicked
        stop_click_event = (
            stop_button.click(stop_process, None, None, queue=False)
            .then(_enable_settings, None, settings_components, queue=False)
            .then(lambda: gr.update(visible=False), None, output_image, queue=False)
            .then(lambda: gr.update(value=None), None, input_video, queue=False)
            .then(
                lambda: gr.update(interactive=True, visible=True),
                None,
                run_button,
                queue=False,
            )
            .then(lambda: gr.update(visible=False), None, stop_button, queue=False)
        )

        # ------------------------------ QUEUE ------------------------------
        # Enable queuing so that generator outputs are streamed properly.
        demo.queue()

    return demo

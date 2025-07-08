"""User interface components for the license plate detection system.

This package provides the graphical user interface for interacting with the
license plate detection system, built using Gradio.

"""

from .gradio_ui import build_interface
from .drawing import draw_fps_on_frame, FPSCounter
from .interface import VideoInterface

__all__ = ['build_interface', 'draw_fps_on_frame', 'FPSCounter', 'VideoInterface']